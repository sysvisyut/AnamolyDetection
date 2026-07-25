"""
Session Builder — groups raw access log records into per-entity sessions and
maintains the sliding state required for sequence-level feature extraction.

ARCHITECTURE COMPONENT: Feature Engineering
BOUNDARY RESPONSIBILITY: Consumes B (AccessLogInference / AccessLogTraining);
                         contributes to producing C (EngineeredFeatures).
TIER: T1

The SessionBuilder is responsible for:
  1. Grouping raw access-log records by (entity_id, session_id), sorted
     chronologically within each session.
  2. Maintaining per-entity event deques (depth = sequence_window_size) so
     the FeatureExtractor can construct the SDM sequence window without
     rescanning all history.
  3. Tracking sliding 24-hour windows for cross-entity IP ratio features
     (dims 22, 23) which require counting events per IP and per entity.
  4. Tracking per-session state (event_count, distinct_resources) needed for
     dims 20 and 21.

Label leakage prevention:
    The SessionBuilder accepts AccessLogBase (base class) and operates
    solely on the non-label fields. AccessLogTraining's label field is
    never read here. The FeaturePipeline wraps input records into
    AccessLogBase objects before passing them to the builder.
"""

from __future__ import annotations

import datetime
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from anomaly_detection.common.models.access_log import AccessLogBase, GeoLocation
from anomaly_detection.feature_engineering.config import (
    FeatureEngineeringConfig,
    RATIO_WINDOW_SECONDS,
)


class _EntityState:
    """
    Per-entity mutable state maintained across events for feature computation.

    This class is internal to SessionBuilder; it is not part of the
    public API.
    """

    def __init__(self, window_size: int) -> None:
        # Deque of (timestamp_str, feature_vector) — populated by FeatureExtractor
        # after each event.  Here we only track raw event metadata.
        self.recent_events: Deque[AccessLogBase] = deque(maxlen=window_size)
        # Session-level state (reset when session_id changes)
        self.current_session_id: Optional[str] = None
        self.session_event_count: int = 0
        self.session_distinct_resources: Set[str] = set()
        # Last known geo-location and its timestamp for geo-velocity
        self.last_geo: Optional[GeoLocation] = None
        self.last_timestamp_dt: Optional[datetime.datetime] = None


class _IPWindow:
    """
    Sliding 24-hour event count window for a single source IP.

    Records the UTC timestamps of each event from this IP so stale
    entries can be expired before computing ip_entity_ratio (dim 22).
    """

    def __init__(self, window_seconds: float) -> None:
        self._window_seconds = window_seconds
        # List of (timestamp_dt, entity_id) tuples
        self._entries: List[Tuple[datetime.datetime, str]] = []

    def record(self, ts: datetime.datetime, entity_id: str) -> None:
        """Record a new event from this IP."""
        self._entries.append((ts, entity_id))

    def get_stats(
        self, now: datetime.datetime
    ) -> Tuple[int, Set[str]]:
        """
        Return (total_events_in_window, set_of_entity_ids_in_window) after
        expiring entries older than window_seconds.

        Args:
            now: Current event timestamp used as the reference point.

        Returns:
            (total_event_count, distinct_entity_ids_set)
        """
        cutoff = now - datetime.timedelta(seconds=self._window_seconds)
        self._entries = [(t, e) for t, e in self._entries if t >= cutoff]
        entity_ids = {e for _, e in self._entries}
        return (len(self._entries), entity_ids)


class _EntityIPWindow:
    """
    Sliding 24-hour distinct-IP window for a single entity.

    Tracks distinct source IPs seen for this entity within the rolling
    time window, enabling computation of entity_ip_ratio (dim 23).
    """

    def __init__(self, window_seconds: float) -> None:
        self._window_seconds = window_seconds
        # List of (timestamp_dt, source_ip) tuples
        self._entries: List[Tuple[datetime.datetime, str]] = []

    def record(self, ts: datetime.datetime, source_ip: str) -> None:
        """Record a new event for this entity."""
        self._entries.append((ts, source_ip))

    def get_stats(self, now: datetime.datetime) -> Tuple[int, int]:
        """
        Return (total_events_in_window, distinct_ip_count) after expiring stale entries.

        Args:
            now: Current event timestamp used as the reference point.

        Returns:
            (total_event_count, distinct_ip_count)
        """
        cutoff = now - datetime.timedelta(seconds=self._window_seconds)
        self._entries = [(t, ip) for t, ip in self._entries if t >= cutoff]
        distinct_ips = len({ip for _, ip in self._entries})
        return (len(self._entries), distinct_ips)


def _parse_timestamp(ts_str: str) -> datetime.datetime:
    """
    Parse an ISO-8601 UTC timestamp string into a timezone-aware datetime.

    Handles both ``+00:00`` and ``Z`` suffixes.

    Args:
        ts_str: ISO-8601 timestamp string from the event.

    Returns:
        UTC-aware ``datetime.datetime`` object.

    Raises:
        ValueError: If the timestamp string cannot be parsed.
    """
    # Python's fromisoformat does not handle 'Z' suffix before 3.11
    ts_normalised = ts_str.replace("Z", "+00:00")
    return datetime.datetime.fromisoformat(ts_normalised)


class SessionBuilder:
    """
    Groups raw access-log records into per-entity sessions and tracks
    all state needed for sequence-level feature extraction.

    Consumes boundary B (AccessLogBase / InboundEvent).
    Maintains per-entity deques, session state, and sliding 24-hour windows.
    Does NOT compute feature vectors — that is FeatureExtractor's job.

    Usage (batch mode):
        builder = SessionBuilder(config)
        # Process all events chronologically
        for event in sorted_events:
            ctx = builder.process_event(event)
            feature_vector = extractor.extract(event, ctx, profile)

    Usage (single-record inference mode):
        ctx = builder.process_event(single_event)
        feature_vector = extractor.extract(single_event, ctx, profile)
    """

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        """
        Initialise the SessionBuilder with the given pipeline configuration.

        Args:
            config: Fully constructed ``FeatureEngineeringConfig`` object.
        """
        self._config = config
        self._window_size = config.sequence_window_size
        self._ratio_window_seconds = config.ratio_window_seconds

        # Per-entity state (keyed by entity_id)
        self._entity_states: Dict[str, _EntityState] = {}

        # IP-level sliding windows (keyed by source_ip)
        self._ip_windows: Dict[str, _IPWindow] = defaultdict(
            lambda: _IPWindow(self._ratio_window_seconds)
        )

        # Entity-level IP sliding windows (keyed by entity_id)
        self._entity_ip_windows: Dict[str, _EntityIPWindow] = defaultdict(
            lambda: _EntityIPWindow(self._ratio_window_seconds)
        )

    def reset(self) -> None:
        """
        Clear all accumulated state.

        Call between training runs or when switching datasets to avoid
        state bleed between batches.
        """
        self._entity_states.clear()
        self._ip_windows.clear()
        self._entity_ip_windows.clear()

    def _get_or_create_entity_state(self, entity_id: str) -> _EntityState:
        """
        Retrieve the existing entity state or create a fresh one.

        Args:
            entity_id: Entity identifier string.

        Returns:
            The ``_EntityState`` for this entity.
        """
        if entity_id not in self._entity_states:
            self._entity_states[entity_id] = _EntityState(self._window_size)
        return self._entity_states[entity_id]

    def process_event(self, event: AccessLogBase) -> "EventContext":
        """
        Process a single access-log event and return the computed context
        needed by FeatureExtractor.

        This is the primary entry point for both batch and single-record
        inference modes. It updates all sliding windows and session counters
        before returning the context object.

        IMPORTANT: Events must be processed in chronological order per entity.
        In batch mode, sort the DataFrame by timestamp before calling this.

        Args:
            event: A raw access-log record (AccessLogBase or any subclass).
                   The ``label`` field from AccessLogTraining is never read here.

        Returns:
            An ``EventContext`` containing all pre-computed state needed by
            FeatureExtractor for this event.

        Label leakage prevention:
            This method only reads fields defined on ``AccessLogBase``.
            The ``label`` field on ``AccessLogTraining`` is structurally
            present but never accessed.
        """
        entity_id = event.entity_id
        session_id = event.session_id
        ts = _parse_timestamp(event.timestamp)

        state = self._get_or_create_entity_state(entity_id)

        # ── Session state management ───────────────────────────────────────
        if state.current_session_id != session_id:
            # New session: reset session-level counters
            state.current_session_id = session_id
            state.session_event_count = 0
            state.session_distinct_resources.clear()

        state.session_event_count += 1
        state.session_distinct_resources.add(event.resource_accessed)

        # ── Inter-event gap ────────────────────────────────────────────────
        if state.last_timestamp_dt is not None:
            gap_seconds = (ts - state.last_timestamp_dt).total_seconds()
            # Guard against clock skew producing negative gaps
            gap_seconds = max(0.0, gap_seconds)
        else:
            gap_seconds = 0.0  # First event: no previous event

        # ── Geo-velocity context ───────────────────────────────────────────
        prev_geo = state.last_geo
        prev_timestamp = state.last_timestamp_dt
        geo_elapsed_seconds = gap_seconds  # Same as inter-event gap

        # ── Sliding 24h window updates ─────────────────────────────────────
        ip_win = self._ip_windows[event.source_ip]
        ip_win.record(ts, entity_id)
        ip_total, ip_entities = ip_win.get_stats(ts)

        entity_ip_win = self._entity_ip_windows[entity_id]
        entity_ip_win.record(ts, event.source_ip)
        entity_total, distinct_ip_count = entity_ip_win.get_stats(ts)

        # Number of events from this specific IP attributable to THIS entity
        # in the past 24h window = events recorded from this IP where entity_id
        # matches. We approximate: all ip_total events are from this IP.
        # The ratio is ip_total / entity_total (events from IP / entity events).
        events_from_ip_for_entity = ip_total

        # ── Update persistent state ────────────────────────────────────────
        state.last_geo = event.geo_location
        state.last_timestamp_dt = ts
        # Append current raw event to the entity's event deque
        state.recent_events.append(event)

        # Produces EventContext (consumed by FeatureExtractor)
        return EventContext(
            entity_id=entity_id,
            session_id=session_id,
            timestamp_dt=ts,
            gap_seconds=gap_seconds,
            prev_geo=prev_geo,
            geo_elapsed_seconds=geo_elapsed_seconds,
            session_event_count=state.session_event_count,
            session_distinct_resource_count=len(state.session_distinct_resources),
            events_from_ip_in_window=events_from_ip_for_entity,
            total_events_for_entity_in_window=entity_total,
            distinct_ips_for_entity_in_window=distinct_ip_count,
        )

    def get_recent_events(self, entity_id: str) -> List[AccessLogBase]:
        """
        Retrieve the most recent W raw events for an entity.

        Args:
            entity_id: Entity identifier.

        Returns:
            Ordered list (oldest first) of recent ``AccessLogBase`` records;
            empty list if the entity has not been seen yet.
        """
        state = self._entity_states.get(entity_id)
        if state is None:
            return []
        return list(state.recent_events)

    def get_entity_ids(self) -> List[str]:
        """
        Return all entity IDs that have been processed.

        Returns:
            Sorted list of entity identifier strings.
        """
        return sorted(self._entity_states.keys())

    @staticmethod
    def sort_events_chronologically(
        events: List[AccessLogBase],
    ) -> List[AccessLogBase]:
        """
        Sort a list of access-log events by timestamp in ascending order.

        Must be called on the full batch before calling process_event
        in batch mode to ensure correct inter-event gap computation.

        Args:
            events: Unsorted list of raw access-log records.

        Returns:
            New list sorted by ``timestamp`` in ascending order.
        """
        return sorted(events, key=lambda e: e.timestamp)


class EventContext:
    """
    Pre-computed contextual data for a single event, produced by SessionBuilder
    and consumed by FeatureExtractor.

    All fields are label-free and derived purely from raw event data plus
    the rolling state maintained by SessionBuilder.
    """

    __slots__ = (
        "entity_id",
        "session_id",
        "timestamp_dt",
        "gap_seconds",
        "prev_geo",
        "geo_elapsed_seconds",
        "session_event_count",
        "session_distinct_resource_count",
        "events_from_ip_in_window",
        "total_events_for_entity_in_window",
        "distinct_ips_for_entity_in_window",
    )

    def __init__(
        self,
        entity_id: str,
        session_id: str,
        timestamp_dt: datetime.datetime,
        gap_seconds: float,
        prev_geo: Optional[GeoLocation],
        geo_elapsed_seconds: float,
        session_event_count: int,
        session_distinct_resource_count: int,
        events_from_ip_in_window: int,
        total_events_for_entity_in_window: int,
        distinct_ips_for_entity_in_window: int,
    ) -> None:
        self.entity_id = entity_id
        self.session_id = session_id
        self.timestamp_dt = timestamp_dt
        self.gap_seconds = gap_seconds
        self.prev_geo = prev_geo
        self.geo_elapsed_seconds = geo_elapsed_seconds
        self.session_event_count = session_event_count
        self.session_distinct_resource_count = session_distinct_resource_count
        self.events_from_ip_in_window = events_from_ip_in_window
        self.total_events_for_entity_in_window = total_events_for_entity_in_window
        self.distinct_ips_for_entity_in_window = distinct_ips_for_entity_in_window
