"""
Feature Extractor — computes the full 24-dimensional feature vector for a
single access-log event given its EventContext and EntityProfile.

ARCHITECTURE COMPONENT: Feature Engineering
BOUNDARY RESPONSIBILITY: Consumes B (AccessLogBase) + E (EntityProfile);
                         produces the feature_vector component of C (EngineeredFeatures).
TIER: T1

The FeatureExtractor is a pure (stateless) computation layer. All mutable
state lives in SessionBuilder (sliding windows) or EntityProfile (profile store).
The extractor simply combines inputs → outputs a length-24 list[float].

Label leakage prevention:
    The extractor accepts AccessLogBase (label-free). The label field on
    AccessLogTraining is structurally invisible at this layer because we
    always pass the base class. The ProfileStoreInterface.get_profile()
    contract forbids storing label information in EntityProfile.

Cold-start path:
    When profile is None (brand-new entity) or profile.cold_start_flag is True
    (fewer than MIN_PROFILE_EVENTS events), the extractor substitutes
    population-level fallback statistics from FeatureEngineeringConfig for
    all profile-dependent features. No NaN, Inf, or missing values are produced.
"""

from __future__ import annotations

import datetime
from typing import List, Optional

from anomaly_detection.common.models.access_log import AccessLogBase
from anomaly_detection.common.models.entities import EntityProfile
from anomaly_detection.feature_engineering.config import (
    COLD_START_COMMAND_RARITY_NEUTRAL,
    COLD_START_FINGERPRINT_NEUTRAL,
    FEATURE_DIM,
    MAX_INTER_EVENT_GAP_SECONDS,
    FeatureEngineeringConfig,
)
from anomaly_detection.feature_engineering.encoders import (
    encode_auth_method,
    encode_auth_outcome,
    encode_command_rarity,
    encode_command_seq_length,
    encode_day_of_week,
    encode_entity_ip_ratio,
    encode_entity_type,
    encode_failure_count,
    encode_fingerprint_mac_match,
    encode_fingerprint_os_match,
    encode_fingerprint_protocol_match,
    encode_has_exfil_command,
    encode_hour_of_day,
    encode_inter_event_gap,
    encode_ip_entity_ratio,
    encode_is_new_geo,
    encode_resource_breadth,
    encode_resource_category,
    encode_resource_rarity,
    encode_session_event_count,
)
from anomaly_detection.feature_engineering.geo_velocity import compute_geo_velocity
from anomaly_detection.feature_engineering.session_builder import EventContext


class FeatureExtractor:
    """
    Computes the 24-dimensional feature vector for a single access-log event.

    Stateless between calls — all state is in EventContext (from SessionBuilder)
    and EntityProfile (from the ProfileStore).  Accepts a ProfileStoreInterface
    via constructor injection for testability.

    The extractor is designed so that:
      1. No feature dimension reads from the ``label`` field (directly or
         indirectly).
      2. Cold-start entities produce valid, non-degenerate vectors using
         population fallback statistics from FeatureEngineeringConfig.
      3. Every dimension has a documented attack-signal rationale.

    Consumes boundary B (AccessLogBase) and boundary E (EntityProfile).
    Contributes to producing boundary C (EngineeredFeatures).
    """

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        """
        Initialise the FeatureExtractor with the pipeline configuration.

        Args:
            config: Fully constructed ``FeatureEngineeringConfig`` with
                population fallback statistics populated by
                ``FeaturePipeline.fit()``.
        """
        self._config = config

    def extract(
        self,
        event: AccessLogBase,
        ctx: EventContext,
        profile: Optional[EntityProfile],
    ) -> List[float]:
        """
        Compute the 24-dimensional feature vector for a single event.

        Args:
            event: Raw access-log record (AccessLogBase or any subclass;
                   the ``label`` field is never accessed here).
            ctx: EventContext computed by ``SessionBuilder.process_event()``.
            profile: The entity's current ``EntityProfile`` from the profile
                     store, or ``None`` if the entity is brand-new (cold-start).

        Returns:
            A list of 24 floats representing the feature vector for this event.
            All values are in their documented ranges (no NaN, Inf, or missing).

        Label leakage prevention:
            This method only accesses fields on ``AccessLogBase`` (the base
            class). The ``label`` field exists only on ``AccessLogTraining``
            and is structurally inaccessible here.
        """
        ts_dt: datetime.datetime = ctx.timestamp_dt
        is_cold_start = (profile is None or profile.cold_start_flag)

        # ── dims 0–1: hour_of_day circular encoding ────────────────────────
        hour_sin, hour_cos = encode_hour_of_day(ts_dt.hour)

        # ── dims 2–3: day_of_week circular encoding ────────────────────────
        dow_sin, dow_cos = encode_day_of_week(ts_dt.weekday())

        # ── dim 4: session_duration_norm ───────────────────────────────────
        # Normalise session_duration against entity's historical distribution.
        # Cold-start: use population mean/std from config.
        session_duration_norm = self._compute_session_duration_norm(
            event.session_duration, profile, is_cold_start
        )

        # ── dim 5: failure_count_norm ──────────────────────────────────────
        failure_count_norm = encode_failure_count(event.failure_count)

        # ── dims 6–7: geo features ─────────────────────────────────────────
        if self._config.enable_geo_features:
            _, geo_velocity_norm = compute_geo_velocity(
                prev_geo=ctx.prev_geo,
                curr_geo=event.geo_location,
                elapsed_seconds=ctx.geo_elapsed_seconds,
            )
            is_new_geo = encode_is_new_geo(
                current_country=event.geo_location.country,
                most_frequent_country=(
                    profile.most_frequent_country if profile else None
                ),
            )
        else:
            geo_velocity_norm = 0.0
            is_new_geo = 0.0

        # ── dim 8: resource_category_enc ──────────────────────────────────
        resource_category_enc = encode_resource_category(event.resource_accessed)

        # ── dim 9: resource_rarity_score ──────────────────────────────────
        resource_rarity_score = encode_resource_rarity(
            resource_accessed=event.resource_accessed,
            resource_access_counts=(
                profile.resource_access_counts if profile else None
            ),
            total_entity_events=profile.event_count if profile else 0,
        )

        # ── dim 10: auth_method_enc ────────────────────────────────────────
        auth_method_enc = encode_auth_method(event.auth_method)

        # ── dim 11: auth_outcome_enc ───────────────────────────────────────
        auth_outcome_enc = encode_auth_outcome(event.auth_outcome)

        # ── dim 12: command_seq_length_norm ───────────────────────────────
        if self._config.enable_command_features:
            command_seq_length_norm = encode_command_seq_length(event.command_sequence)
        else:
            command_seq_length_norm = 0.0

        # ── dim 13: command_rarity_score ───────────────────────────────────
        if self._config.enable_command_features:
            command_rarity_score = encode_command_rarity(
                command_sequence=event.command_sequence,
                command_frequency=(
                    profile.command_frequency if profile else None
                ),
                total_entity_events=profile.event_count if profile else 0,
                cold_start_default=COLD_START_COMMAND_RARITY_NEUTRAL,
            )
        else:
            command_rarity_score = 0.0

        # ── dim 14: has_exfil_command ──────────────────────────────────────
        if self._config.enable_command_features:
            has_exfil_command = encode_has_exfil_command(event.command_sequence)
        else:
            has_exfil_command = 0.0

        # ── dims 15–17: device fingerprint match ───────────────────────────
        if self._config.enable_device_features and profile is not None:
            known_os_profiles = profile.known_os_profiles
            known_macs = profile.known_mac_addresses
            known_protocols = profile.known_protocols
        else:
            # Cold-start or device features disabled → neutral 0.5
            known_os_profiles = None
            known_macs = None
            known_protocols = None

        fingerprint_os_match = encode_fingerprint_os_match(
            os_family=event.device_fingerprint.os_family,
            os_version=event.device_fingerprint.os_version,
            known_os_profiles=known_os_profiles,
            cold_start_default=COLD_START_FINGERPRINT_NEUTRAL,
        )
        fingerprint_mac_match = encode_fingerprint_mac_match(
            mac_address=event.device_fingerprint.mac_address,
            known_mac_addresses=known_macs,
            cold_start_default=COLD_START_FINGERPRINT_NEUTRAL,
        )
        fingerprint_protocol_match = encode_fingerprint_protocol_match(
            protocol=event.device_fingerprint.protocol,
            known_protocols=known_protocols,
            cold_start_default=COLD_START_FINGERPRINT_NEUTRAL,
        )

        # ── dim 18: entity_type_enc ────────────────────────────────────────
        entity_type_enc = encode_entity_type(event.entity_type)

        # ── dim 19: inter_event_gap_norm ───────────────────────────────────
        inter_event_gap_norm = encode_inter_event_gap(ctx.gap_seconds)

        # ── dim 20: session_event_count_norm ──────────────────────────────
        session_event_count_norm = encode_session_event_count(
            ctx.session_event_count
        )

        # ── dim 21: resource_breadth_norm ─────────────────────────────────
        resource_breadth_norm = encode_resource_breadth(
            ctx.session_distinct_resource_count
        )

        # ── dims 22–23: IP / entity ratio features ────────────────────────
        if self._config.enable_ratio_features:
            ip_entity_ratio = encode_ip_entity_ratio(
                events_from_ip_in_window=ctx.events_from_ip_in_window,
                total_events_for_entity_in_window=ctx.total_events_for_entity_in_window,
            )
            entity_ip_ratio = encode_entity_ip_ratio(
                distinct_ips_for_entity_in_window=ctx.distinct_ips_for_entity_in_window,
                total_events_for_entity_in_window=ctx.total_events_for_entity_in_window,
            )
        else:
            ip_entity_ratio = 0.0
            entity_ip_ratio = 0.0

        # ── Assemble the 24-dimensional feature vector ─────────────────────
        feature_vector: List[float] = [
            hour_sin,               # dim 0
            hour_cos,               # dim 1
            dow_sin,                # dim 2
            dow_cos,                # dim 3
            session_duration_norm,  # dim 4
            failure_count_norm,     # dim 5
            geo_velocity_norm,      # dim 6
            is_new_geo,             # dim 7
            resource_category_enc,  # dim 8
            resource_rarity_score,  # dim 9
            auth_method_enc,        # dim 10
            auth_outcome_enc,       # dim 11
            command_seq_length_norm, # dim 12
            command_rarity_score,   # dim 13
            has_exfil_command,      # dim 14
            fingerprint_os_match,   # dim 15
            fingerprint_mac_match,  # dim 16
            fingerprint_protocol_match, # dim 17
            entity_type_enc,        # dim 18
            inter_event_gap_norm,   # dim 19
            session_event_count_norm, # dim 20
            resource_breadth_norm,  # dim 21
            ip_entity_ratio,        # dim 22
            entity_ip_ratio,        # dim 23
        ]

        # Safety check: must always produce exactly FEATURE_DIM values
        assert len(feature_vector) == FEATURE_DIM, (
            f"Feature vector length {len(feature_vector)} != {FEATURE_DIM}"
        )

        return feature_vector

    def _compute_session_duration_norm(
        self,
        session_duration: float,
        profile: Optional[EntityProfile],
        is_cold_start: bool,
    ) -> float:
        """
        Normalise session_duration using entity historical min-max or population fallback.

        For entities with a profile: use min-max normalisation derived from the
        entity's baseline_vector[4] (the stored mean) and baseline_std[4].
        Specifically: normalise as ``min((value - mean) / (5 * std + ε), 1.0)``
        clipped to [0, 1] — so values within ±5σ of the entity's mean map to [0, 1].

        For cold-start: normalise against the population mean/std stored in config.

        Args:
            session_duration: Raw session duration in seconds.
            profile: Entity profile (may be None for cold-start).
            is_cold_start: Whether the entity is in cold-start state.

        Returns:
            Normalised float in [0, 1].

        Attack relevance:
            dim 4 — Low-and-Slow (unusually long sessions during off-hours).
        """
        from anomaly_detection.feature_engineering.config import BASELINE_STD_EPSILON

        if not is_cold_start and profile is not None and len(profile.baseline_vector) > 4:
            # Use entity-level baseline for normalisation
            mean_val = profile.baseline_vector[4]
            std_val = profile.baseline_std[4] if len(profile.baseline_std) > 4 else 1.0
            std_val = max(std_val, BASELINE_STD_EPSILON)
            # Normalise deviation: how many std deviations is this above the mean?
            # Cap at ±5σ range → [0, 1] where 0 = at/below mean, 1 = 5σ above mean
            deviation = (session_duration - mean_val) / (5.0 * std_val)
            return float(max(0.0, min(1.0, deviation)))
        else:
            # Cold-start: use population statistics from config
            mean_val = self._config.cold_start_session_duration_mean
            std_val = max(
                self._config.cold_start_session_duration_std,
                BASELINE_STD_EPSILON,
            )
            deviation = (session_duration - mean_val) / (5.0 * std_val)
            return float(max(0.0, min(1.0, deviation)))
