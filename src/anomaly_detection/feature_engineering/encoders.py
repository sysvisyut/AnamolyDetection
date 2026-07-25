"""
Categorical and numeric encoding helpers for the feature engineering pipeline.

ARCHITECTURE COMPONENT: Feature Engineering
BOUNDARY RESPONSIBILITY: Internal helper; does not cross a named boundary directly.
TIER: T1

Provides deterministic, label-leakage-free encodings for all categorical
fields that appear in the 24-dimensional feature vector. Every encoder
function is a pure function of its input fields only — no label or
ground-truth information is used or accessible.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from anomaly_detection.feature_engineering.config import (
    AUTH_METHOD_MAP,
    AUTH_METHOD_MAX,
    AUTH_OUTCOME_MAP,
    EXFIL_COMMANDS,
    MAX_COMMAND_SEQ_LENGTH,
    MAX_FAILURE_COUNT,
    MAX_IP_ENTITY_RATIO,
    MAX_RESOURCE_BREADTH,
    MAX_SESSION_EVENT_COUNT,
    RESOURCE_CATEGORY_MAP,
    RESOURCE_CATEGORY_MAX,
    RESOURCE_CATEGORY_OTHER,
    MAX_ENTITY_IP_RATIO,
)
from anomaly_detection.common.models.access_log import CommandEntry


# ---------------------------------------------------------------------------
# Temporal encodings (dims 0–3)
# ---------------------------------------------------------------------------


def encode_hour_of_day(hour: int) -> tuple[float, float]:
    """
    Circular encoding of the hour-of-day to avoid the 23→0 discontinuity.

    Args:
        hour: Integer hour in [0, 23].

    Returns:
        Tuple (sin_val, cos_val) both in [-1, 1].

    Attack relevance:
        dims 0–1 — Low-and-Slow off-hours access detection. Circular
        encoding ensures 23:00 and 01:00 are close in feature space.

    Derives from:
        ``timestamp`` field (hour extracted before calling this function).
    """
    angle = 2.0 * math.pi * hour / 24.0
    return (math.sin(angle), math.cos(angle))


def encode_day_of_week(weekday: int) -> tuple[float, float]:
    """
    Circular encoding of the day-of-week to avoid the Sunday→Monday discontinuity.

    Args:
        weekday: Integer in [0, 6] where 0 = Monday (Python datetime convention).

    Returns:
        Tuple (sin_val, cos_val) both in [-1, 1].

    Attack relevance:
        dims 2–3 — Weekend vs. weekday access baseline for Low-and-Slow
        and Insider Drift pattern detection.

    Derives from:
        ``timestamp`` field (weekday extracted before calling this function).
    """
    angle = 2.0 * math.pi * weekday / 7.0
    return (math.sin(angle), math.cos(angle))


# ---------------------------------------------------------------------------
# Auth encodings (dims 5, 10, 11)
# ---------------------------------------------------------------------------


def encode_failure_count(failure_count: int) -> float:
    """
    Normalise consecutive auth failure count to [0, 1].

    Formula: ``min(failure_count / 20, 1.0)``

    Args:
        failure_count: Raw consecutive auth failure count from the event.

    Returns:
        Normalised float in [0, 1]; 1.0 if failure_count ≥ 20.

    Attack relevance:
        dim 5 — Primary signal for Brute Force (burst of 10–50 failures)
        and secondary signal for Credential Stuffing (1–5 failures per entity).

    Derives from:
        ``failure_count`` field.
    """
    return min(float(failure_count) / MAX_FAILURE_COUNT, 1.0)


def encode_auth_method(auth_method: str) -> float:
    """
    Ordinal encoding of authentication method to [0, 1].

    Mapping: password=0, token=1, certificate=2, biometric=3, none=4;
    divided by 4.

    Args:
        auth_method: Raw auth method string from the event.

    Returns:
        Normalised float in {0.0, 0.25, 0.5, 0.75, 1.0} for known methods,
        or 0.0 for unknown (safe default).

    Attack relevance:
        dim 10 — Credential misuse: a switch from ``certificate`` (normal for
        service accounts) to ``password`` is anomalous.

    Derives from:
        ``auth_method`` field.
    """
    idx = AUTH_METHOD_MAP.get(auth_method, 0)
    return float(idx) / AUTH_METHOD_MAX


def encode_auth_outcome(auth_outcome: str) -> float:
    """
    Encode authentication outcome to a 3-valued ordinal feature.

    Mapping: success=0.0, mfa_required=0.5, failure=1.0

    Args:
        auth_outcome: Raw auth outcome string from the event.

    Returns:
        One of {0.0, 0.5, 1.0}; defaults to 0.0 for unknown values.

    Attack relevance:
        dim 11 — Primary signal for Brute Force and Credential Stuffing.
        Failure=1.0 is the peak value; repeated 1.0 values in a sequence
        create a strong SDM reconstruction-error spike.

    Derives from:
        ``auth_outcome`` field.
    """
    return AUTH_OUTCOME_MAP.get(auth_outcome, 0.0)


# ---------------------------------------------------------------------------
# Resource encodings (dims 8, 9, 20, 21)
# ---------------------------------------------------------------------------


def encode_resource_category(resource_accessed: str) -> float:
    """
    Label-encode the resource category prefix to [0, 1].

    Extracts the category from the ``<category>/<identifier>`` format.
    Mapping: file=0, port=1, api=2, db=3, device=4, other=5; divided by 5.

    Args:
        resource_accessed: Raw resource string, e.g. ``file/reports/q1.xlsx``.

    Returns:
        Normalised float in [0, 1].

    Attack relevance:
        dim 8 — Lateral Movement detection: an entity normally accessing
        ``file/`` resources suddenly accessing ``port/`` or ``api/admin/``
        produces a category shift signal.

    Derives from:
        ``resource_accessed`` field.
    """
    prefix = resource_accessed.split("/")[0].lower() if "/" in resource_accessed else resource_accessed.lower()
    idx = RESOURCE_CATEGORY_MAP.get(prefix, RESOURCE_CATEGORY_OTHER)
    return float(idx) / RESOURCE_CATEGORY_MAX


def encode_resource_rarity(
    resource_accessed: str,
    resource_access_counts: Optional[Dict[str, int]],
    total_entity_events: int,
) -> float:
    """
    Compute the rarity of the accessed resource relative to entity history.

    Formula: ``1.0 - (count_for_this_resource / total_entity_events)``
    Returns 1.0 if the resource has never been seen (cold-start or novel resource).

    Args:
        resource_accessed: The resource being accessed in this event.
        resource_access_counts: Dict of ``{resource: count}`` from the
            entity's profile. If ``None`` or empty (cold-start), the resource
            is treated as completely novel → rarity = 1.0.
        total_entity_events: Total historical events for this entity.
            Used as denominator; defaults to 1 if zero to avoid division-by-zero.

    Returns:
        Float in [0, 1]. 0.0 = resource accessed every event (fully known);
        1.0 = never seen before.

    Attack relevance:
        dim 9 — Primary signal for Lateral Movement (all resources novel,
        rarity=1.0) and secondary signal for Insider Drift (gradually
        increasing rarity as new resources accumulate over weeks).

    Derives from:
        ``resource_accessed`` field + ``EntityProfile.resource_access_counts``.

    Label leakage prevention:
        The resource_access_counts dict is derived exclusively from past
        events (stored in EntityProfile); it never contains the label field.
    """
    if not resource_access_counts or total_entity_events <= 0:
        # Cold-start: all resources are novel — rarity = 1.0
        return 1.0
    count = resource_access_counts.get(resource_accessed, 0)
    freq = float(count) / max(float(total_entity_events), 1.0)
    return max(0.0, 1.0 - freq)


def encode_session_event_count(session_event_count: int) -> float:
    """
    Normalise the count of events in the current session to [0, 1].

    Formula: ``min(count / 200, 1.0)``

    Args:
        session_event_count: Number of events seen in the current session
            up to and including the current event.

    Returns:
        Float in [0, 1].

    Attack relevance:
        dim 20 — Brute Force detection: many events per session from
        rapid repeated authentication failures.

    Derives from:
        Count of events with the same ``session_id`` in the current batch.
    """
    return min(float(session_event_count) / MAX_SESSION_EVENT_COUNT, 1.0)


def encode_resource_breadth(distinct_resource_count: int) -> float:
    """
    Normalise the count of distinct resources accessed in the current session.

    Formula: ``min(distinct / 50, 1.0)``

    Args:
        distinct_resource_count: Number of unique resources accessed during
            the current session up to and including the current event.

    Returns:
        Float in [0, 1].

    Attack relevance:
        dim 21 — Lateral Movement (many distinct resources in one session)
        and Insider Drift (slowly increasing breadth over weeks).

    Derives from:
        Distinct values of ``resource_accessed`` within the current session.
    """
    return min(float(distinct_resource_count) / MAX_RESOURCE_BREADTH, 1.0)


# ---------------------------------------------------------------------------
# Command sequence encodings (dims 12, 13, 14)
# ---------------------------------------------------------------------------


def encode_command_seq_length(command_sequence: List[CommandEntry]) -> float:
    """
    Normalise command sequence length to [0, 1].

    Formula: ``min(len(sequence) / 50, 1.0)``

    Args:
        command_sequence: List of ``CommandEntry`` objects for this event.
            Empty list for non-privileged sessions.

    Returns:
        Float in [0, 1]; 0.0 for empty sequences.

    Attack relevance:
        dim 12 — Lateral Movement: the recon-to-exfil progression produces
        8–20 commands, far above the normal baseline.

    Derives from:
        ``command_sequence`` field.
    """
    return min(float(len(command_sequence)) / MAX_COMMAND_SEQ_LENGTH, 1.0)


def encode_command_rarity(
    command_sequence: List[CommandEntry],
    command_frequency: Optional[Dict[str, int]],
    total_entity_events: int,
    cold_start_default: float = 0.5,
) -> float:
    """
    Compute the mean per-command rarity relative to the entity's command history.

    For each command in the sequence, compute its rarity as
    ``1.0 - (count / total_entity_events)``. Return the mean rarity.
    Returns ``cold_start_default`` (0.5) for empty sequences or cold-start entities.

    Args:
        command_sequence: List of ``CommandEntry`` objects for this event.
        command_frequency: Dict of ``{command: count}`` from the entity's
            profile. If ``None``, treated as cold-start (no history).
        total_entity_events: Total historical events; used as denominator.
        cold_start_default: Value to return when sequence is empty or
            the entity has no command history. Default 0.5 (neutral).

    Returns:
        Float in [0, 1]. Higher = commands are more unusual for this entity.

    Attack relevance:
        dim 13 — Insider Drift (gradually introducing rare commands as
        footprint expands) and Lateral Movement (all commands in recon-to-exfil
        chain are rare for non-privileged normal entities).

    Derives from:
        ``command_sequence`` field + ``EntityProfile.command_frequency``.

    Label leakage prevention:
        command_frequency is built from past events in EntityProfile; label
        field is never stored there.
    """
    if not command_sequence:
        return cold_start_default
    if not command_frequency or total_entity_events <= 0:
        # Cold-start: no history — return neutral value
        return cold_start_default

    total_events = max(float(total_entity_events), 1.0)
    rarities = []
    for entry in command_sequence:
        count = command_frequency.get(entry.command, 0)
        freq = float(count) / total_events
        rarities.append(max(0.0, 1.0 - freq))
    return sum(rarities) / len(rarities)


def encode_has_exfil_command(command_sequence: List[CommandEntry]) -> float:
    """
    Binary indicator: 1.0 if any command in the sequence is an exfil command.

    Exfil commands: ``{scp, rsync, ftp, curl, wget, nc}``

    Args:
        command_sequence: List of ``CommandEntry`` objects for this event.

    Returns:
        1.0 if an exfil command is present; 0.0 otherwise.

    Attack relevance:
        dim 14 — Low-and-Slow (exfil command in every campaign event) and
        Lateral Movement (exfil command at the end of the recon-to-exfil chain).

    Derives from:
        ``command_sequence`` field.
    """
    for entry in command_sequence:
        if entry.command.lower() in EXFIL_COMMANDS:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Device fingerprint encodings (dims 15, 16, 17)
# ---------------------------------------------------------------------------


def encode_fingerprint_os_match(
    os_family: str,
    os_version: str,
    known_os_profiles: Optional[List[Dict[str, str]]],
    cold_start_default: float = 0.5,
) -> float:
    """
    Binary indicator: 1.0 if the (os_family, os_version) pair matches any
    known profile for this entity.

    Args:
        os_family: OS family from the current event's device_fingerprint.
        os_version: OS version from the current event's device_fingerprint.
        known_os_profiles: List of ``{os_family, os_version}`` dicts from
            ``EntityProfile.known_os_profiles``. ``None`` = cold-start.
        cold_start_default: Value returned when no profile exists (0.5 = neutral).

    Returns:
        1.0 (match), 0.0 (mismatch), or cold_start_default (unknown).

    Attack relevance:
        dim 15 — Device Spoofing (strategy B: OS family + version changed).
        A mismatch for a known device_id indicates a spoofed device.

    Derives from:
        ``device_fingerprint.os_family``, ``device_fingerprint.os_version``
        + ``EntityProfile.known_os_profiles``.
    """
    if known_os_profiles is None:
        return cold_start_default
    for profile in known_os_profiles:
        if profile.get("os_family") == os_family and profile.get("os_version") == os_version:
            return 1.0
    # Profile exists but this (os_family, os_version) was never seen
    if known_os_profiles:  # non-empty → entity is known → definite mismatch
        return 0.0
    return cold_start_default


def encode_fingerprint_mac_match(
    mac_address: str,
    known_mac_addresses: Optional[List[str]],
    cold_start_default: float = 0.5,
) -> float:
    """
    Binary indicator: 1.0 if the MAC address matches any known MAC for this entity.

    Args:
        mac_address: MAC address from the current event's device_fingerprint.
        known_mac_addresses: List of previously seen MACs from
            ``EntityProfile.known_mac_addresses``. ``None`` = cold-start.
        cold_start_default: Value returned when no profile exists.

    Returns:
        1.0 (MAC known), 0.0 (MAC unknown to this entity's profile),
        or cold_start_default (entity brand-new).

    Attack relevance:
        dim 16 — Primary signal for Device Spoofing. An attacker reusing
        a stolen device_id will have an unknown MAC address. Unlike Tier 2
        noise (which generates a new device_id), spoofing keeps the same
        device_id but changes the MAC.

    Derives from:
        ``device_fingerprint.mac_address`` + ``EntityProfile.known_mac_addresses``.
    """
    if known_mac_addresses is None:
        return cold_start_default
    if mac_address in known_mac_addresses:
        return 1.0
    if known_mac_addresses:  # non-empty list → entity is known → definite mismatch
        return 0.0
    return cold_start_default


def encode_fingerprint_protocol_match(
    protocol: str,
    known_protocols: Optional[List[str]],
    cold_start_default: float = 0.5,
) -> float:
    """
    Binary indicator: 1.0 if the protocol matches any known protocol for this entity.

    Args:
        protocol: Protocol from the current event's device_fingerprint.
        known_protocols: List of previously seen protocols from
            ``EntityProfile.known_protocols``. ``None`` = cold-start.
        cold_start_default: Value returned when no profile exists.

    Returns:
        1.0 (protocol known), 0.0 (protocol unknown), or cold_start_default.

    Attack relevance:
        dim 17 — Device Spoofing (strategy C, especially for edge devices
        where a switch from ``Modbus`` to ``HTTPS`` is flagrantly anomalous).

    Derives from:
        ``device_fingerprint.protocol`` + ``EntityProfile.known_protocols``.
    """
    if known_protocols is None:
        return cold_start_default
    if protocol in known_protocols:
        return 1.0
    if known_protocols:  # non-empty → entity is known → definite mismatch
        return 0.0
    return cold_start_default


# ---------------------------------------------------------------------------
# Entity type encoding (dim 18)
# ---------------------------------------------------------------------------


def encode_entity_type(entity_type: str) -> float:
    """
    Ordinal encoding of entity type for peer-group differentiation.

    Mapping: user=0.0, service_account=0.5, edge_device=1.0

    Args:
        entity_type: Raw entity type string from the event.

    Returns:
        One of {0.0, 0.5, 1.0}; defaults to 0.0 for unknown types.

    Attack relevance:
        dim 18 — Not directly an attack signal; routes events to the
        correct per-entity-type SDM model (Section 3.4 of ML_PIPELINE.md).

    Derives from:
        ``entity_type`` field.
    """
    _map: Dict[str, float] = {
        "user": 0.0,
        "service_account": 0.5,
        "edge_device": 1.0,
    }
    return _map.get(entity_type, 0.0)


# ---------------------------------------------------------------------------
# Inter-event gap (dim 19)
# ---------------------------------------------------------------------------


def encode_inter_event_gap(gap_seconds: float) -> float:
    """
    Normalise the time gap between consecutive events for the same entity.

    Formula: ``min(gap_seconds / 86400, 1.0)`` — normalised to 1 day.

    Args:
        gap_seconds: Time in seconds since the entity's previous event.
                     0.0 for the first event (no previous event).

    Returns:
        Float in [0, 1]; 1.0 if gap ≥ 24 hours.

    Attack relevance:
        dim 19 — Low-and-Slow: the attacker maintains long quiet periods
        (multiple days of silence) between covert access events. A recurring
        pattern of large inter_event_gap followed by off-hours exfil commands
        is the hallmark signature detected by the SDM.

    Derives from:
        ``timestamp`` difference between consecutive events for same entity.
    """
    return min(gap_seconds / 86400.0, 1.0)


# ---------------------------------------------------------------------------
# Cross-entity IP ratio features (dims 22, 23)
# ---------------------------------------------------------------------------


def encode_ip_entity_ratio(
    events_from_ip_in_window: int,
    total_events_for_entity_in_window: int,
) -> float:
    """
    Compute the ratio of events from this IP to events for this entity in a 24h window.

    Formula: ``min((events_from_ip / events_for_entity) / 10, 1.0)``

    A high ratio indicates one IP is responsible for a disproportionate
    fraction of this entity's recent events — the primary signal for
    Credential Stuffing (one IP hammering many entities).

    Args:
        events_from_ip_in_window: Number of events originating from
            the current source_ip in the past 24 hours.
        total_events_for_entity_in_window: Total events for this entity
            in the past 24 hours (denominator).

    Returns:
        Float in [0, 1].

    Attack relevance:
        dim 22 — Credential Stuffing: the attacker's single source IP
        appears in events for many different entities, making this ratio
        high across the target entity set.

    Derives from:
        ``source_ip`` + sliding 24h event counts maintained by SessionBuilder.

    Label leakage prevention:
        Ratio computed from event counts only — no label field involved.
    """
    if total_events_for_entity_in_window <= 0:
        return 0.0
    ratio = float(events_from_ip_in_window) / float(total_events_for_entity_in_window)
    return min(ratio / MAX_IP_ENTITY_RATIO, 1.0)


def encode_entity_ip_ratio(
    distinct_ips_for_entity_in_window: int,
    total_events_for_entity_in_window: int,
) -> float:
    """
    Compute the ratio of distinct IPs to total events for this entity in a 24h window.

    Formula: ``min((distinct_ips / total_events) / 5, 1.0)``

    A high ratio indicates the entity is authenticating from many different
    IP addresses, which may indicate compromised credentials being used across
    multiple locations.

    Args:
        distinct_ips_for_entity_in_window: Number of unique source IPs
            seen for this entity in the past 24 hours.
        total_events_for_entity_in_window: Total events for this entity
            in the past 24 hours (denominator).

    Returns:
        Float in [0, 1].

    Attack relevance:
        dim 23 — Compromised credentials (many IPs for one entity).
        Complements dim 22 (ip_entity_ratio) to distinguish Credential Stuffing
        (one IP, many entities) from credential compromise (one entity, many IPs).

    Derives from:
        ``source_ip`` + ``entity_id`` + sliding 24h event counts from SessionBuilder.
    """
    if total_events_for_entity_in_window <= 0:
        return 0.0
    ratio = float(distinct_ips_for_entity_in_window) / float(total_events_for_entity_in_window)
    return min(ratio / MAX_ENTITY_IP_RATIO, 1.0)


# ---------------------------------------------------------------------------
# geo is_new_geo (dim 7)
# ---------------------------------------------------------------------------


def encode_is_new_geo(
    current_country: str,
    most_frequent_country: Optional[str],
) -> float:
    """
    Binary indicator: 1.0 if the current event's country differs from the
    entity's most-frequent country.

    Args:
        current_country: ISO 3166-1 alpha-2 country code for the current event.
        most_frequent_country: The entity's most common country from
            ``EntityProfile.most_frequent_country``. ``None`` = cold-start
            (first event; entity has no known home country).

    Returns:
        1.0 if country is different from entity baseline, 0.0 if it matches.
        Returns 0.0 for cold-start (no baseline to compare against).

    Attack relevance:
        dim 7 — Impossible Travel: compound signal with geo_velocity_kmph.
        A new country on its own is a weak signal (2% Tier 2 noise);
        combined with high geo_velocity it becomes a near-certain indicator.

    Derives from:
        ``geo_location.country`` + ``EntityProfile.most_frequent_country``.
    """
    if most_frequent_country is None:
        # Cold-start: no baseline country known → cannot determine novelty
        return 0.0
    return 1.0 if current_country != most_frequent_country else 0.0
