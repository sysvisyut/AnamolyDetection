"""
Configuration dataclass for the Feature Engineering Pipeline.

ARCHITECTURE COMPONENT: Feature Engineering
BOUNDARY RESPONSIBILITY: Internal configuration; does not cross a named boundary directly.
TIER: T1

All parameters are typed and documented. Population fallback statistics
(cold_start_*) are computed from training data at FeaturePipeline.fit()
time and stored here for inference-time use — they must never be hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants (module-level, not class-level, per CODING_GUIDELINES §1.4)
# ---------------------------------------------------------------------------

#: Minimum events before an entity graduates from cold-start status.
#: Matches DEFAULT in DATA_SCHEMA.md §5b and COLDSTART_DRIFT_STRATEGY.md §2.
MIN_PROFILE_EVENTS: int = 10

#: Default SDM sequence window length (DATA_SCHEMA.md §3.3).
DEFAULT_SEQUENCE_WINDOW: int = 20

#: Default sliding-window stride for sequence construction.
DEFAULT_STRIDE: int = 1

#: Exfil commands triggering has_exfil_command (dim 14).
EXFIL_COMMANDS: frozenset[str] = frozenset(
    {"scp", "rsync", "ftp", "curl", "wget", "nc"}
)

#: Mapping from resource category prefix to encoded integer (dim 8).
RESOURCE_CATEGORY_MAP: Dict[str, int] = {
    "file": 0,
    "port": 1,
    "api": 2,
    "db": 3,
    "device": 4,
}
#: Default encoding for unknown resource categories.
RESOURCE_CATEGORY_OTHER: int = 5
#: Divisor to normalise resource category encoding to [0, 1].
RESOURCE_CATEGORY_MAX: float = 5.0

#: Mapping from auth_method string to integer encoding (dim 10).
AUTH_METHOD_MAP: Dict[str, int] = {
    "password": 0,
    "token": 1,
    "certificate": 2,
    "biometric": 3,
    "none": 4,
}
#: Divisor to normalise auth_method encoding to [0, 1].
AUTH_METHOD_MAX: float = 4.0

#: Mapping from auth_outcome string to float encoding (dim 11).
AUTH_OUTCOME_MAP: Dict[str, float] = {
    "success": 0.0,
    "mfa_required": 0.5,
    "failure": 1.0,
}

#: Max failure_count for normalisation cap (dim 5).
MAX_FAILURE_COUNT: float = 20.0

#: Max geo-velocity in km/h before capping at 1.0 (dim 6).
MAX_GEO_VELOCITY_KMPH: float = 2000.0

#: Max inter-event gap in seconds (1 day = 86400 s) for normalisation (dim 19).
MAX_INTER_EVENT_GAP_SECONDS: float = 86400.0

#: Max command sequence length before capping at 1.0 (dim 12).
MAX_COMMAND_SEQ_LENGTH: float = 50.0

#: Max events in a session before capping session_event_count_norm at 1.0 (dim 20).
MAX_SESSION_EVENT_COUNT: float = 200.0

#: Max distinct resources before capping resource_breadth_norm at 1.0 (dim 21).
MAX_RESOURCE_BREADTH: float = 50.0

#: Max ip→entity ratio before capping ip_entity_ratio at 1.0 (dim 22).
MAX_IP_ENTITY_RATIO: float = 10.0

#: Max entity→IP ratio before capping entity_ip_ratio at 1.0 (dim 23).
MAX_ENTITY_IP_RATIO: float = 5.0

#: Rolling 24-hour window in seconds for ip_entity_ratio and entity_ip_ratio.
RATIO_WINDOW_SECONDS: float = 86400.0

#: Minimum standard deviation (epsilon) to avoid division-by-zero in z-score.
BASELINE_STD_EPSILON: float = 1e-6

#: Number of feature dimensions (locked at Phase 4; never change without MAJOR schema rev).
FEATURE_DIM: int = 24

#: Neutral fingerprint match value for cold-start entities (0.5 = unknown).
COLD_START_FINGERPRINT_NEUTRAL: float = 0.5

#: Neutral command rarity value for entities with no command history.
COLD_START_COMMAND_RARITY_NEUTRAL: float = 0.5


@dataclass
class FeatureEngineeringConfig:
    """
    Configuration for the Feature Engineering Pipeline.

    All parameters are documented with their source in DATA_SCHEMA.md §3.2
    or ML_PIPELINE.md. Population fallback statistics (cold_start_*) are
    populated by FeaturePipeline.fit() from training data and must never be
    hardcoded.
    """

    # ── Sequence parameters ────────────────────────────────────────────────
    sequence_window_size: int = DEFAULT_SEQUENCE_WINDOW
    """Length W of the SDM sliding window (DATA_SCHEMA.md §3.3)."""

    stride: int = DEFAULT_STRIDE
    """Step size between consecutive sequence windows for batch training."""

    # ── Profile graduation ─────────────────────────────────────────────────
    min_profile_events: int = MIN_PROFILE_EVENTS
    """Events before cold-start graduation (COLDSTART_DRIFT_STRATEGY.md §2)."""

    # ── Feature flags ──────────────────────────────────────────────────────
    enable_geo_features: bool = True
    """Compute geo_velocity_kmph and is_new_geo (dims 6, 7)."""

    enable_device_features: bool = True
    """Compute fingerprint match features (dims 15–17)."""

    enable_command_features: bool = True
    """Compute command-sequence features (dims 12–14)."""

    enable_ratio_features: bool = True
    """Compute ip_entity_ratio and entity_ip_ratio (dims 22–23)."""

    # ── Cold-start population fallback statistics ──────────────────────────
    # All values below are set by FeaturePipeline.fit(); defaults are 0.0
    # (safe no-op for normalised features) until fit() is called.
    cold_start_session_duration_mean: float = 0.0
    """Population mean session_duration used to normalise dim 4 for cold-start."""

    cold_start_session_duration_std: float = 1.0
    """Population std session_duration for cold-start normalisation."""

    # Per-entity-type fallback profiles: keyed by entity_type string value.
    # Each value is a list[float] of length FEATURE_DIM representing
    # the mean feature vector for that entity type over the training data.
    population_baseline_vectors: Dict[str, List[float]] = field(
        default_factory=dict
    )
    """
    Entity-type-level mean feature vectors, set by FeaturePipeline.fit().

    Keys: 'user', 'service_account', 'edge_device'.
    Values: list[float] of length 24.
    Used as baseline_vector fallback for cold-start entities.
    """

    population_baseline_stds: Dict[str, List[float]] = field(
        default_factory=dict
    )
    """
    Entity-type-level std feature vectors, set by FeaturePipeline.fit().

    Keys: 'user', 'service_account', 'edge_device'.
    Values: list[float] of length 24.
    Used as baseline_std fallback for cold-start entities.
    """

    # ── Ratio tracking ─────────────────────────────────────────────────────
    ratio_window_seconds: float = RATIO_WINDOW_SECONDS
    """Sliding time window (in seconds) for ip_entity_ratio / entity_ip_ratio."""

    # ── Most-frequent countries per entity type (cold-start fallback for dim 7) ──
    population_most_frequent_country: Optional[str] = None
    """
    Most common country across all training data.
    Used as fallback for is_new_geo when entity has no profile.
    Set by FeaturePipeline.fit().
    """
