"""
Configuration and constants for the Behavioral Profiling Model (M06).

Contains parameters for graduation, Z-score computation, and the exact
human-readable feature names required for M09 Explainability.
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


# 24-dimensional feature names derived exactly from M05's Feature Inventory Table.
# These names are the keys used in ProfilingOutput.per_feature_deviations so that
# M09 (Explainability) can directly reference them in narrative generation.
FEATURE_NAMES: List[str] = [
    "hour_sin",                  # 0
    "hour_cos",                  # 1
    "dow_sin",                   # 2
    "dow_cos",                   # 3
    "session_duration_norm",     # 4
    "failure_count_norm",        # 5
    "geo_velocity_norm",         # 6
    "is_new_geo",                # 7
    "resource_category_enc",     # 8
    "resource_rarity_score",     # 9
    "auth_method_enc",           # 10
    "auth_outcome_enc",          # 11
    "command_seq_length_norm",   # 12
    "command_rarity_score",      # 13
    "has_exfil_command",         # 14
    "fingerprint_os_match",      # 15
    "fingerprint_mac_match",     # 16
    "fingerprint_protocol_match",# 17
    "entity_type_enc",           # 18
    "inter_event_gap_norm",      # 19
    "session_event_count_norm",  # 20
    "resource_breadth_norm",     # 21
    "ip_entity_ratio",           # 22
    "entity_ip_ratio",           # 23
]


class ProfilingConfig(BaseSettings):
    """
    Configuration for Behavioral Profiling Model parameters.
    """
    # 20 events provides a statistically minimum viable sample to estimate variance
    # without being overly noisy, while graduating most active entities within
    # a typical observation window (e.g., 30 days).
    graduation_threshold: int = Field(
        default=20,
        ge=1,
        description="Number of events required to graduate from cold-start to warm"
    )

    # Cap Z-scores at 5.0 to map deviation to a [0, 1] range (where 1.0 is 5σ or more).
    z_score_cap: float = Field(
        default=5.0,
        ge=1.0,
        description="Maximum absolute Z-score, used for anomaly score normalization"
    )

    # Minimum variance to prevent division by zero in zero-variance features
    # (e.g., constant entity_type_enc).
    variance_epsilon: float = Field(
        default=1e-4,
        gt=0.0,
        description="Minimum standard deviation epsilon to avoid division by zero"
    )

    # Persistence settings
    profile_store_path: str = Field(
        default="data/profiles",
        description="Directory path to save/load entity profiles"
    )

    population_prior_path: str = Field(
        default="data/population_prior.json",
        description="File path to save/load population prior statistics"
    )

    model_config = {"env_prefix": "PROFILING_"}
