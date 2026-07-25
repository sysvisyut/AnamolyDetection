"""
Attack injection configuration model for M04.

Implements the configuration surface for all attack types as specified in
SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 4.1 (injection block).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class BruteForceConfig(BaseModel):
    """Configuration for Brute Force attack injection.
    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.1.
    """
    n_fail_min: int = Field(default=10, ge=1, description="Min authentication failures per burst.")
    n_fail_max: int = Field(default=50, ge=1, description="Max authentication failures per burst.")
    inter_event_sec_min: float = Field(default=1.0, ge=0.1, description="Min seconds between failures.")
    inter_event_sec_max: float = Field(default=8.0, ge=0.1, description="Max seconds between failures.")
    success_probability: float = Field(default=0.4, ge=0.0, le=1.0, description="P(successful compromise after burst).")


class ImpossibleTravelConfig(BaseModel):
    """Configuration for Impossible Travel injection.
    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.2.
    """
    delta_t_min_minutes: float = Field(default=5.0, ge=1.0, description="Min minutes between anchor and impossible event.")
    delta_t_max_minutes: float = Field(default=30.0, ge=1.0, description="Max minutes between anchor and impossible event.")
    min_distance_km: float = Field(default=500.0, ge=100.0, description="Min geographic distance for impossibility.")
    min_velocity_kmph: float = Field(default=800.0, ge=200.0, description="Min velocity to qualify as impossible.")


class CredentialStuffingConfig(BaseModel):
    """Configuration for Credential Stuffing injection.
    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.3.
    """
    n_entities_min: int = Field(default=15, ge=2, description="Min entities targeted per campaign.")
    n_entities_max: int = Field(default=60, ge=2, description="Max entities targeted per campaign.")
    n_fail_per_entity_min: int = Field(default=1, ge=1, description="Min failures per entity.")
    n_fail_per_entity_max: int = Field(default=5, ge=1, description="Max failures per entity.")
    compromise_probability: float = Field(default=0.10, ge=0.0, le=1.0, description="P(successful compromise per entity).")
    stagger_sec_min: float = Field(default=2.0, ge=0.1, description="Min seconds between inter-entity events.")
    stagger_sec_max: float = Field(default=15.0, ge=0.1, description="Max seconds between inter-entity events.")


class LateralMovementConfig(BaseModel):
    """Configuration for Lateral Movement injection.
    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.4.
    """
    n_expand_min: int = Field(default=10, ge=2, description="Min expansion resources per session.")
    n_expand_max: int = Field(default=30, ge=2, description="Max expansion resources per session.")
    min_categories: int = Field(default=3, ge=2, description="Min distinct resource categories accessed.")
    cmd_length_min: int = Field(default=8, ge=2, description="Min command sequence length.")
    cmd_length_max: int = Field(default=20, ge=2, description="Max command sequence length.")
    inter_event_sec_min: float = Field(default=30.0, ge=5.0, description="Min seconds between events.")
    inter_event_sec_max: float = Field(default=120.0, ge=5.0, description="Max seconds between events.")


class DeviceSpoofingConfig(BaseModel):
    """Configuration for Device Spoofing injection.
    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.5.
    """
    n_spoof_events_min: int = Field(default=1, ge=1, description="Min events with spoofed fingerprint.")
    n_spoof_events_max: int = Field(default=5, ge=1, description="Max events with spoofed fingerprint.")
    strategy_weights: tuple = Field(default=(0.5, 0.3, 0.2), description="Weights for MAC/OS/Protocol spoof strategy.")
    min_prior_events: int = Field(default=10, ge=1, description="Min prior events before injection.")


class LowAndSlowConfig(BaseModel):
    """Configuration for Low-and-Slow Exfiltration injection.
    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.6.
    """
    duration_days_min: int = Field(default=5, ge=2, description="Min campaign duration in days.")
    duration_days_max: int = Field(default=20, ge=2, description="Max campaign duration in days.")
    p_daily_min: float = Field(default=0.4, ge=0.0, le=1.0, description="Min daily operation probability.")
    p_daily_max: float = Field(default=0.8, ge=0.0, le=1.0, description="Max daily operation probability.")
    off_hours_start_utc: int = Field(default=1, ge=0, le=23, description="Start hour (UTC) of off-hours window.")
    off_hours_end_utc: float = Field(default=4.5, ge=0.0, le=24.0, description="End hour (UTC) of off-hours window.")
    n_exfil_resources_min: int = Field(default=5, ge=1, description="Min resources in exfil target set.")
    n_exfil_resources_max: int = Field(default=15, ge=1, description="Max resources in exfil target set.")
    exfil_commands: list = Field(
        default_factory=lambda: ["scp", "rsync", "wget", "curl"],
        description="Commands that qualify as exfiltration."
    )


class InsiderDriftConfig(BaseModel):
    """Configuration for Insider Drift injection.
    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.7.
    """
    drift_start_day_min: int = Field(default=5, ge=1, description="Min simulation day to start drift.")
    drift_start_day_max: int = Field(default=20, ge=1, description="Max simulation day to start drift.")
    drift_rate_min: int = Field(default=1, ge=1, description="Min new resources per week.")
    drift_rate_max: int = Field(default=3, ge=1, description="Max new resources per week.")
    p_daily_min: float = Field(default=0.2, ge=0.0, le=1.0, description="Min daily operation probability.")
    p_daily_max: float = Field(default=0.5, ge=0.0, le=1.0, description="Max daily operation probability.")
    resource_category_constrained: bool = Field(
        default=True,
        description="If True, drift resources stay within same category family (harder to detect)."
    )


class AttackInjectionConfig(BaseModel):
    """Top-level attack injection configuration for M04.

    Maps to SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 4.1 injection block.
    All per-attack shares must sum to 1.0.
    """
    random_seed: int = Field(default=42, ge=0, description="Base seed for all attacker RNGs.")
    target_anomaly_rate: float = Field(
        default=0.015, ge=0.0, le=1.0,
        description="Target fraction of total events to be non-normal. Default 1.5%."
    )
    anomaly_rate_min: float = Field(default=0.005, ge=0.0, description="Hard floor on anomaly rate.")
    anomaly_rate_max: float = Field(default=0.030, ge=0.0, description="Hard ceiling on anomaly rate.")

    # Per-attack-type budget shares (must sum to 1.0)
    brute_force_share: float = Field(default=0.20, ge=0.0, le=1.0)
    impossible_travel_share: float = Field(default=0.15, ge=0.0, le=1.0)
    credential_stuffing_share: float = Field(default=0.15, ge=0.0, le=1.0)
    lateral_movement_share: float = Field(default=0.15, ge=0.0, le=1.0)
    device_spoofing_share: float = Field(default=0.15, ge=0.0, le=1.0)
    low_and_slow_share: float = Field(default=0.10, ge=0.0, le=1.0)
    insider_drift_share: float = Field(default=0.10, ge=0.0, le=1.0)

    # Per-attack-type config objects
    brute_force: BruteForceConfig = Field(default_factory=BruteForceConfig)
    impossible_travel: ImpossibleTravelConfig = Field(default_factory=ImpossibleTravelConfig)
    credential_stuffing: CredentialStuffingConfig = Field(default_factory=CredentialStuffingConfig)
    lateral_movement: LateralMovementConfig = Field(default_factory=LateralMovementConfig)
    device_spoofing: DeviceSpoofingConfig = Field(default_factory=DeviceSpoofingConfig)
    low_and_slow: LowAndSlowConfig = Field(default_factory=LowAndSlowConfig)
    insider_drift: InsiderDriftConfig = Field(default_factory=InsiderDriftConfig)

    @model_validator(mode="after")
    def validate_shares_sum_to_one(self) -> "AttackInjectionConfig":
        total = (
            self.brute_force_share
            + self.impossible_travel_share
            + self.credential_stuffing_share
            + self.lateral_movement_share
            + self.device_spoofing_share
            + self.low_and_slow_share
            + self.insider_drift_share
        )
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Attack type shares must sum to 1.0, got {total:.4f}")
        return self

    def normalized_shares(self) -> dict:
        """Return a dict of attack_type → normalized share."""
        raw = {
            "brute_force": self.brute_force_share,
            "impossible_travel": self.impossible_travel_share,
            "credential_stuffing": self.credential_stuffing_share,
            "lateral_movement": self.lateral_movement_share,
            "device_spoofing": self.device_spoofing_share,
            "low_and_slow": self.low_and_slow_share,
            "insider_drift": self.insider_drift_share,
        }
        total = sum(raw.values())
        if total == 0:
            return raw
        return {k: v / total for k, v in raw.items()}
