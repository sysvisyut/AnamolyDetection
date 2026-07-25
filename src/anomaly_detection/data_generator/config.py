"""
Generator configuration model for the Synthetic Data Generator.

Implements the configuration surface defined in
SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 4.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class EntityPopulationConfig(BaseModel):
    """Entity population distribution configuration.

    Implements SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 4.1,
    data_generator.entity_population block.
    """

    total_entities: int = Field(
        default=500,
        ge=10,
        le=100_000,
        description="Total number of entities to generate.",
    )
    user_fraction: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Fraction of entities of type 'user' (default 70%).",
    )
    service_account_fraction: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Fraction of entities of type 'service_account' (default 20%).",
    )
    edge_device_fraction: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Fraction of entities of type 'edge_device' (default 10%).",
    )

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "EntityPopulationConfig":
        """Validate that entity type fractions sum to 1.0."""
        total = self.user_fraction + self.service_account_fraction + self.edge_device_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"user_fraction + service_account_fraction + edge_device_fraction must sum to 1.0, "
                f"got {total:.4f}"
            )
        return self


class InjectionConfig(BaseModel):
    """Attack injection rate configuration (placeholder for M04).

    Implements SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 4.1,
    data_generator.injection block. M03 does not inject attacks;
    these values are the interface that M04 will consume.
    """

    target_anomaly_rate: float = Field(
        default=0.015,
        ge=0.0,
        le=1.0,
        description="Target fraction of total events labeled non-normal. Default 1.5%.",
    )
    anomaly_rate_min: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description="Hard floor on anomaly rate per problem statement.",
    )
    anomaly_rate_max: float = Field(
        default=0.030,
        ge=0.0,
        le=1.0,
        description="Hard ceiling on anomaly rate per problem statement.",
    )
    brute_force_share: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Fraction of anomaly budget allocated to brute_force.",
    )
    impossible_travel_share: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Fraction of anomaly budget allocated to impossible_travel.",
    )
    credential_stuffing_share: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Fraction of anomaly budget allocated to credential_stuffing.",
    )
    lateral_movement_share: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Fraction of anomaly budget allocated to lateral_movement.",
    )
    device_spoofing_share: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Fraction of anomaly budget allocated to device_spoofing.",
    )
    low_and_slow_share: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Fraction of anomaly budget allocated to low_and_slow.",
    )
    insider_drift_share: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Fraction of anomaly budget allocated to insider_drift.",
    )

    @model_validator(mode="after")
    def shares_sum_to_one(self) -> "InjectionConfig":
        """Validate that per-attack shares sum to 1.0."""
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
            raise ValueError(
                f"Attack type shares must sum to 1.0, got {total:.4f}"
            )
        return self


class OutputConfig(BaseModel):
    """Output path configuration for generated data files.

    Implements SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 4.1,
    data_generator.output block.
    """

    raw_parquet_path: str = Field(
        default="data/raw/synthetic_logs_{run_id}.parquet",
        description="Path template for full training schema Parquet output (includes label).",
    )
    labels_parquet_path: str = Field(
        default="data/labeled/labels_{run_id}.parquet",
        description="Path template for label-only Parquet output (event_id + label).",
    )
    inference_parquet_path: str = Field(
        default="data/processed/inference_logs_{run_id}.parquet",
        description="Path template for inference schema Parquet output (label field absent).",
    )


class GeneratorConfig(BaseModel):
    """Top-level configuration model for the Synthetic Data Generator.

    Implements the complete configuration surface defined in
    SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 4.

    All parameters are documented with their purpose, type, default value,
    and valid range as required by M03 acceptance criteria.
    """

    run_id: str = Field(
        default="auto",
        description=(
            "'auto' = UUID v4 generated at runtime. Set explicitly for reproducibility. "
            "Stored in Parquet metadata for traceability."
        ),
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description=(
            "NumPy and Faker global random seed. "
            "Implements SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 4.3. "
            "Same seed + same config = byte-identical output."
        ),
    )
    simulation_days: int = Field(
        default=30,
        ge=7,
        le=365,
        description=(
            "Length of simulated timeline in days. "
            "Days 1–25 are the training window; Days 26–30 are the evaluation window."
        ),
    )
    simulation_start_date: str = Field(
        default="2026-07-01T00:00:00Z",
        description="ISO-8601 UTC start of the simulated period.",
    )
    late_joiner_fraction: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description=(
            "Fraction of entities designated as 'Late Joiners' per "
            "SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2f. "
            "Late Joiners emit no events before Day 26."
        ),
    )
    late_joiner_start_day: int = Field(
        default=26,
        ge=1,
        description=(
            "First day (1-indexed) on which Late Joiner events may occur. "
            "Implements the Phase 12.5 patch: evaluation window starts Day 26."
        ),
    )
    entity_population: EntityPopulationConfig = Field(
        default_factory=EntityPopulationConfig,
        description="Entity population distribution configuration.",
    )
    events_per_entity_lambda: int = Field(
        default=120,
        ge=10,
        le=10_000,
        description=(
            "Poisson λ for events per entity per simulation_days window. "
            "Produces ~60–200 events per entity."
        ),
    )
    injection: InjectionConfig = Field(
        default_factory=InjectionConfig,
        description="Attack injection rate configuration (consumed by M04).",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output file path templates.",
    )
