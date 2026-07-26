"""
Sequence Detection Model configuration.

All hyperparameters, training settings, and persistence paths are
centralised here. The model class reads exclusively from this config,
ensuring zero hardcoded dimensions or layer counts in the model.

ML_PIPELINE.md §3.3, §3.7 — frozen Phase 6 design.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Feature names — canonical 24-dim ordering per DATA_SCHEMA.md §3.2
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "session_duration_norm",
    "failure_count_norm",
    "geo_velocity_kmph",
    "is_new_geo",
    "resource_category_enc",
    "resource_rarity_score",
    "auth_method_enc",
    "auth_outcome_enc",
    "command_seq_length_norm",
    "command_rarity_score",
    "has_exfil_command",
    "fingerprint_os_match",
    "fingerprint_mac_match",
    "fingerprint_protocol_match",
    "entity_type_enc",
    "inter_event_gap_norm",
    "session_event_count_norm",
    "resource_breadth_norm",
    "ip_entity_ratio",
    "entity_ip_ratio",
]

FEATURE_DIM: int = len(FEATURE_NAMES)  # always 24, locked in DATA_SCHEMA.md §6.4


# ---------------------------------------------------------------------------
# Model & training configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectionModelConfig:
    """
    Full configuration for the GRU Autoencoder Sequence Detection Model.

    Every architectural decision, training hyper-parameter, and I/O path
    is expressed here. The model class MUST NOT contain any literals that
    duplicate or override these values.

    ML_PIPELINE.md §3.3 and §3.7 specify default values; they are frozen
    from Phase 6 and are NOT changed here.
    """

    # --- Architecture (ML_PIPELINE.md §3.3) ---
    feature_dim: int = FEATURE_DIM           # F = 24; locked
    window_size: int = 20                    # W = 20; locked per DATA_SCHEMA.md §3.3
    hidden_size: int = 64                    # encoder/decoder GRU hidden units
    num_encoder_layers: int = 2              # 2-layer GRU encoder
    num_decoder_layers: int = 2              # 2-layer GRU decoder
    dropout: float = 0.2                     # inter-layer dropout (not applied on last layer)

    # --- Training (ML_PIPELINE.md §3.7) ---
    learning_rate: float = 1e-3              # Adam lr
    weight_decay: float = 1e-5              # Adam L2 regularisation
    batch_size: int = 64                     # sequences per batch
    max_epochs: int = 50                     # hard cap; early stopping fires first
    early_stopping_patience: int = 5        # epochs without val-MSE improvement
    lr_scheduler_factor: float = 0.5        # ReduceLROnPlateau reduction factor
    lr_scheduler_patience: int = 3          # epochs without val-MSE improvement for LR reduce
    lr_min: float = 1e-5                    # minimum LR floor

    # --- Score Calibration ---
    # Percentiles used to clip raw reconstruction error to [0, 1].
    # Populated during training; stored in checkpoint.
    calibration_err_min: float = 0.0        # 1st percentile of training errors (filled post-fit)
    calibration_err_max: float = 1.0        # 99th percentile of training errors (filled post-fit)
    raw_score_weight_mean: float = 1.0      # weight on mean per-event error (see ML_PIPELINE.md §3.6)
    raw_score_weight_max: float = 0.3       # weight on max per-event error (see ML_PIPELINE.md §3.6)

    # --- Cold-Start (COLDSTART_DRIFT_STRATEGY.md §2) ---
    cold_start_score_factor: float = 0.7    # multiply anomaly_score when is_cold_start=True
    cold_start_confidence_cap: float = 0.5  # cap confidence for cold-start entities
    cold_start_padding_threshold: float = 0.5  # padding fraction above which cold_start_flag=True

    # --- Persistence ---
    artifacts_dir: str = "models/sequence_detection/artifacts"
    checkpoint_filename_template: str = "sdm_{entity_type}.pt"  # {entity_type} filled at save time

    # --- Reproducibility ---
    random_seed: int = 42

    # --- Imbalance strategy annotation (informational; train-on-normals-only) ---
    imbalance_strategy: Literal["train_on_normals"] = "train_on_normals"

    def checkpoint_path(self, entity_type: str) -> str:
        """Return the full path for the given entity type's checkpoint."""
        filename = self.checkpoint_filename_template.format(entity_type=entity_type)
        return os.path.join(self.artifacts_dir, filename)

    def to_dict(self) -> dict:
        """Serialise config to a plain dict for checkpoint storage."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DetectionModelConfig":
        """Deserialise config from a plain dict (e.g., loaded checkpoint)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
