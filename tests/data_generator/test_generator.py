"""
Tests for the Synthetic Data Generator — M03.

@pytest.mark.tier1

Tests cover:
- Reproducibility
- Schema conformance
- Late Joiner correctness
- Label correctness
- Inference export label-free
- Profile variance
- Entity type distribution
- CommandSequence and DeviceFingerprint validity
- GeneratorConfig validation
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from anomaly_detection.common.models.access_log import AccessLogTraining
from anomaly_detection.data_generator.config import (
    EntityPopulationConfig,
    GeneratorConfig,
    InjectionConfig,
)
from anomaly_detection.data_generator.entity_profiles import EntityProfile
from anomaly_detection.data_generator.generator import DataGenerator


@pytest.fixture(scope="module")
def small_config() -> GeneratorConfig:
    """Small config for fast test runs (50 entities, 7 days)."""
    return GeneratorConfig(
        random_seed=42,
        simulation_days=30,
        entity_population=EntityPopulationConfig(
            total_entities=100,
            user_fraction=0.70,
            service_account_fraction=0.20,
            edge_device_fraction=0.10,
        ),
        events_per_entity_lambda=20,
    )


@pytest.fixture(scope="module")
def profiles(small_config: GeneratorConfig) -> dict:
    gen = DataGenerator(small_config)
    return gen.generate_entity_profiles()


@pytest.fixture(scope="module")
def events_df(small_config: GeneratorConfig, profiles: dict) -> pd.DataFrame:
    gen = DataGenerator(small_config)
    return gen.generate_normal_events(profiles)


# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_reproducibility_same_seed(small_config: GeneratorConfig) -> None:
    """Same seed → identical DataFrame output."""
    gen1 = DataGenerator(small_config)
    gen2 = DataGenerator(small_config)
    p1 = gen1.generate_entity_profiles()
    p2 = gen2.generate_entity_profiles()
    df1 = gen1.generate_normal_events(p1)
    df2 = gen2.generate_normal_events(p2)

    assert list(df1.columns) == list(df2.columns)
    assert len(df1) == len(df2)
    # Compare event_ids (most sensitive reproducibility check)
    assert sorted(df1["event_id"].tolist()) == sorted(df2["event_id"].tolist())


@pytest.mark.tier1
def test_different_seed_different_output(small_config: GeneratorConfig) -> None:
    """Different seed → different output."""
    gen1 = DataGenerator(small_config)
    config2 = small_config.model_copy(update={"random_seed": 99})
    gen2 = DataGenerator(config2)
    p1 = gen1.generate_entity_profiles()
    p2 = gen2.generate_entity_profiles()
    df1 = gen1.generate_normal_events(p1)
    df2 = gen2.generate_normal_events(p2)
    # Event IDs will differ
    assert sorted(df1["event_id"].tolist()) != sorted(df2["event_id"].tolist())


# ---------------------------------------------------------------------------
# 2. Schema conformance
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_schema_conformance(events_df: pd.DataFrame) -> None:
    """All generated rows validate as AccessLogTraining with no ValidationError."""
    from pydantic import ValidationError

    errors = []
    for _, row in events_df.iterrows():
        row_dict = row.to_dict()
        # Parse JSON-string nested fields
        for field in ["geo_location", "device_fingerprint"]:
            if isinstance(row_dict.get(field), str):
                row_dict[field] = json.loads(row_dict[field])
        for field in ["command_sequence"]:
            if isinstance(row_dict.get(field), str):
                row_dict[field] = json.loads(row_dict[field])
        try:
            AccessLogTraining(**row_dict)
        except ValidationError as e:
            errors.append((row_dict.get("event_id"), str(e)))

    assert not errors, f"Schema validation errors: {errors[:3]}"


# ---------------------------------------------------------------------------
# 3. Late Joiner correctness
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_late_joiner_fraction(profiles: dict, small_config: GeneratorConfig) -> None:
    """Exactly 5% of entities are Late Joiners."""
    n_total = len(profiles)
    n_lj = sum(1 for p in profiles.values() if p.is_late_joiner)
    expected_lj = round(n_total * small_config.late_joiner_fraction)
    assert n_lj == expected_lj, f"Expected {expected_lj} Late Joiners, got {n_lj}"


@pytest.mark.tier1
def test_late_joiner_events_after_day26(
    profiles: dict, events_df: pd.DataFrame, small_config: GeneratorConfig
) -> None:
    """Late Joiner events must not occur before Day 26."""
    sim_start = datetime.fromisoformat(
        small_config.simulation_start_date.replace("Z", "+00:00")
    )
    lj_start = sim_start.replace(tzinfo=timezone.utc)
    from datetime import timedelta
    lj_start_day = sim_start + timedelta(days=small_config.late_joiner_start_day - 1)
    lj_start_day = lj_start_day.replace(tzinfo=timezone.utc)

    late_joiner_ids = {eid for eid, p in profiles.items() if p.is_late_joiner}

    if not late_joiner_ids:
        pytest.skip("No Late Joiners generated (population too small)")

    lj_events = events_df[events_df["entity_id"].isin(late_joiner_ids)]
    for _, row in lj_events.iterrows():
        ts_str = row["timestamp"]
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        assert ts >= lj_start_day, (
            f"Late Joiner event before Day 26: entity={row['entity_id']}, ts={ts_str}"
        )


# ---------------------------------------------------------------------------
# 4. Label correctness
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_all_labels_normal(events_df: pd.DataFrame) -> None:
    """All M03 output must have label='normal'."""
    assert events_df["label"].unique().tolist() == ["normal"], (
        f"Non-normal labels found: {events_df['label'].unique().tolist()}"
    )


# ---------------------------------------------------------------------------
# 5. Inference export label-free
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_inference_export_label_free(
    tmp_path, events_df: pd.DataFrame, small_config: GeneratorConfig
) -> None:
    """Inference export must not contain label field."""
    import pyarrow.parquet as pq

    gen = DataGenerator(small_config)
    path = str(tmp_path / "inference.parquet")
    gen.export_inference(events_df, path)
    table = pq.read_table(path)
    assert "label" not in table.schema.names, "label column found in inference export"
    assert "delivery_mode" in table.schema.names


@pytest.mark.tier1
def test_training_export_has_label(
    tmp_path, events_df: pd.DataFrame, small_config: GeneratorConfig
) -> None:
    """Training export must contain label field."""
    import pyarrow.parquet as pq

    gen = DataGenerator(small_config)
    path = str(tmp_path / "training.parquet")
    gen.export_training(events_df, path)
    table = pq.read_table(path)
    assert "label" in table.schema.names


# ---------------------------------------------------------------------------
# 6. Profile variance
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_user_profile_login_hour_variance(profiles: dict) -> None:
    """Login hour distributions differ across user entities."""
    user_profiles = [p for p in profiles.values() if p.entity_type == "user"]
    if len(user_profiles) < 5:
        pytest.skip("Too few user profiles for variance test")

    centers = [p.active_hour_center for p in user_profiles]
    import statistics
    variance = statistics.variance(centers)
    # With 70 users drawn from [7, 23], we expect high variance
    assert variance > 2.0, f"Login hour variance too low: {variance:.2f}"


# ---------------------------------------------------------------------------
# 7. Entity type distribution
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_entity_type_distribution(
    profiles: dict, small_config: GeneratorConfig
) -> None:
    """Generated population matches configured entity type proportions."""
    n_total = len(profiles)
    n_users = sum(1 for p in profiles.values() if p.entity_type == "user")
    n_svc = sum(1 for p in profiles.values() if p.entity_type == "service_account")
    n_edge = sum(1 for p in profiles.values() if p.entity_type == "edge_device")

    cfg = small_config.entity_population
    tol = 0.05  # allow ±5% tolerance due to rounding

    assert abs(n_users / n_total - cfg.user_fraction) <= tol, (
        f"User fraction off: expected {cfg.user_fraction}, got {n_users / n_total:.2f}"
    )
    assert abs(n_svc / n_total - cfg.service_account_fraction) <= tol, (
        f"SVC fraction off: expected {cfg.service_account_fraction}, got {n_svc / n_total:.2f}"
    )


# ---------------------------------------------------------------------------
# 8. CommandSequence and DeviceFingerprint validity
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_device_fingerprint_fields_present(events_df: pd.DataFrame) -> None:
    """All generated records have valid DeviceFingerprint sub-model instances."""
    required_fields = {
        "device_id", "os_family", "os_version",
        "mac_address", "protocol", "user_agent", "firmware_version",
    }
    for _, row in events_df.iterrows():
        fp = row["device_fingerprint"]
        if isinstance(fp, str):
            fp = json.loads(fp)
        assert required_fields.issubset(fp.keys()), (
            f"Missing fingerprint fields: {required_fields - fp.keys()}"
        )


@pytest.mark.tier1
def test_command_sequence_structure(events_df: pd.DataFrame) -> None:
    """Command sequence entries have the required sub-fields."""
    required_fields = {
        "sequence_position", "command", "target", "outcome", "elapsed_seconds"
    }
    for _, row in events_df.iterrows():
        cs = row["command_sequence"]
        if isinstance(cs, str):
            cs = json.loads(cs)
        for entry in cs:
            assert required_fields.issubset(entry.keys()), (
                f"Missing command entry fields: {required_fields - entry.keys()}"
            )


# ---------------------------------------------------------------------------
# 9. GeneratorConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_config_invalid_fractions() -> None:
    """Invalid entity type fractions should raise ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EntityPopulationConfig(
            total_entities=100,
            user_fraction=0.50,
            service_account_fraction=0.50,
            edge_device_fraction=0.50,  # sums to 1.5
        )


@pytest.mark.tier1
def test_config_invalid_injection_shares() -> None:
    """Invalid injection shares should raise ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        InjectionConfig(
            brute_force_share=0.50,
            impossible_travel_share=0.50,
            credential_stuffing_share=0.50,  # sums > 1
            lateral_movement_share=0.15,
            device_spoofing_share=0.15,
            low_and_slow_share=0.10,
            insider_drift_share=0.10,
        )


# ---------------------------------------------------------------------------
# 10. EntityProfile serialization
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_entity_profile_serializable(profiles: dict) -> None:
    """EntityProfile must serialize/deserialize correctly."""
    for entity_id, profile in list(profiles.items())[:3]:
        json_str = profile.to_json()
        restored = EntityProfile.from_dict(json.loads(json_str))
        assert restored.entity_id == profile.entity_id
        assert restored.entity_type == profile.entity_type
        assert restored.is_late_joiner == profile.is_late_joiner
