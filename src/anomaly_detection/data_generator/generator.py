"""
Synthetic Data Generator — Core and Normal Behavior.

Implements M03 of the Phase 14 development roadmap.
Generates synthetic access log events for normal entity behavior only.
Attack injection is handled by M04 (attack_injector.py).

Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Sections 2 and 4.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.data_generator.config import GeneratorConfig
from anomaly_detection.data_generator.entity_profiles import (
    ALL_RESOURCES,
    API_RESOURCES,
    CITY_GEO_POOL,
    COMMAND_VOCABULARY,
    DB_RESOURCES,
    EDGE_DEVICE_PERSONAS,
    EXFIL_COMMANDS,
    FILE_RESOURCES,
    OS_FAMILY_CHOICES,
    OS_FAMILY_WEIGHTS,
    OS_VERSION_MAP,
    PORT_RESOURCES,
    SERVICE_ACCOUNT_PERSONAS,
    USER_AUTH_METHOD_CHOICES,
    USER_AUTH_METHOD_WEIGHTS,
    USER_PERSONAS,
    DeviceRecord,
    EntityProfile,
    GeoPoint,
)

# Persona-specific parameters for users
_USER_PERSONA_PARAMS: Dict[str, Dict[str, Any]] = {
    "executive": {
        "hour_center_range": (8, 18),
        "geo_variety": (2, 3),
        "resource_set_size": (20, 30),
        "privileged_session_prob": 0.05,
        "protocol": "HTTPS",
    },
    "developer": {
        "hour_center_range": (9, 17),
        "geo_variety": (1, 2),
        "resource_set_size": (40, 60),
        "privileged_session_prob": 0.30,
        "protocol": "HTTPS",
    },
    "analyst": {
        "hour_center_range": (8, 18),
        "geo_variety": (1, 1),
        "resource_set_size": (40, 60),
        "privileged_session_prob": 0.15,
        "protocol": "HTTPS",
    },
    "support": {
        "hour_center_range": (8, 20),
        "geo_variety": (1, 1),
        "resource_set_size": (50, 80),
        "privileged_session_prob": 0.15,
        "protocol": "RDP",
    },
    "remote_worker": {
        "hour_center_range": (8, 18),
        "geo_variety": (1, 1),
        "resource_set_size": (40, 60),
        "privileged_session_prob": 0.15,
        "protocol": "HTTPS",
    },
}

# Service account persona parameters
_SVC_PERSONA_PARAMS: Dict[str, Dict[str, Any]] = {
    "cicd_pipeline": {
        "interval_range": (60, 900),
        "jitter": 5.0,
        "session_mu": 5.5,
        "session_sigma": 0.4,
        "protocol": "HTTPS",
        "resource_set_size": (10, 20),
        "hour_range": (7, 19),
    },
    "monitoring_agent": {
        "interval_range": (60, 900),
        "jitter": 5.0,
        "session_mu": 4.0,
        "session_sigma": 0.3,
        "protocol": "HTTPS",
        "resource_set_size": (5, 10),
        "hour_range": (0, 24),
    },
    "etl_job": {
        "interval_range": (30, 120),
        "jitter": 2.0,
        "session_mu": 4.0,
        "session_sigma": 0.3,
        "protocol": "HTTPS",
        "resource_set_size": (5, 15),
        "hour_range": (0, 5),
    },
    "api_integration": {
        "interval_range": (60, 900),
        "jitter": 5.0,
        "session_mu": 4.0,
        "session_sigma": 0.3,
        "protocol": "HTTPS",
        "resource_set_size": (3, 8),
        "hour_range": (7, 19),
    },
}

# Edge device persona parameters
_EDGE_PERSONA_PARAMS: Dict[str, Dict[str, Any]] = {
    "iot_sensor": {
        "interval_range": (30, 60),
        "jitter": 1.0,
        "session_mu": 2.0,
        "session_sigma": 0.2,
        "protocol": "MQTT",
        "resource_count": (1, 2),
    },
    "plc_controller": {
        "interval_range": (1, 10),
        "jitter": 0.5,
        "session_mu": 2.0,
        "session_sigma": 0.2,
        "protocol": "Modbus",
        "resource_count": (1, 1),
    },
    "security_camera": {
        "interval_range": (25, 35),
        "jitter": 1.0,
        "session_mu": 2.0,
        "session_sigma": 0.2,
        "protocol": "HTTPS",
        "resource_count": (1, 1),
    },
    "rtu_device": {
        "interval_range": (5, 30),
        "jitter": 1.0,
        "session_mu": 2.0,
        "session_sigma": 0.2,
        "protocol": "DNP3",
        "resource_count": (1, 1),
    },
}


class DataGenerator:
    """Stateless synthetic data generator for normal entity behavior.

    Implements M03 of the development roadmap. Given the same config and
    random seed, two separate invocations produce byte-identical output.

    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Sections 2–4.

    The generator is stateless between calls given the same config and seed.
    All internal RNG state is recreated from config.random_seed at the start
    of each public method call.

    M04 (attack_injector.py) calls generate_entity_profiles() to obtain the
    entity behavioral profiles, then generate_normal_events(profiles) to get
    the baseline DataFrame, and overlays attack events using the profiles.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None) -> None:
        """Initialize the DataGenerator with optional config.

        Args:
            config: Generator configuration. Uses defaults if not provided.
        """
        self.config = config or GeneratorConfig()

    # ------------------------------------------------------------------
    # Public interface — consumed by M04
    # ------------------------------------------------------------------

    def generate_entity_profiles(self) -> Dict[str, EntityProfile]:
        """Generate per-entity behavioral profiles for the entire population.

        Returns a dict mapping entity_id → EntityProfile. Each profile
        is fully parameterized and serializable. Profiles are generated
        in a deterministic order from the seeded RNG.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2a–2f.
        """
        rng = np.random.default_rng(seed=self.config.random_seed)
        faker = Faker()
        Faker.seed(self.config.random_seed)

        cfg = self.config.entity_population
        n_total = cfg.total_entities
        n_users = round(n_total * cfg.user_fraction)
        n_svc = round(n_total * cfg.service_account_fraction)
        n_edge = n_total - n_users - n_svc

        profiles: Dict[str, EntityProfile] = {}

        # Assign Late Joiner slots proportionally across entity types
        n_late_joiners = round(n_total * self.config.late_joiner_fraction)
        n_lj_users = round(n_late_joiners * cfg.user_fraction)
        n_lj_svc = round(n_late_joiners * cfg.service_account_fraction)
        n_lj_edge = n_late_joiners - n_lj_users - n_lj_svc

        # Generate user profiles
        for i in range(n_users):
            entity_seed = self.config.random_seed + i * 1000
            is_lj = i < n_lj_users
            profile = self._generate_user_profile(
                entity_index=i,
                entity_seed=entity_seed,
                faker=faker,
                is_late_joiner=is_lj,
            )
            profiles[profile.entity_id] = profile

        # Generate service account profiles
        for i in range(n_svc):
            entity_seed = self.config.random_seed + (n_users + i) * 1000
            is_lj = i < n_lj_svc
            profile = self._generate_service_account_profile(
                entity_index=n_users + i,
                entity_seed=entity_seed,
                faker=faker,
                is_late_joiner=is_lj,
            )
            profiles[profile.entity_id] = profile

        # Generate edge device profiles
        for i in range(n_edge):
            entity_seed = self.config.random_seed + (n_users + n_svc + i) * 1000
            is_lj = i < n_lj_edge
            profile = self._generate_edge_device_profile(
                entity_index=n_users + n_svc + i,
                entity_seed=entity_seed,
                faker=faker,
                is_late_joiner=is_lj,
            )
            profiles[profile.entity_id] = profile

        return profiles

    def generate_normal_events(
        self, profiles: Dict[str, EntityProfile]
    ) -> pd.DataFrame:
        """Generate normal access log events for all entity profiles.

        All generated records have label='normal'. Late Joiner entities
        produce events only on Days 26–30.

        Args:
            profiles: Dict mapping entity_id → EntityProfile, as returned
                      by generate_entity_profiles().

        Returns:
            pd.DataFrame with columns matching AccessLogTraining schema,
            sorted by timestamp. All rows have label='normal'.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Sections 2a–2e.
        """
        sim_start = datetime.fromisoformat(
            self.config.simulation_start_date.replace("Z", "+00:00")
        )
        sim_end = sim_start + timedelta(days=self.config.simulation_days)
        lj_start = sim_start + timedelta(days=self.config.late_joiner_start_day - 1)

        all_rows: List[Dict[str, Any]] = []

        for entity_id, profile in profiles.items():
            entity_seed = (
                self.config.random_seed
                + abs(hash(entity_id)) % (2**31)
            )
            rng = np.random.default_rng(seed=entity_seed)
            faker = Faker()
            Faker.seed(entity_seed)

            # Determine event window
            event_start = lj_start if profile.is_late_joiner else sim_start
            event_end = sim_end

            # Sample event count from Poisson
            n_events = int(rng.poisson(lam=self.config.events_per_entity_lambda))
            n_events = max(n_events, 1)

            rows = self._generate_entity_events(
                profile=profile,
                n_events=n_events,
                event_start=event_start,
                event_end=event_end,
                sim_start=sim_start,
                rng=rng,
                faker=faker,
            )
            all_rows.extend(rows)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def export_training(self, df: pd.DataFrame, path: str) -> None:
        """Export the full training schema DataFrame to Parquet.

        Writes all columns including 'label'. Creates parent directories
        if needed.

        Args:
            df: DataFrame with all AccessLogTraining fields.
            path: Output file path (may contain {run_id} placeholder).

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 4.4.
        """
        import os

        path = self._resolve_path(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Serialize nested columns as JSON strings for Parquet compatibility
        export_df = self._flatten_for_parquet(df)
        export_df.to_parquet(path, index=False, engine="pyarrow")

    def export_labels(self, df: pd.DataFrame, path: str) -> None:
        """Export a label-only Parquet with just event_id and label columns.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 4.4 Step 2.
        """
        import os

        path = self._resolve_path(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        labels_df = df[["event_id", "label"]].copy()
        labels_df.to_parquet(path, index=False, engine="pyarrow")

    def export_inference(self, df: pd.DataFrame, path: str) -> None:
        """Export inference schema Parquet with the label column absent.

        Adds delivery_mode='batch' as required by AccessLogInference schema.
        Verifies that the exported file does not contain a 'label' column.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 4.4 Step 3–4.
        """
        import os

        path = self._resolve_path(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        inference_df = df.drop(columns=["label"], errors="ignore").copy()
        inference_df["delivery_mode"] = "batch"
        export_df = self._flatten_for_parquet(inference_df)
        export_df.to_parquet(path, index=False, engine="pyarrow")

        # Assert label is absent — required by Section 4.4 Step 4
        assert "label" not in export_df.columns, (
            "CRITICAL: label column found in inference export. "
            "Label stripping failed at boundary B."
        )

    def get_run_id(self) -> str:
        """Return the run_id for the current config."""
        if self.config.run_id == "auto":
            # Deterministic run_id from seed for reproducibility
            seed_bytes = str(self.config.random_seed).encode()
            return str(uuid.UUID(bytes=hashlib.md5(seed_bytes).digest()))
        return self.config.run_id

    # ------------------------------------------------------------------
    # Profile generation helpers
    # ------------------------------------------------------------------

    def _generate_user_profile(
        self,
        entity_index: int,
        entity_seed: int,
        faker: Faker,
        is_late_joiner: bool,
    ) -> EntityProfile:
        """Generate a behavioral profile for a user entity.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2b.
        """
        rng = np.random.default_rng(seed=entity_seed)

        entity_id = f"usr_{self._hex8(entity_seed)}"
        persona = USER_PERSONAS[int(rng.integers(0, len(USER_PERSONAS)))]
        params = _USER_PERSONA_PARAMS[persona]

        # Login timing — Section 2b: Login Timing Distribution
        h_min, h_max = params["hour_center_range"]
        active_hour_center = float(rng.uniform(h_min, h_max))
        active_hour_spread = float(rng.uniform(1.5, 3.0))

        # Tier 3: schedule drift for 10% of users
        has_schedule_drift = rng.random() < 0.10
        drift_rate = float(rng.uniform(0.1, 0.5)) if has_schedule_drift else 0.0

        # Tier 3: role expansion for 5% of users
        has_role_expansion = rng.random() < 0.05
        expansion_per_10d = int(rng.integers(1, 4)) if has_role_expansion else 0

        # Geo — Section 2b: Geo-Location Distribution
        geo_min, geo_max = params["geo_variety"]
        if geo_min == geo_max:
            n_cities = geo_min
        else:
            n_cities = int(rng.integers(geo_min, geo_max + 1))
        n_cities = min(n_cities, len(CITY_GEO_POOL))
        city_indices = rng.choice(len(CITY_GEO_POOL), size=n_cities, replace=False)
        home_geo_set = [
            GeoPoint(
                city=CITY_GEO_POOL[i]["city"],
                country=CITY_GEO_POOL[i]["country"],
                lat=CITY_GEO_POOL[i]["lat"],
                lon=CITY_GEO_POOL[i]["lon"],
            )
            for i in city_indices
        ]
        geo_weights_raw = [0.8, 0.15, 0.05][: len(home_geo_set)]
        geo_weights = self._normalize_weights(geo_weights_raw)

        # Resources — Section 2b: Resource Access Distribution
        r_min, r_max = params["resource_set_size"]
        n_resources = int(rng.integers(r_min, r_max + 1))
        resource_indices = rng.choice(len(ALL_RESOURCES), size=n_resources, replace=False)
        resource_set = [ALL_RESOURCES[i] for i in resource_indices]
        # Dirichlet α=0.5 produces heavy-tailed weights
        resource_weights_raw = rng.dirichlet(np.full(n_resources, 0.5)).tolist()
        resource_weights = self._normalize_weights(resource_weights_raw)

        # Devices — Section 2b: Device Fingerprint
        n_devices = int(rng.integers(1, 3))
        device_set = []
        for d in range(n_devices):
            dev_seed = entity_seed + d * 7
            device_set.append(self._make_user_device(rng, params["protocol"]))
        device_weights = [0.85, 0.15][: len(device_set)]
        device_weights = self._normalize_weights(device_weights)

        # Auth — Section 2b: Authentication
        auth_idx = rng.choice(
            len(USER_AUTH_METHOD_CHOICES),
            p=USER_AUTH_METHOD_WEIGHTS,
        )
        primary_auth_method = USER_AUTH_METHOD_CHOICES[int(auth_idx)]

        # Command pool — Section 2b: Command Sequence
        pool_size = int(rng.integers(5, 16))
        cmd_indices = rng.choice(len(COMMAND_VOCABULARY), size=pool_size, replace=False)
        command_pool = [COMMAND_VOCABULARY[i] for i in cmd_indices]

        # Normal subnet
        normal_ip_subnet = f"192.168.{int(rng.integers(0, 255))}"

        return EntityProfile(
            entity_id=entity_id,
            entity_type="user",
            persona=persona,
            is_late_joiner=is_late_joiner,
            active_hour_center=active_hour_center,
            active_hour_spread=active_hour_spread,
            has_schedule_drift=has_schedule_drift,
            drift_rate_hours_per_week=drift_rate,
            has_role_expansion=has_role_expansion,
            role_expansion_resources_per_10d=expansion_per_10d,
            home_geo_set=home_geo_set,
            home_geo_weights=geo_weights,
            resource_set=resource_set,
            resource_weights=resource_weights,
            device_set=device_set,
            device_weights=device_weights,
            primary_auth_method=primary_auth_method,
            session_duration_mu=7.0,
            session_duration_sigma=0.8,
            command_pool=command_pool,
            privileged_session_prob=params["privileged_session_prob"],
            normal_ip_subnet=normal_ip_subnet,
        )

    def _generate_service_account_profile(
        self,
        entity_index: int,
        entity_seed: int,
        faker: Faker,
        is_late_joiner: bool,
    ) -> EntityProfile:
        """Generate a behavioral profile for a service account entity.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2c.
        """
        rng = np.random.default_rng(seed=entity_seed)

        entity_id = f"svc_{self._hex8(entity_seed)}"
        persona = SERVICE_ACCOUNT_PERSONAS[int(rng.integers(0, len(SERVICE_ACCOUNT_PERSONAS)))]
        params = _SVC_PERSONA_PARAMS[persona]

        # Geo — single fixed location
        city_idx = int(rng.integers(0, len(CITY_GEO_POOL)))
        home_geo_set = [
            GeoPoint(
                city=CITY_GEO_POOL[city_idx]["city"],
                country=CITY_GEO_POOL[city_idx]["country"],
                lat=CITY_GEO_POOL[city_idx]["lat"],
                lon=CITY_GEO_POOL[city_idx]["lon"],
            )
        ]

        # Resources
        r_min, r_max = params["resource_set_size"]
        n_resources = int(rng.integers(r_min, r_max + 1))
        resource_pool = DB_RESOURCES + API_RESOURCES
        resource_indices = rng.choice(len(resource_pool), size=min(n_resources, len(resource_pool)), replace=False)
        resource_set = [resource_pool[i] for i in resource_indices]
        resource_weights = self._normalize_weights(rng.dirichlet(np.full(len(resource_set), 0.5)).tolist())

        # Device
        dev_seed = entity_seed + 7
        svc_device = DeviceRecord(
            device_id=f"dev_{self._hex8(dev_seed)}",
            os_family="Linux",
            os_version=OS_VERSION_MAP["Linux"][0],
            mac_address=self._fake_mac(dev_seed),
            protocol=params["protocol"],
            user_agent="",
            firmware_version="",
        )

        # Auth — Section 2c: token (70%) or certificate (30%)
        primary_auth_method = "token" if rng.random() < 0.70 else "certificate"

        # Interval
        i_min, i_max = params["interval_range"]
        interval = float(rng.uniform(i_min, i_max))

        # Active hour center (for near-deterministic timing)
        h_min, h_max = params["hour_range"]
        active_hour_center = float(rng.uniform(h_min, h_max))

        return EntityProfile(
            entity_id=entity_id,
            entity_type="service_account",
            persona=persona,
            is_late_joiner=is_late_joiner,
            active_hour_center=active_hour_center,
            active_hour_spread=0.5,  # very narrow for service accounts
            home_geo_set=home_geo_set,
            home_geo_weights=[1.0],
            resource_set=resource_set,
            resource_weights=resource_weights,
            device_set=[svc_device],
            device_weights=[1.0],
            primary_auth_method=primary_auth_method,
            session_duration_mu=params["session_mu"],
            session_duration_sigma=params["session_sigma"],
            command_pool=[],
            privileged_session_prob=0.0,
            event_interval_seconds=interval,
            event_interval_jitter=params["jitter"],
            normal_ip_subnet=f"10.0.{int(rng.integers(0, 255))}",
        )

    def _generate_edge_device_profile(
        self,
        entity_index: int,
        entity_seed: int,
        faker: Faker,
        is_late_joiner: bool,
    ) -> EntityProfile:
        """Generate a behavioral profile for an edge device entity.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2d.
        """
        rng = np.random.default_rng(seed=entity_seed)

        entity_id = f"dev_{self._hex8(entity_seed)}"
        persona = EDGE_DEVICE_PERSONAS[int(rng.integers(0, len(EDGE_DEVICE_PERSONAS)))]
        params = _EDGE_PERSONA_PARAMS[persona]

        # Geo — fully fixed, no variance
        city_idx = int(rng.integers(0, len(CITY_GEO_POOL)))
        home_geo_set = [
            GeoPoint(
                city=CITY_GEO_POOL[city_idx]["city"],
                country=CITY_GEO_POOL[city_idx]["country"],
                lat=CITY_GEO_POOL[city_idx]["lat"],
                lon=CITY_GEO_POOL[city_idx]["lon"],
            )
        ]

        # Resources — 1–2 endpoints
        r_min, r_max = params["resource_count"]
        n_resources = int(rng.integers(r_min, r_max + 1))
        resource_pool = API_RESOURCES + PORT_RESOURCES
        resource_indices = rng.choice(len(resource_pool), size=min(n_resources, len(resource_pool)), replace=False)
        resource_set = [resource_pool[i] for i in resource_indices]
        resource_weights = [1.0 / len(resource_set)] * len(resource_set)

        # Device — fully fixed
        dev_seed = entity_seed + 13
        fw_version = OS_VERSION_MAP["Embedded/RTU"][int(rng.integers(0, 3))]
        edge_device = DeviceRecord(
            device_id=entity_id,
            os_family="Embedded/RTU",
            os_version=fw_version,
            mac_address=self._fake_mac(dev_seed),
            protocol=params["protocol"],
            user_agent="",
            firmware_version=fw_version.replace("FW-", ""),
        )

        # Auth — Section 2d: certificate (80%), none (20%)
        primary_auth_method = "certificate" if rng.random() < 0.80 else "none"

        # Interval — near-deterministic
        i_min, i_max = params["interval_range"]
        interval = float(rng.uniform(i_min, i_max))

        return EntityProfile(
            entity_id=entity_id,
            entity_type="edge_device",
            persona=persona,
            is_late_joiner=is_late_joiner,
            active_hour_center=12.0,  # 24/7 operation
            active_hour_spread=12.0,  # uniform over day
            home_geo_set=home_geo_set,
            home_geo_weights=[1.0],
            resource_set=resource_set,
            resource_weights=resource_weights,
            device_set=[edge_device],
            device_weights=[1.0],
            primary_auth_method=primary_auth_method,
            session_duration_mu=2.0,
            session_duration_sigma=0.2,
            command_pool=[],
            privileged_session_prob=0.0,
            event_interval_seconds=interval,
            event_interval_jitter=1.0,
            normal_ip_subnet=f"10.1.{int(rng.integers(0, 255))}",
        )

    # ------------------------------------------------------------------
    # Event generation helpers
    # ------------------------------------------------------------------

    def _generate_entity_events(
        self,
        profile: EntityProfile,
        n_events: int,
        event_start: datetime,
        event_end: datetime,
        sim_start: datetime,
        rng: np.random.Generator,
        faker: Faker,
    ) -> List[Dict[str, Any]]:
        """Generate n_events normal events for a single entity profile.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Sections 2b–2e.
        """
        rows = []
        window_seconds = (event_end - event_start).total_seconds()
        if window_seconds <= 0:
            return rows

        # Sample event timestamps uniformly across the window
        offsets = np.sort(rng.uniform(0, window_seconds, size=n_events))

        # Session assignment: events within 30 min share a session_id
        session_id = self._rng_uuid(rng)
        last_ts: Optional[datetime] = None

        for i, offset_sec in enumerate(offsets):
            # Base timestamp
            ts = event_start + timedelta(seconds=float(offset_sec))

            # Tier 1: Gaussian timestamp noise (±30s)
            ts_noise = timedelta(seconds=float(rng.normal(0, 30)))
            ts = ts + ts_noise
            ts = max(ts, event_start)
            ts = min(ts, event_end)

            # Tier 2: off-hours override (probability 0.03)
            if profile.entity_type == "user" and rng.random() < 0.03:
                off_hour = float(rng.uniform(0, 24))
                ts = ts.replace(hour=int(off_hour) % 24, minute=int(rng.integers(0, 60)))

            # Session management
            if last_ts is not None:
                gap = (ts - last_ts).total_seconds()
                if gap > 1800:  # 30 minutes → new session
                    session_id = self._rng_uuid(rng)
            last_ts = ts

            # Geo selection — Section 2b
            geo = self._sample_geo(profile, rng)

            # Resource access — Section 2b
            resource = self._sample_resource(profile, rng)

            # Auth
            auth_method, auth_outcome, failure_count = self._sample_auth(profile, rng)

            # Session duration
            session_duration = self._sample_session_duration(profile, rng, auth_outcome)

            # Source IP
            source_ip = self._sample_source_ip(profile, rng)

            # Device
            device = self._sample_device(profile, rng)

            # Command sequence
            command_sequence = self._sample_command_sequence(profile, rng, resource, auth_outcome)

            event_id = self._rng_uuid(rng)
            timestamp_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            row: Dict[str, Any] = {
                "event_id": event_id,
                "session_id": session_id,
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type,
                "timestamp": timestamp_str,
                "source_ip": source_ip,
                "geo_location": {
                    "city": geo.city,
                    "country": geo.country,
                    "latitude": round(geo.lat + float(rng.normal(0, 0.05)), 6),
                    "longitude": round(geo.lon + float(rng.normal(0, 0.05)), 6),
                },
                "resource_accessed": resource,
                "auth_method": auth_method,
                "auth_outcome": auth_outcome,
                "session_duration": round(session_duration, 2),
                "command_sequence": command_sequence,
                "device_fingerprint": {
                    "device_id": device.device_id,
                    "os_family": device.os_family,
                    "os_version": device.os_version,
                    "mac_address": device.mac_address,
                    "protocol": device.protocol,
                    "user_agent": device.user_agent,
                    "firmware_version": device.firmware_version,
                },
                "failure_count": failure_count,
                "label": AnomalyCategory.NORMAL.value,
            }
            rows.append(row)

        return rows

    def _sample_geo(self, profile: EntityProfile, rng: np.random.Generator) -> GeoPoint:
        """Sample a geo-location from the entity's HomeGeoSet.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2b Geo-Location.
        """
        if not profile.home_geo_set:
            return GeoPoint(city="Unknown", country="XX", lat=0.0, lon=0.0)

        # Tier 2: foreign geo with probability 0.02
        if profile.entity_type == "user" and rng.random() < 0.02:
            idx = int(rng.integers(0, len(CITY_GEO_POOL)))
            c = CITY_GEO_POOL[idx]
            return GeoPoint(city=c["city"], country=c["country"], lat=c["lat"], lon=c["lon"])

        weights = np.array(profile.home_geo_weights)
        idx = int(rng.choice(len(profile.home_geo_set), p=weights))
        return profile.home_geo_set[idx]

    def _sample_resource(self, profile: EntityProfile, rng: np.random.Generator) -> str:
        """Sample a resource from the entity's ResourceSet.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2b Resource Access.
        """
        if not profile.resource_set:
            return "api/health"

        # Tier 2: rare resource (probability 0.03)
        if rng.random() < 0.03 and len(ALL_RESOURCES) > len(profile.resource_set):
            all_set = set(profile.resource_set)
            candidates = [r for r in ALL_RESOURCES if r not in all_set]
            if candidates:
                return candidates[int(rng.integers(0, len(candidates)))]

        weights = np.array(profile.resource_weights)
        idx = int(rng.choice(len(profile.resource_set), p=weights))
        return profile.resource_set[idx]

    def _sample_auth(
        self, profile: EntityProfile, rng: np.random.Generator
    ) -> Tuple[str, str, int]:
        """Sample auth method, outcome and failure count.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2b Authentication.
        """
        # Auth method: primary (95%) or secondary with probability 0.05
        if rng.random() < 0.05:
            method_choices = USER_AUTH_METHOD_CHOICES if profile.entity_type == "user" else ["token", "certificate"]
            method = method_choices[int(rng.integers(0, len(method_choices)))]
        else:
            method = profile.primary_auth_method

        # Auth outcome per entity type
        if profile.entity_type == "user":
            p = rng.random()
            if p < 0.96:
                outcome, failure_count = "success", 0
            elif p < 0.99:
                outcome, failure_count = "mfa_required", 0
            else:
                outcome = "failure"
                failure_count = int(rng.geometric(p=0.7))
        elif profile.entity_type == "service_account":
            outcome = "success" if rng.random() < 0.995 else "failure"
            failure_count = 0
        else:  # edge_device
            outcome = "success" if rng.random() < 0.999 else "failure"
            failure_count = 0

        return method, outcome, failure_count

    def _sample_session_duration(
        self, profile: EntityProfile, rng: np.random.Generator, auth_outcome: str
    ) -> float:
        """Sample session duration from the entity's log-normal distribution.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Sections 2b–2d Session Duration.
        """
        if auth_outcome == "failure":
            return 0.0

        raw = float(np.exp(rng.normal(profile.session_duration_mu, profile.session_duration_sigma)))
        # Tier 1: multiplicative lognormal noise
        noise_factor = float(np.exp(rng.normal(0, 0.1)))
        raw = raw * noise_factor
        # Cap at 8 hours per Section 2b
        return min(raw, 28800.0)

    def _sample_source_ip(self, profile: EntityProfile, rng: np.random.Generator) -> str:
        """Sample source IP from the entity's normal subnet.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2e Tier 2.
        """
        # Tier 2: foreign IP with probability 0.01
        if rng.random() < 0.01:
            return f"{int(rng.integers(1, 255))}.{int(rng.integers(0, 255))}.{int(rng.integers(0, 255))}.{int(rng.integers(1, 255))}"

        last_octet = int(rng.integers(1, 255))
        return f"{profile.normal_ip_subnet}.{last_octet}"

    def _sample_device(self, profile: EntityProfile, rng: np.random.Generator) -> DeviceRecord:
        """Sample a device from the entity's DeviceSet.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2e Tier 2.
        """
        if not profile.device_set:
            return DeviceRecord(
                device_id="dev_unknown",
                os_family="Unknown",
                os_version="0.0",
                mac_address="00:00:00:00:00:00",
                protocol="HTTPS",
                user_agent="",
                firmware_version="",
            )

        # Tier 2: unknown device with probability 0.005
        if len(profile.device_set) > 1 and rng.random() < 0.005:
            seed = int(rng.integers(0, 2**31))
            return self._make_user_device(rng, profile.device_set[0].protocol)

        weights = np.array(profile.device_weights)
        idx = int(rng.choice(len(profile.device_set), p=weights))
        return profile.device_set[idx]

    def _sample_command_sequence(
        self,
        profile: EntityProfile,
        rng: np.random.Generator,
        resource: str,
        auth_outcome: str,
    ) -> List[Dict[str, Any]]:
        """Sample a command sequence for the event.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2b Command Sequence.
        """
        if auth_outcome == "failure" or not profile.command_pool:
            return []

        if rng.random() >= profile.privileged_session_prob:
            return []

        # Sample sequence length from Poisson(λ=5), minimum 1
        n_cmds = max(1, int(rng.poisson(lam=5)))
        cmds = []
        elapsed = 0.0

        for pos in range(n_cmds):
            cmd = profile.command_pool[int(rng.integers(0, len(profile.command_pool)))]
            target = resource  # use event resource as default target
            p = rng.random()
            if p < 0.93:
                outcome = "success"
            elif p < 0.98:
                outcome = "failure"
            else:
                outcome = "denied"

            # elapsed_seconds: sequence_position × Gamma(shape=2, scale=15)
            step = float(rng.gamma(shape=2, scale=15))
            elapsed += step

            cmds.append({
                "sequence_position": pos,
                "command": cmd,
                "target": target,
                "outcome": outcome,
                "elapsed_seconds": round(elapsed, 2),
            })

        return cmds

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_weights(weights: List[float]) -> List[float]:
        """Normalize a list of weights to sum to 1.0."""
        total = sum(weights)
        if total == 0:
            return [1.0 / len(weights)] * len(weights)
        return [w / total for w in weights]

    @staticmethod
    def _hex8(seed: int) -> str:
        """Produce a deterministic 8-character hex string from an integer seed."""
        return format(abs(seed) % (16**8), "08x")

    @staticmethod
    def _rng_uuid(rng: np.random.Generator) -> str:
        """Generate a deterministic UUID-formatted string from the seeded RNG.

        Uses 16 random bytes from the numpy RNG to produce a UUID v4-format string.
        This ensures reproducibility across multiple calls with the same seed.
        """
        raw = rng.integers(0, 256, size=16, dtype=np.uint8)
        raw[6] = (raw[6] & 0x0F) | 0x40  # version 4
        raw[8] = (raw[8] & 0x3F) | 0x80  # variant bits
        h = raw.tobytes().hex()
        return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    @staticmethod
    def _fake_mac(seed: int) -> str:
        """Generate a deterministic fake MAC address from a seed."""
        h = format(abs(seed) % (16**12), "012x")
        return ":".join(h[i:i+2].upper() for i in range(0, 12, 2))

    def _make_user_device(
        self, rng: np.random.Generator, protocol: str
    ) -> DeviceRecord:
        """Create a new DeviceRecord using the entity's RNG."""
        os_idx = int(rng.choice(len(OS_FAMILY_CHOICES), p=OS_FAMILY_WEIGHTS))
        os_family = OS_FAMILY_CHOICES[os_idx]
        os_version = OS_VERSION_MAP[os_family][int(rng.integers(0, len(OS_VERSION_MAP[os_family])))]
        seed_int = int(rng.integers(0, 2**31))
        device_id = f"dev_{self._hex8(seed_int)}"
        mac = self._fake_mac(seed_int)
        user_agent = f"Mozilla/5.0 ({os_family} NT {os_version})"
        return DeviceRecord(
            device_id=device_id,
            os_family=os_family,
            os_version=os_version,
            mac_address=mac,
            protocol=protocol,
            user_agent=user_agent,
            firmware_version="",
        )

    def _resolve_path(self, path: str) -> str:
        """Resolve {run_id} placeholder in path templates."""
        return path.replace("{run_id}", self.get_run_id())

    @staticmethod
    def _flatten_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
        """Convert dict/list columns to JSON strings for Parquet compatibility."""
        import json as _json

        flat = df.copy()
        for col in ["geo_location", "device_fingerprint"]:
            if col in flat.columns:
                flat[col] = flat[col].apply(
                    lambda v: _json.dumps(v) if isinstance(v, dict) else v
                )
        for col in ["command_sequence"]:
            if col in flat.columns:
                flat[col] = flat[col].apply(
                    lambda v: _json.dumps(v) if isinstance(v, list) else v
                )
        return flat
