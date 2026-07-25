"""
Impossible Travel attacker.

Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.2.
ATTACK_TAXONOMY.md: Detection Difficulty = Low; Primary Fields = geo_location, source_ip, timestamp.

Inserts a successful auth event from a geographically remote location
(>500 km, velocity >800 km/h) within 5-30 minutes of a legitimate anchor event.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.data_generator.attackers.base import AttackRecord, BaseAttacker
from anomaly_detection.data_generator.entity_profiles import CITY_GEO_POOL, EntityProfile
from anomaly_detection.data_generator.injection_config import ImpossibleTravelConfig

# Earth radius in km for Haversine distance
_EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance in km between two lat/lon points."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


class ImpossibleTravelAttacker(BaseAttacker):
    """Impossible Travel attack injector.

    Inserts one impossible-velocity event per target entity. The event
    is placed 5-30 minutes after an existing anchor event, with a geo-location
    guaranteed to be >500 km away at >800 km/h velocity.

    Detection Difficulty: LOW — geo-velocity is a physical constraint that
    no normal noise event violates (Tier 2 noise constrains foreign IPs to
    within 500 km of home cities).

    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.2.
    """

    SEED_OFFSET = 2_000

    def inject(
        self,
        entity_df: pd.DataFrame,
        profile: EntityProfile,
        n_sessions: int,
    ) -> Tuple[pd.DataFrame, AttackRecord]:
        cfg: ImpossibleTravelConfig = self._config
        rows: List[Dict[str, Any]] = []
        event_ids: List[str] = []
        timestamps: List[str] = []

        if entity_df.empty or not profile.home_geo_set:
            return pd.DataFrame(), AttackRecord(
                entity_id=profile.entity_id,
                attack_type="impossible_travel",
                event_ids=[],
                timestamps=[],
            )

        # Select anchor event: closest prior success event
        success_events = entity_df[entity_df["auth_outcome"] == "success"]
        if success_events.empty:
            success_events = entity_df

        anchor_row = success_events.sort_values("timestamp").iloc[-1]
        anchor_ts = datetime.fromisoformat(anchor_row["timestamp"].replace("Z", "+00:00"))

        # Get anchor geo coordinates
        anchor_geo = anchor_row["geo_location"]
        if isinstance(anchor_geo, str):
            anchor_geo = json.loads(anchor_geo)
        anchor_lat = anchor_geo["latitude"]
        anchor_lon = anchor_geo["longitude"]

        for _ in range(n_sessions):
            # Select remote location guaranteed to be >min_distance_km away
            remote_city = None
            for _ in range(50):  # max attempts
                idx = int(self.rng.integers(0, len(CITY_GEO_POOL)))
                candidate = CITY_GEO_POOL[idx]
                dist = _haversine_km(anchor_lat, anchor_lon, candidate["lat"], candidate["lon"])
                if dist >= cfg.min_distance_km:
                    # Verify velocity requirement
                    delta_t_min = float(self.rng.uniform(cfg.delta_t_min_minutes, cfg.delta_t_max_minutes))
                    velocity = dist / (delta_t_min / 60.0)
                    if velocity >= cfg.min_velocity_kmph:
                        remote_city = candidate
                        break

            if remote_city is None:
                # Fallback: use the most distant city in the pool
                distances = [
                    (_haversine_km(anchor_lat, anchor_lon, c["lat"], c["lon"]), c)
                    for c in CITY_GEO_POOL
                ]
                remote_city = max(distances, key=lambda x: x[0])[1]
                delta_t_min = cfg.delta_t_min_minutes

            impossible_ts = anchor_ts + timedelta(minutes=delta_t_min)
            ts_str = impossible_ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            event_id = self._rng_uuid(self.rng)
            foreign_ip = self._random_foreign_ip(self.rng)

            row = {
                **anchor_row.to_dict(),
                "event_id": event_id,
                "session_id": self._rng_uuid(self.rng),
                "timestamp": ts_str,
                "source_ip": foreign_ip,
                "geo_location": json.dumps({
                    "city": remote_city["city"],
                    "country": remote_city["country"],
                    "latitude": round(remote_city["lat"] + float(self.rng.normal(0, 0.01)), 6),
                    "longitude": round(remote_city["lon"] + float(self.rng.normal(0, 0.01)), 6),
                }),
                "auth_outcome": "success",
                "failure_count": 0,
                "label": AnomalyCategory.IMPOSSIBLE_TRAVEL.value,
                "cold_start_flag": profile.is_late_joiner,
            }
            rows.append(row)
            event_ids.append(event_id)
            timestamps.append(ts_str)

        attack_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        return attack_df, AttackRecord(
            entity_id=profile.entity_id,
            attack_type="impossible_travel",
            event_ids=event_ids,
            timestamps=timestamps,
            extra={"remote_city": remote_city["city"] if remote_city else "unknown"},
        )
