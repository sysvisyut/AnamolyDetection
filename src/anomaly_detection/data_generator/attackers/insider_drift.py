"""
Insider Drift attacker — genuinely ambiguous edge case.

Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.7.
ATTACK_TAXONOMY.md: Detection Difficulty = VERY HIGH; Edge Case; Expected Risk Tier = Medium (25-49).

CRITICAL IMPLEMENTATION NOTE: Insider Drift must be genuinely ambiguous, not a milder
version of Lateral Movement. All five fast signals must be ABSENT:
  1. Normal business hours (no timing anomaly)
  2. Authorized access / auth_outcome=success (no auth anomaly)
  3. No exfil commands (no exfil signal)
  4. Registered device (no device anomaly)
  5. Home geo-location (no travel anomaly)

The only detectable signal is the slow monotonic expansion of resource_accessed
to resources outside the entity's ResourceSet but within the SAME CATEGORY FAMILY.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.data_generator.attackers.base import AttackRecord, BaseAttacker
from anomaly_detection.data_generator.entity_profiles import ALL_RESOURCES, EntityProfile
from anomaly_detection.data_generator.injection_config import InsiderDriftConfig


def _resource_category(resource: str) -> str:
    """Extract the top-level category from a resource path."""
    return resource.split("/")[0] if "/" in resource else "other"


class InsiderDriftAttacker(BaseAttacker):
    """Insider Drift attack injector.

    Injects a gradual, multi-week expansion of resource access into the
    same category family as the entity's existing ResourceSet. Every
    individual event is indistinguishable from normal behavior on all
    fast-signal dimensions. Only the slow monotonic trend in
    resource_rarity_score and resource_breadth_norm reveals the attack.

    Detection Difficulty: VERY HIGH — all five fast signals are absent by design.
    The BPM will score individual events as 0.30-0.55 (near-normal range).
    The SDM must detect the slow gradient over the 30-day sequence window.

    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.7.
    """

    SEED_OFFSET = 7_000

    def inject(
        self,
        entity_df: pd.DataFrame,
        profile: EntityProfile,
        n_sessions: int,
    ) -> Tuple[pd.DataFrame, AttackRecord]:
        cfg: InsiderDriftConfig = self._config
        rows: List[Dict[str, Any]] = []
        event_ids: List[str] = []
        timestamps: List[str] = []

        if entity_df.empty or not profile.resource_set or profile.entity_type != "user":
            return pd.DataFrame(), AttackRecord(
                entity_id=profile.entity_id,
                attack_type="insider_drift",
                event_ids=[],
                timestamps=[],
            )

        # Identify entity's primary resource categories
        entity_categories: Set[str] = {_resource_category(r) for r in profile.resource_set}

        # Build drift resource pool: outside entity's ResourceSet but SAME category family
        entity_resource_set = set(profile.resource_set)
        drift_pool = self._build_drift_pool(entity_resource_set, entity_categories, cfg)

        if not drift_pool:
            return pd.DataFrame(), AttackRecord(
                entity_id=profile.entity_id,
                attack_type="insider_drift",
                event_ids=[],
                timestamps=[],
            )

        # Determine drift start day
        min_ts_str = entity_df["timestamp"].min()
        max_ts_str = entity_df["timestamp"].max()
        try:
            sim_start = datetime.fromisoformat(min_ts_str.replace("Z", "+00:00"))
            sim_end = datetime.fromisoformat(max_ts_str.replace("Z", "+00:00"))
        except Exception:
            return pd.DataFrame(), AttackRecord(
                entity_id=profile.entity_id,
                attack_type="insider_drift",
                event_ids=[],
                timestamps=[],
            )

        available_days = max(1, (sim_end - sim_start).days)
        max_start_day = min(cfg.drift_start_day_max, max(1, available_days // 2))
        min_start_day = min(cfg.drift_start_day_min, max_start_day)
        drift_start_day = int(self.rng.integers(min_start_day, max_start_day + 1))
        drift_start_dt = sim_start + timedelta(days=drift_start_day)

        # Drift rate: 1-3 new resources per week
        drift_rate = int(self.rng.integers(cfg.drift_rate_min, cfg.drift_rate_max + 1))
        p_daily = float(self.rng.uniform(cfg.p_daily_min, cfg.p_daily_max))

        # Track which drift resources have been accessed (for rarity decay simulation)
        accessed_drift: List[str] = []
        drift_pool_iter = list(drift_pool)
        self.rng.shuffle(drift_pool_iter)

        # Get a representative anchor row for non-geo/device/auth fields
        success_events = entity_df[entity_df["auth_outcome"] == "success"]
        anchor_pool = success_events if not success_events.empty else entity_df

        total_drift_days = available_days - drift_start_day
        resources_added_this_week = 0
        week_counter = 0

        for d in range(drift_start_day, available_days):
            # Weekly resource allocation: drift_rate new resources per week
            if d % 7 == 0 and d > drift_start_day:
                week_counter += 1
                resources_added_this_week = 0

            if self.rng.random() > p_daily:
                continue  # Attacker doesn't operate every day

            # Select drift resource: prefer new resources but occasionally re-access old ones
            # After 14 days, re-access early drift resources (simulating "settling in")
            d_since_start = d - drift_start_day
            if d_since_start >= 14 and accessed_drift and self.rng.random() < 0.4:
                # Re-access an early drift resource (simulates role normalization)
                resource = accessed_drift[int(self.rng.integers(0, len(accessed_drift)))]
            elif drift_pool_iter:
                resource = drift_pool_iter.pop(0)
                accessed_drift.append(resource)
                resources_added_this_week += 1
            elif accessed_drift:
                resource = accessed_drift[int(self.rng.integers(0, len(accessed_drift)))]
            else:
                continue

            # Inject 1-2 events per active day
            n_daily = int(self.rng.integers(1, 3))
            for _ in range(n_daily):
                # AMBIGUITY PROPERTY 1: Normal business hours (not off-hours)
                hour = float(self.rng.normal(profile.active_hour_center, profile.active_hour_spread))
                hour = max(7.0, min(22.0, hour))  # truncate to plausible business hours

                day_ts = drift_start_dt + timedelta(days=d - drift_start_day)
                ts = day_ts.replace(
                    hour=int(hour),
                    minute=int(self.rng.integers(0, 60)),
                    second=int(self.rng.integers(0, 60)),
                    microsecond=0,
                )

                anchor_row = anchor_pool.sample(n=1, random_state=int(self.rng.integers(0, 2**31))).iloc[0]
                event_id = self._rng_uuid(self.rng)
                ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

                # AMBIGUITY PROPERTIES 2-5: success auth, no exfil, registered device, home geo
                # Command sequence: entity's NORMAL commands, no exfil
                normal_cmds: List[Dict] = []
                if profile.command_pool and self.rng.random() < profile.privileged_session_prob:
                    cmd_name = profile.command_pool[int(self.rng.integers(0, len(profile.command_pool)))]
                    if cmd_name not in {"scp", "rsync", "wget", "curl", "tar", "ftp", "nc"}:
                        normal_cmds = [{
                            "sequence_position": 0,
                            "command": cmd_name,
                            "target": resource,
                            "outcome": "success",
                            "elapsed_seconds": round(float(self.rng.gamma(2, 15)), 2),
                        }]

                row = {
                    **anchor_row.to_dict(),  # preserves normal device, geo, source_ip
                    "event_id": event_id,
                    "session_id": self._rng_uuid(self.rng),
                    "timestamp": ts_str,
                    "resource_accessed": resource,
                    "auth_outcome": "success",  # AMBIGUITY: always success
                    "failure_count": 0,
                    "session_duration": round(
                        float(np.exp(self.rng.normal(profile.session_duration_mu, profile.session_duration_sigma))),
                        2
                    ),
                    "command_sequence": json.dumps(normal_cmds),  # AMBIGUITY: no exfil commands
                    "label": AnomalyCategory.INSIDER_DRIFT.value,
                    "cold_start_flag": profile.is_late_joiner,
                }
                rows.append(row)
                event_ids.append(event_id)
                timestamps.append(ts_str)

        attack_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        return attack_df, AttackRecord(
            entity_id=profile.entity_id,
            attack_type="insider_drift",
            event_ids=event_ids,
            timestamps=timestamps,
            extra={
                "drift_start_day": drift_start_day,
                "drift_rate_per_week": drift_rate,
                "drift_resources_total": len(accessed_drift),
                "entity_categories": list(entity_categories),
            },
        )

    def _build_drift_pool(
        self,
        entity_resource_set: Set[str],
        entity_categories: Set[str],
        cfg: InsiderDriftConfig,
    ) -> List[str]:
        """Build drift resource pool outside entity's set but within same category family.

        If category-constrained pool is too small (<3 resources), falls back to
        any resource not in entity's set to avoid empty drift (flagged ambiguity).
        """
        if cfg.resource_category_constrained:
            # Same-category resources not in entity's set
            pool = [
                r for r in ALL_RESOURCES
                if r not in entity_resource_set and _resource_category(r) in entity_categories
            ]
            if len(pool) >= 3:
                return pool
            # Fallback: any resource not in entity's set
        return [r for r in ALL_RESOURCES if r not in entity_resource_set]
