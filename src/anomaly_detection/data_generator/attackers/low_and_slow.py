"""
Low-and-Slow Exfiltration attacker.

Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.6.
ATTACK_TAXONOMY.md: Detection Difficulty = High; Primary Signal = has_exfil_command
+ off-hours timestamp pattern over multiple days (SDM essential).

Injects 1-3 off-hours events per day over 5-20 days, each accessing an
authorized resource with an exfil command. Individually near-normal.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.data_generator.attackers.base import AttackRecord, BaseAttacker
from anomaly_detection.data_generator.entity_profiles import EntityProfile
from anomaly_detection.data_generator.injection_config import LowAndSlowConfig


class LowAndSlowAttacker(BaseAttacker):
    """Low-and-Slow Exfiltration attack injector.

    Injects multi-day campaigns of off-hours access events with exfil commands.
    Individual events are designed to appear near-normal; only the cumulative
    temporal pattern (detected by the SDM) reveals the attack.

    Detection Difficulty: HIGH — the hardest attack for BPM-only systems.
    SDM sequence-level pattern detection is essential.

    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.6.
    """

    SEED_OFFSET = 6_000

    def inject(
        self,
        entity_df: pd.DataFrame,
        profile: EntityProfile,
        n_sessions: int,
    ) -> Tuple[pd.DataFrame, AttackRecord]:
        cfg: LowAndSlowConfig = self._config
        rows: List[Dict[str, Any]] = []
        event_ids: List[str] = []
        timestamps: List[str] = []

        if entity_df.empty or not profile.resource_set:
            return pd.DataFrame(), AttackRecord(
                entity_id=profile.entity_id,
                attack_type="low_and_slow",
                event_ids=[],
                timestamps=[],
            )

        # Select exfil target resources from entity's ResourceSet (authorized access)
        n_exfil = int(self.rng.integers(
            cfg.n_exfil_resources_min,
            min(cfg.n_exfil_resources_max + 1, len(profile.resource_set) + 1)
        ))
        exfil_resource_indices = self.rng.choice(
            len(profile.resource_set), size=min(n_exfil, len(profile.resource_set)), replace=False
        )
        exfil_resources = [profile.resource_set[i] for i in exfil_resource_indices]
        if not exfil_resources:
            exfil_resources = profile.resource_set[:1]

        # Campaign duration constrained to entity's event window
        min_ts = entity_df["timestamp"].min()
        max_ts = entity_df["timestamp"].max()
        try:
            start_dt = datetime.fromisoformat(min_ts.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(max_ts.replace("Z", "+00:00"))
        except Exception:
            return pd.DataFrame(), AttackRecord(
                entity_id=profile.entity_id,
                attack_type="low_and_slow",
                event_ids=[],
                timestamps=[],
            )

        available_days = max(1, (end_dt - start_dt).days)
        max_duration = min(cfg.duration_days_max, available_days)
        min_duration = min(cfg.duration_days_min, max_duration)
        d_days = int(self.rng.integers(min_duration, max_duration + 1))
        p_daily = float(self.rng.uniform(cfg.p_daily_min, cfg.p_daily_max))

        anchor_row = entity_df.sample(n=1, random_state=int(self.rng.integers(0, 2**31))).iloc[0]

        for d in range(d_days):
            if self.rng.random() > p_daily:
                continue  # Attacker doesn't operate every day

            n_daily = int(self.rng.integers(1, 4))  # 1-3 events per active day
            for _ in range(n_daily):
                # Off-hours: 01:00-04:30 UTC
                off_hour = float(self.rng.uniform(cfg.off_hours_start_utc, cfg.off_hours_end_utc))
                day_ts = start_dt + timedelta(days=d)
                ts = day_ts.replace(
                    hour=int(off_hour),
                    minute=int((off_hour % 1) * 60),
                    second=int(self.rng.integers(0, 60)),
                    microsecond=0,
                )

                resource = exfil_resources[int(self.rng.integers(0, len(exfil_resources)))]
                exfil_cmd = cfg.exfil_commands[int(self.rng.integers(0, len(cfg.exfil_commands)))]
                event_id = self._rng_uuid(self.rng)
                ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

                # Session duration: brief but purposeful
                session_dur = float(self.rng.uniform(300, 3600))

                exfil_command_entry = [{
                    "sequence_position": 0,
                    "command": exfil_cmd,
                    "target": f"{self._random_foreign_ip(self.rng)}:/{resource.split('/')[-1]}",
                    "outcome": "success",
                    "elapsed_seconds": round(float(self.rng.uniform(10, 120)), 2),
                }]

                row = {
                    **anchor_row.to_dict(),
                    "event_id": event_id,
                    "session_id": self._rng_uuid(self.rng),
                    "timestamp": ts_str,
                    "resource_accessed": resource,
                    "auth_outcome": "success",
                    "failure_count": 0,
                    "session_duration": round(session_dur, 2),
                    "command_sequence": json.dumps(exfil_command_entry),
                    "label": AnomalyCategory.LOW_AND_SLOW.value,
                    "cold_start_flag": profile.is_late_joiner,
                }
                rows.append(row)
                event_ids.append(event_id)
                timestamps.append(ts_str)

        attack_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        return attack_df, AttackRecord(
            entity_id=profile.entity_id,
            attack_type="low_and_slow",
            event_ids=event_ids,
            timestamps=timestamps,
            extra={"campaign_days": d_days, "exfil_resources": exfil_resources},
        )
