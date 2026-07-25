"""
Brute Force attacker.

Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.1.
ATTACK_TAXONOMY.md: Detection Difficulty = Low; Primary Fields = failure_count, source_ip, timestamp.

Injects N=10-50 rapid authentication failure events against a single target entity
from a single source IP, spaced 1-8 seconds apart.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.data_generator.attackers.base import AttackRecord, BaseAttacker
from anomaly_detection.data_generator.entity_profiles import EntityProfile
from anomaly_detection.data_generator.injection_config import BruteForceConfig


class BruteForceAttacker(BaseAttacker):
    """Brute Force attack injector.

    Injects a rapid authentication failure burst against a target entity from
    a single foreign IP, optionally followed by a successful compromise event.

    Detection Difficulty: LOW — the burst of consecutive failures from a single
    IP is structurally distinct from normal noise (max 1-3 normal failures ever).

    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.1.
    """

    SEED_OFFSET = 1_000

    def inject(
        self,
        entity_df: pd.DataFrame,
        profile: EntityProfile,
        n_sessions: int,
    ) -> Tuple[pd.DataFrame, AttackRecord]:
        """Inject brute force burst(s) for a single entity.

        Args:
            entity_df: Normal events for this entity (read-only).
            profile: Entity behavioral profile.
            n_sessions: Number of burst attacks to inject.

        Returns:
            (attack_df, record) — new rows only, normal rows unmodified.
        """
        cfg: BruteForceConfig = self._config
        rows: List[Dict] = []
        event_ids: List[str] = []
        timestamps: List[str] = []

        if entity_df.empty:
            return pd.DataFrame(), AttackRecord(
                entity_id=profile.entity_id,
                attack_type="brute_force",
                event_ids=[],
                timestamps=[],
            )

        # Select injection timestamp from entity's active hours
        anchor_row = entity_df.sample(n=1, random_state=int(self.rng.integers(0, 2**31))).iloc[0]
        anchor_ts = datetime.fromisoformat(anchor_row["timestamp"].replace("Z", "+00:00"))

        for _ in range(n_sessions):
            # Attacker source IP — single foreign IP for the entire burst
            foreign_ip = self._random_foreign_ip(self.rng)
            n_fail = int(self.rng.integers(cfg.n_fail_min, cfg.n_fail_max + 1))
            session_id = self._rng_uuid(self.rng)

            # Space failures 1-8 seconds apart
            ts = anchor_ts - timedelta(seconds=float(n_fail * cfg.inter_event_sec_max))
            for k in range(n_fail):
                gap = float(self.rng.uniform(cfg.inter_event_sec_min, cfg.inter_event_sec_max))
                ts = ts + timedelta(seconds=gap)
                ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                event_id = self._rng_uuid(self.rng)

                row = {
                    **anchor_row.to_dict(),
                    "event_id": event_id,
                    "session_id": session_id,
                    "timestamp": ts_str,
                    "source_ip": foreign_ip,
                    "auth_outcome": "failure",
                    "auth_method": "password",
                    "failure_count": k + 1,
                    "session_duration": 0.0,
                    "command_sequence": "[]",
                    "label": AnomalyCategory.BRUTE_FORCE.value,
                    "cold_start_flag": profile.is_late_joiner,
                }
                rows.append(row)
                event_ids.append(event_id)
                timestamps.append(ts_str)

            # Optional successful compromise event (probability 0.4)
            if self.rng.random() < cfg.success_probability:
                ts = ts + timedelta(seconds=float(self.rng.uniform(1, 5)))
                ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                event_id = self._rng_uuid(self.rng)
                row = {
                    **anchor_row.to_dict(),
                    "event_id": event_id,
                    "session_id": self._rng_uuid(self.rng),
                    "timestamp": ts_str,
                    "source_ip": foreign_ip,
                    "auth_outcome": "success",
                    "failure_count": 0,
                    "session_duration": float(
                        np.exp(self.rng.normal(profile.session_duration_mu, profile.session_duration_sigma))
                    ),
                    "label": AnomalyCategory.BRUTE_FORCE.value,
                    "cold_start_flag": profile.is_late_joiner,
                }
                rows.append(row)
                event_ids.append(event_id)
                timestamps.append(ts_str)

        attack_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        return attack_df, AttackRecord(
            entity_id=profile.entity_id,
            attack_type="brute_force",
            event_ids=event_ids,
            timestamps=timestamps,
            extra={"source_ip": foreign_ip if rows else ""},
        )
