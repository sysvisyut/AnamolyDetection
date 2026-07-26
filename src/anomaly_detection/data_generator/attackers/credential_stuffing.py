"""
Credential Stuffing campaign attacker.

Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.3.
ATTACK_TAXONOMY.md: Detection Difficulty = Medium; Primary Signal = ip_entity_ratio (cross-entity).

This is a population-level attack: a single campaign IP targets 15-60 entities
simultaneously, injecting 1-5 failures per entity from the same source IP.
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
from anomaly_detection.data_generator.injection_config import CredentialStuffingConfig


class CredentialStuffingAttacker(BaseAttacker):
    """Credential Stuffing campaign injector.

    Unlike per-entity attackers, this is a campaign-level attacker that
    operates across a pool of target entities, sharing a single source IP
    across all of them. Called once per campaign (not per entity).

    Detection Difficulty: MEDIUM — per-entity signal is weak (1-5 failures),
    but the cross-entity signal (ip_entity_ratio) is strong.

    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.3.
    """

    SEED_OFFSET = 3_000

    def inject(
        self,
        entity_df: pd.DataFrame,
        profile: EntityProfile,
        n_sessions: int,
    ) -> Tuple[pd.DataFrame, AttackRecord]:
        """Single-entity injection stub — real logic is in inject_campaign()."""
        # inject_campaign() is the primary interface for this attacker
        return pd.DataFrame(), AttackRecord(
            entity_id=profile.entity_id,
            attack_type="credential_stuffing",
            event_ids=[],
            timestamps=[],
        )

    def inject_campaign(
        self,
        all_entity_dfs: Dict[str, pd.DataFrame],
        profiles: Dict[str, EntityProfile],
        n_campaigns: int,
    ) -> Tuple[pd.DataFrame, List[AttackRecord]]:
        """Inject credential stuffing campaigns across multiple entities.

        Args:
            all_entity_dfs: Dict of entity_id → entity normal event DataFrame.
            profiles: Dict of entity_id → EntityProfile.
            n_campaigns: Number of campaigns to inject.

        Returns:
            (attack_df, records) — all campaign rows and per-entity records.
        """
        cfg: CredentialStuffingConfig = self._config
        all_rows: List[Dict[str, Any]] = []
        records: List[AttackRecord] = []

        # Only target users per Section 3.3
        user_ids = [
            eid for eid, p in profiles.items()
            if p.entity_type == "user" and eid in all_entity_dfs and not all_entity_dfs[eid].empty
        ]
        if not user_ids:
            return pd.DataFrame(), []

        for _ in range(n_campaigns):
            campaign_ip = self._random_foreign_ip(self.rng)
            n_targets = int(self.rng.integers(
                cfg.n_entities_min,
                min(cfg.n_entities_max + 1, len(user_ids) + 1)
            ))
            target_indices = self.rng.choice(len(user_ids), size=min(n_targets, len(user_ids)), replace=False)
            target_ids = [user_ids[i] for i in target_indices]

            # Campaign start time: pick from any entity's timeline
            if not all_entity_dfs[target_ids[0]].empty:
                ref_row = all_entity_dfs[target_ids[0]].sample(n=1, random_state=int(self.rng.integers(0, 2**31))).iloc[0]
                campaign_start = datetime.fromisoformat(ref_row["timestamp"].replace("Z", "+00:00"))
            else:
                campaign_start = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)

            cumulative_offset = 0.0

            for eid in target_ids:
                entity_df = all_entity_dfs[eid]
                profile = profiles[eid]
                if entity_df.empty:
                    continue

                anchor = entity_df.sample(n=1, random_state=int(self.rng.integers(0, 2**31))).iloc[0]
                n_fail = int(self.rng.integers(cfg.n_fail_per_entity_min, cfg.n_fail_per_entity_max + 1))
                session_id = self._rng_uuid(self.rng)
                entity_event_ids: List[str] = []
                entity_timestamps: List[str] = []

                for k in range(n_fail):
                    stagger = float(self.rng.uniform(cfg.stagger_sec_min, cfg.stagger_sec_max))
                    cumulative_offset += stagger
                    ts = campaign_start + timedelta(seconds=cumulative_offset)
                    ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                    event_id = self._rng_uuid(self.rng)

                    row = {
                        **anchor.to_dict(),
                        "event_id": event_id,
                        "session_id": session_id,
                        "entity_id": eid,
                        "timestamp": ts_str,
                        "source_ip": campaign_ip,
                        "auth_outcome": "failure",
                        "auth_method": "password",
                        "failure_count": k + 1,
                        "session_duration": 0.0,
                        "command_sequence": "[]",
                        "label": AnomalyCategory.CREDENTIAL_STUFFING.value,
                    }
                    all_rows.append(row)
                    entity_event_ids.append(event_id)
                    entity_timestamps.append(ts_str)

                # Optional compromise per entity (probability 0.10)
                if self.rng.random() < cfg.compromise_probability:
                    cumulative_offset += float(self.rng.uniform(1, 5))
                    ts = campaign_start + timedelta(seconds=cumulative_offset)
                    ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                    event_id = self._rng_uuid(self.rng)
                    row = {
                        **anchor.to_dict(),
                        "event_id": event_id,
                        "session_id": self._rng_uuid(self.rng),
                        "entity_id": eid,
                        "timestamp": ts_str,
                        "source_ip": campaign_ip,
                        "auth_outcome": "success",
                        "failure_count": 0,
                        "label": AnomalyCategory.CREDENTIAL_STUFFING.value,
                    }
                    all_rows.append(row)
                    entity_event_ids.append(event_id)
                    entity_timestamps.append(ts_str)

                records.append(AttackRecord(
                    entity_id=eid,
                    attack_type="credential_stuffing",
                    event_ids=entity_event_ids,
                    timestamps=entity_timestamps,
                    extra={"campaign_ip": campaign_ip},
                ))

        attack_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
        return attack_df, records
