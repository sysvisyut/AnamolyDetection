"""
Lateral Movement attacker.

Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.4.
ATTACK_TAXONOMY.md: Detection Difficulty = Medium; Primary Fields = resource_accessed,
command_sequence, session_duration.

Injects a single compromised session with 10-30 events accessing resources
entirely outside the entity's ResourceSet, spanning ≥3 resource categories,
with a recon-to-exfil command sequence.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.data_generator.attackers.base import AttackRecord, BaseAttacker
from anomaly_detection.data_generator.entity_profiles import (
    ALL_RESOURCES,
    API_RESOURCES,
    DB_RESOURCES,
    FILE_RESOURCES,
    PORT_RESOURCES,
    EntityProfile,
)
from anomaly_detection.data_generator.injection_config import LateralMovementConfig

# Reconnaissance → access → exfil command progression
_RECON_CMDS = ["ls", "find", "grep", "ps"]
_ACCESS_CMDS = ["cat", "ssh", "curl", "netstat"]
_EXFIL_CMDS = ["scp", "rsync", "wget", "tar"]

# Resource category pools
_CATEGORY_POOLS = {
    "file": FILE_RESOURCES,
    "api": API_RESOURCES,
    "port": PORT_RESOURCES,
    "db": DB_RESOURCES,
}


class LateralMovementAttacker(BaseAttacker):
    """Lateral Movement attack injector.

    In a single compromised session, injects events accessing resources
    outside the entity's ResourceSet across ≥3 distinct resource categories,
    with a recon → access → exfil command sequence.

    Detection Difficulty: MEDIUM — resource_rarity_score and has_exfil_command
    are strong signals; developer persona legitimate cross-service work narrows margin.

    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.4.
    """

    SEED_OFFSET = 4_000

    def inject(
        self,
        entity_df: pd.DataFrame,
        profile: EntityProfile,
        n_sessions: int,
    ) -> Tuple[pd.DataFrame, AttackRecord]:
        cfg: LateralMovementConfig = self._config
        rows: List[Dict[str, Any]] = []
        event_ids: List[str] = []
        timestamps: List[str] = []

        if entity_df.empty:
            return pd.DataFrame(), AttackRecord(
                entity_id=profile.entity_id,
                attack_type="lateral_movement",
                event_ids=[],
                timestamps=[],
            )

        entity_resource_set = set(profile.resource_set)

        for _ in range(n_sessions):
            # Build expansion resource list outside entity's ResourceSet
            expansion_resources = self._build_expansion_resources(
                entity_resource_set, cfg
            )
            if not expansion_resources:
                continue

            # Build recon-to-exfil command sequence
            cmd_length = int(self.rng.integers(cfg.cmd_length_min, cfg.cmd_length_max + 1))
            command_sequence = self._build_lateral_command_sequence(cmd_length)

            # Choose anchor timestamp from entity's timeline
            anchor_row = entity_df.sample(n=1, random_state=int(self.rng.integers(0, 2**31))).iloc[0]
            anchor_ts = datetime.fromisoformat(anchor_row["timestamp"].replace("Z", "+00:00"))
            session_id = self._rng_uuid(self.rng)

            n_expand = min(len(expansion_resources), int(self.rng.integers(cfg.n_expand_min, cfg.n_expand_max + 1)))
            elapsed = 0.0

            for i in range(n_expand):
                gap = float(self.rng.uniform(cfg.inter_event_sec_min, cfg.inter_event_sec_max))
                ts = anchor_ts + timedelta(seconds=elapsed + gap)
                elapsed += gap
                ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                event_id = self._rng_uuid(self.rng)

                # Spread command sequence across events
                cmd_slice_start = i * len(command_sequence) // n_expand
                cmd_slice_end = (i + 1) * len(command_sequence) // n_expand
                event_cmds = command_sequence[cmd_slice_start:cmd_slice_end]

                row = {
                    **anchor_row.to_dict(),
                    "event_id": event_id,
                    "session_id": session_id,
                    "timestamp": ts_str,
                    "resource_accessed": expansion_resources[i % len(expansion_resources)],
                    "auth_outcome": "success",
                    "failure_count": 0,
                    "session_duration": min(float(elapsed), 7200.0),
                    "command_sequence": json.dumps(event_cmds),
                    "label": AnomalyCategory.LATERAL_MOVEMENT.value,
                }
                rows.append(row)
                event_ids.append(event_id)
                timestamps.append(ts_str)

        attack_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        return attack_df, AttackRecord(
            entity_id=profile.entity_id,
            attack_type="lateral_movement",
            event_ids=event_ids,
            timestamps=timestamps,
        )

    def _build_expansion_resources(
        self, entity_resource_set: set, cfg: LateralMovementConfig
    ) -> List[str]:
        """Build expansion resources outside entity's set spanning ≥3 categories."""
        # Sample from each category pool to ensure category diversity
        categories = list(_CATEGORY_POOLS.keys())
        self.rng.shuffle(categories)
        expansion: List[str] = []
        used_categories = 0

        for cat in categories:
            pool = [r for r in _CATEGORY_POOLS[cat] if r not in entity_resource_set]
            if pool:
                n_from_cat = max(2, int(self.rng.integers(2, max(3, len(pool) // 4))))
                indices = self.rng.choice(len(pool), size=min(n_from_cat, len(pool)), replace=False)
                expansion.extend([pool[i] for i in indices])
                used_categories += 1

            if used_categories >= cfg.min_categories and len(expansion) >= cfg.n_expand_min:
                break

        return expansion

    def _build_lateral_command_sequence(self, length: int) -> List[Dict[str, Any]]:
        """Build recon → access → exfil command progression."""
        n_recon = max(1, length // 3)
        n_access = max(1, length // 3)
        n_exfil = length - n_recon - n_access

        cmds: List[Dict[str, Any]] = []
        elapsed = 0.0
        pos = 0

        for cmd_name in (
            [_RECON_CMDS[int(self.rng.integers(0, len(_RECON_CMDS)))] for _ in range(n_recon)]
            + [_ACCESS_CMDS[int(self.rng.integers(0, len(_ACCESS_CMDS)))] for _ in range(n_access)]
            + [_EXFIL_CMDS[int(self.rng.integers(0, len(_EXFIL_CMDS)))] for _ in range(max(1, n_exfil))]
        ):
            step = float(self.rng.gamma(shape=2, scale=30))
            elapsed += step
            cmds.append({
                "sequence_position": pos,
                "command": cmd_name,
                "target": f"target_{self._rng_uuid(self.rng)[:8]}",
                "outcome": "success",
                "elapsed_seconds": round(elapsed, 2),
            })
            pos += 1

        return cmds
