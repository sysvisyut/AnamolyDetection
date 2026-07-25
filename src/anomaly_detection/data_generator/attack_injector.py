"""
Attack Injection orchestrator for M04.

Wraps M03's DataGenerator output and decorates the normal event stream
with labeled attack sessions. All injection is additive — normal records
are never deleted or modified.

Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3 (Global Injection Framework).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from anomaly_detection.data_generator.attackers.base import AttackRecord, InjectionLog
from anomaly_detection.data_generator.attackers.brute_force import BruteForceAttacker
from anomaly_detection.data_generator.attackers.credential_stuffing import CredentialStuffingAttacker
from anomaly_detection.data_generator.attackers.device_spoofing import DeviceSpoofingAttacker
from anomaly_detection.data_generator.attackers.impossible_travel import ImpossibleTravelAttacker
from anomaly_detection.data_generator.attackers.insider_drift import InsiderDriftAttacker
from anomaly_detection.data_generator.attackers.lateral_movement import LateralMovementAttacker
from anomaly_detection.data_generator.attackers.low_and_slow import LowAndSlowAttacker
from anomaly_detection.data_generator.entity_profiles import EntityProfile
from anomaly_detection.data_generator.injection_config import AttackInjectionConfig

# Per-attack-type minimum prior events required before targeting
_MIN_PRIOR_EVENTS: Dict[str, int] = {
    "brute_force": 5,
    "impossible_travel": 1,
    "credential_stuffing": 1,
    "lateral_movement": 5,
    "device_spoofing": 10,
    "low_and_slow": 10,
    "insider_drift": 5,
}

# Entity types eligible for each attack type
_ELIGIBLE_ENTITY_TYPES: Dict[str, List[str]] = {
    "brute_force": ["user", "service_account"],
    "impossible_travel": ["user"],
    "credential_stuffing": ["user"],
    "lateral_movement": ["user", "service_account"],
    "device_spoofing": ["user", "edge_device"],
    "low_and_slow": ["user"],
    "insider_drift": ["user"],
}


def _inject_attacker_config(attacker, config_obj) -> None:
    """Attach a config object to an attacker instance."""
    attacker._config = config_obj


class AttackInjector:
    """Orchestrates attack injection over a normal event DataFrame.

    Consumes M03 output and produces a mixed dataset (normal + attack records)
    conforming to AccessLogTraining schema. All injection is additive: normal
    rows are never modified or removed.

    Usage:
        injector = AttackInjector(config)
        mixed_df, log = injector.inject(normal_df, entity_profiles)
    """

    def __init__(self, config: Optional[AttackInjectionConfig] = None) -> None:
        """Initialize with optional injection config. Uses defaults if not provided."""
        self.config = config or AttackInjectionConfig()

        # Instantiate per-attack-type attackers with derived seeds
        seed = self.config.random_seed
        self._bf = BruteForceAttacker(seed)
        self._it = ImpossibleTravelAttacker(seed)
        self._cs = CredentialStuffingAttacker(seed)
        self._lm = LateralMovementAttacker(seed)
        self._ds = DeviceSpoofingAttacker(seed)
        self._las = LowAndSlowAttacker(seed)
        self._id = InsiderDriftAttacker(seed)

        # Attach per-type config objects
        _inject_attacker_config(self._bf, self.config.brute_force)
        _inject_attacker_config(self._it, self.config.impossible_travel)
        _inject_attacker_config(self._cs, self.config.credential_stuffing)
        _inject_attacker_config(self._lm, self.config.lateral_movement)
        _inject_attacker_config(self._ds, self.config.device_spoofing)
        _inject_attacker_config(self._las, self.config.low_and_slow)
        _inject_attacker_config(self._id, self.config.insider_drift)

    def inject(
        self,
        normal_df: pd.DataFrame,
        entity_profiles: Dict[str, EntityProfile],
    ) -> Tuple[pd.DataFrame, InjectionLog]:
        """Inject attack events into the normal event stream.

        Args:
            normal_df: Complete normal event DataFrame from M03 (read-only).
            entity_profiles: Dict mapping entity_id → EntityProfile from M03.

        Returns:
            (mixed_df, log): mixed_df is the full dataset (normal + attacks)
            sorted by timestamp. log is the per-entity injection metadata.

        Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3, Global Injection Framework.
        """
        n_total = len(normal_df)
        if n_total == 0:
            return normal_df.copy(), InjectionLog()

        # Compute anomaly budget
        n_target_anomaly = round(n_total * self.config.target_anomaly_rate)
        n_target_anomaly = max(
            round(n_total * self.config.anomaly_rate_min),
            min(n_target_anomaly, round(n_total * self.config.anomaly_rate_max)),
        )

        shares = self.config.normalized_shares()
        log = InjectionLog()
        attack_dfs: List[pd.DataFrame] = []

        # Pre-compute per-entity event slices
        entity_dfs: Dict[str, pd.DataFrame] = {
            eid: normal_df[normal_df["entity_id"] == eid].copy()
            for eid in entity_profiles
        }

        # Allocate attack budgets per type
        budgets: Dict[str, int] = {
            attack_type: max(1, round(n_target_anomaly * share))
            for attack_type, share in shares.items()
            if share > 0
        }

        # ----------------------------------------------------------------
        # 1. Brute Force
        # ----------------------------------------------------------------
        if budgets.get("brute_force", 0) > 0:
            targets = self._select_targets("brute_force", entity_profiles, entity_dfs)
            n_targets = max(1, min(len(targets), budgets["brute_force"] // 20 + 1))
            for eid in targets[:n_targets]:
                df_slice, record = self._bf.inject(entity_dfs[eid], entity_profiles[eid], 1)
                if not df_slice.empty:
                    attack_dfs.append(df_slice)
                    log.add(record)

        # ----------------------------------------------------------------
        # 2. Impossible Travel
        # ----------------------------------------------------------------
        if budgets.get("impossible_travel", 0) > 0:
            targets = self._select_targets("impossible_travel", entity_profiles, entity_dfs)
            n_targets = max(1, min(len(targets), budgets["impossible_travel"]))
            for eid in targets[:n_targets]:
                df_slice, record = self._it.inject(entity_dfs[eid], entity_profiles[eid], 1)
                if not df_slice.empty:
                    attack_dfs.append(df_slice)
                    log.add(record)

        # ----------------------------------------------------------------
        # 3. Credential Stuffing (campaign-level)
        # ----------------------------------------------------------------
        if budgets.get("credential_stuffing", 0) > 0:
            n_campaigns = max(1, budgets["credential_stuffing"] // 30)
            cs_df, cs_records = self._cs.inject_campaign(entity_dfs, entity_profiles, n_campaigns)
            if not cs_df.empty:
                attack_dfs.append(cs_df)
                for r in cs_records:
                    log.add(r)

        # ----------------------------------------------------------------
        # 4. Lateral Movement
        # ----------------------------------------------------------------
        if budgets.get("lateral_movement", 0) > 0:
            targets = self._select_targets("lateral_movement", entity_profiles, entity_dfs)
            n_targets = max(1, min(len(targets), budgets["lateral_movement"] // 15 + 1))
            for eid in targets[:n_targets]:
                df_slice, record = self._lm.inject(entity_dfs[eid], entity_profiles[eid], 1)
                if not df_slice.empty:
                    attack_dfs.append(df_slice)
                    log.add(record)

        # ----------------------------------------------------------------
        # 5. Device Spoofing
        # ----------------------------------------------------------------
        if budgets.get("device_spoofing", 0) > 0:
            targets = self._select_targets("device_spoofing", entity_profiles, entity_dfs)
            n_targets = max(1, min(len(targets), budgets["device_spoofing"] // 3 + 1))
            for eid in targets[:n_targets]:
                df_slice, record = self._ds.inject(entity_dfs[eid], entity_profiles[eid], 1)
                if not df_slice.empty:
                    attack_dfs.append(df_slice)
                    log.add(record)

        # ----------------------------------------------------------------
        # 6. Low-and-Slow
        # ----------------------------------------------------------------
        if budgets.get("low_and_slow", 0) > 0:
            targets = self._select_targets("low_and_slow", entity_profiles, entity_dfs)
            n_targets = max(1, min(len(targets), budgets["low_and_slow"] // 10 + 1))
            for eid in targets[:n_targets]:
                df_slice, record = self._las.inject(entity_dfs[eid], entity_profiles[eid], 1)
                if not df_slice.empty:
                    attack_dfs.append(df_slice)
                    log.add(record)

        # ----------------------------------------------------------------
        # 7. Insider Drift
        # ----------------------------------------------------------------
        if budgets.get("insider_drift", 0) > 0:
            targets = self._select_targets("insider_drift", entity_profiles, entity_dfs)
            n_targets = max(1, min(len(targets), budgets["insider_drift"] // 10 + 1))
            for eid in targets[:n_targets]:
                df_slice, record = self._id.inject(entity_dfs[eid], entity_profiles[eid], 1)
                if not df_slice.empty:
                    attack_dfs.append(df_slice)
                    log.add(record)

        # ----------------------------------------------------------------
        # Combine and sort
        # ----------------------------------------------------------------
        if attack_dfs:
            attack_combined = pd.concat(attack_dfs, ignore_index=True)
            mixed_df = pd.concat([normal_df, attack_combined], ignore_index=True)
        else:
            mixed_df = normal_df.copy()

        mixed_df = mixed_df.sort_values("timestamp").reset_index(drop=True)
        return mixed_df, log

    def _select_targets(
        self,
        attack_type: str,
        profiles: Dict[str, EntityProfile],
        entity_dfs: Dict[str, pd.DataFrame],
    ) -> List[str]:
        """Select eligible entity IDs for a given attack type.

        Filters by entity type eligibility and minimum prior event count.
        Shuffles the result deterministically using the configured seed.
        """
        eligible_types = _ELIGIBLE_ENTITY_TYPES.get(attack_type, ["user"])
        min_events = _MIN_PRIOR_EVENTS.get(attack_type, 1)

        eligible = [
            eid
            for eid, profile in profiles.items()
            if profile.entity_type in eligible_types
            and eid in entity_dfs
            and len(entity_dfs[eid]) >= min_events
        ]

        # Deterministic shuffle using attack-type-specific seed
        rng = np.random.default_rng(seed=self.config.random_seed + hash(attack_type) % (2**16))
        indices = rng.permutation(len(eligible))
        return [eligible[i] for i in indices]
