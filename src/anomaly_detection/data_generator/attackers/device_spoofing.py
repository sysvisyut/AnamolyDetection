"""
Device Spoofing attacker.

Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.5.
ATTACK_TAXONOMY.md: Detection Difficulty = Medium; Primary Fields = device_fingerprint
(device_id same, MAC/OS/protocol changed).

Inserts 1-5 events where the entity's registered device_id appears with a mutated
fingerprint: Strategy A = MAC change, B = OS change, C = protocol change.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.data_generator.attackers.base import AttackRecord, BaseAttacker
from anomaly_detection.data_generator.entity_profiles import OS_FAMILY_CHOICES, OS_VERSION_MAP, EntityProfile
from anomaly_detection.data_generator.injection_config import DeviceSpoofingConfig

# Alternative OS families used for OS-spoof strategy
_SPOOF_OS_PAIRS = [
    ("Linux", "22.04"),
    ("Windows", "10.0"),
    ("macOS", "13.5"),
]

# Alternative protocols for protocol-spoof strategy
_SPOOF_PROTOCOLS = ["Modbus", "MQTT", "DNP3", "SSH", "FTP"]


class DeviceSpoofingAttacker(BaseAttacker):
    """Device Spoofing attack injector.

    Injects events using the entity's known device_id but with a mutated
    fingerprint (MAC, OS, or protocol change). The key signal is that
    device_id is known but fingerprint fields don't match registered values.

    Detection Difficulty: MEDIUM — fingerprint_mac_match and fingerprint_os_match
    feature dimensions will be 0.0 for a known device_id with changed fields.

    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3.5.
    """

    SEED_OFFSET = 5_000

    def inject(
        self,
        entity_df: pd.DataFrame,
        profile: EntityProfile,
        n_sessions: int,
    ) -> Tuple[pd.DataFrame, AttackRecord]:
        cfg: DeviceSpoofingConfig = self._config
        rows: List[Dict[str, Any]] = []
        event_ids: List[str] = []
        timestamps: List[str] = []

        if entity_df.empty or not profile.device_set or len(entity_df) < cfg.min_prior_events:
            return pd.DataFrame(), AttackRecord(
                entity_id=profile.entity_id,
                attack_type="device_spoofing",
                event_ids=[],
                timestamps=[],
            )

        # Use entity's primary registered device
        primary_device = profile.device_set[0]
        device_id = primary_device.device_id

        # Select spoof strategy per Section 3.5 weights (0.5, 0.3, 0.2)
        strategy_weights = list(cfg.strategy_weights)
        strategy_total = sum(strategy_weights)
        strategy_probs = [w / strategy_total for w in strategy_weights]
        strategy_idx = int(self.rng.choice(3, p=strategy_probs))
        strategy = ["mac_spoof", "os_spoof", "protocol_spoof"][strategy_idx]

        for _ in range(n_sessions):
            anchor_row = entity_df.sample(n=1, random_state=int(self.rng.integers(0, 2**31))).iloc[0]
            anchor_ts = datetime.fromisoformat(anchor_row["timestamp"].replace("Z", "+00:00"))

            n_spoof = int(self.rng.integers(cfg.n_spoof_events_min, cfg.n_spoof_events_max + 1))
            session_id = self._rng_uuid(self.rng)

            # Build spoofed fingerprint
            spoofed_fp = self._build_spoofed_fingerprint(primary_device, strategy)

            for k in range(n_spoof):
                ts = anchor_ts + timedelta(minutes=float(k * 5 + self.rng.uniform(0, 5)))
                ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                event_id = self._rng_uuid(self.rng)

                row = {
                    **anchor_row.to_dict(),
                    "event_id": event_id,
                    "session_id": session_id,
                    "timestamp": ts_str,
                    "device_fingerprint": json.dumps(spoofed_fp),
                    "auth_outcome": "success",
                    "failure_count": 0,
                    "label": AnomalyCategory.DEVICE_SPOOFING.value,
                    "cold_start_flag": profile.is_late_joiner,
                }
                rows.append(row)
                event_ids.append(event_id)
                timestamps.append(ts_str)

        attack_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        return attack_df, AttackRecord(
            entity_id=profile.entity_id,
            attack_type="device_spoofing",
            event_ids=event_ids,
            timestamps=timestamps,
            extra={"strategy": strategy, "device_id": device_id},
        )

    def _build_spoofed_fingerprint(self, device, strategy: str) -> Dict[str, Any]:
        """Build a spoofed device fingerprint dict based on the spoof strategy."""
        seed_int = int(self.rng.integers(0, 2**31))
        new_mac = self._fake_mac(seed_int)

        if strategy == "mac_spoof":
            # Strategy A: same device_id, OS, protocol; different MAC
            return {
                "device_id": device.device_id,
                "os_family": device.os_family,
                "os_version": device.os_version,
                "mac_address": new_mac,
                "protocol": device.protocol,
                "user_agent": device.user_agent,
                "firmware_version": device.firmware_version,
            }
        elif strategy == "os_spoof":
            # Strategy B: same device_id and MAC; different OS family + version
            available = [pair for pair in _SPOOF_OS_PAIRS if pair[0] != device.os_family]
            if not available:
                available = _SPOOF_OS_PAIRS
            os_pair = available[int(self.rng.integers(0, len(available)))]
            return {
                "device_id": device.device_id,
                "os_family": os_pair[0],
                "os_version": os_pair[1],
                "mac_address": device.mac_address,
                "protocol": device.protocol,
                "user_agent": f"Mozilla/5.0 ({os_pair[0]} NT {os_pair[1]})",
                "firmware_version": device.firmware_version,
            }
        else:
            # Strategy C: same device_id; different protocol
            available = [p for p in _SPOOF_PROTOCOLS if p != device.protocol]
            if not available:
                available = _SPOOF_PROTOCOLS
            new_protocol = available[int(self.rng.integers(0, len(available)))]
            return {
                "device_id": device.device_id,
                "os_family": device.os_family,
                "os_version": device.os_version,
                "mac_address": device.mac_address,
                "protocol": new_protocol,
                "user_agent": device.user_agent,
                "firmware_version": device.firmware_version,
            }

    @staticmethod
    def _fake_mac(seed: int) -> str:
        """Generate a deterministic fake MAC address from a seed."""
        h = format(abs(seed) % (16**12), "012x")
        return ":".join(h[i:i+2].upper() for i in range(0, 12, 2))
