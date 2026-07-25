"""
Base interface for all attack injectors.

Every attack type implements BaseAttacker to ensure the Open/Closed Principle:
new attack types can be added without modifying existing attackers.

Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3 (Global Injection Framework).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from anomaly_detection.data_generator.entity_profiles import EntityProfile


@dataclass
class AttackRecord:
    """Metadata about a single attack injection event for the InjectionLog."""

    entity_id: str
    attack_type: str
    event_ids: List[str]
    timestamps: List[str]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InjectionLog:
    """Per-entity injection log for ground-truth verification.

    Records which entities received which attack types and which event
    timestamps were affected. Used downstream for evaluation metrics.

    Acceptance criterion: every injected event_id must appear in this log.
    """

    records: List[AttackRecord] = field(default_factory=list)

    def add(self, record: AttackRecord) -> None:
        """Append an attack record to the log."""
        self.records.append(record)

    def all_event_ids(self) -> List[str]:
        """Return a flat list of every injected event_id."""
        ids: List[str] = []
        for r in self.records:
            ids.extend(r.event_ids)
        return ids

    def entities_attacked(self) -> Dict[str, List[str]]:
        """Return mapping of entity_id → list of attack_types received."""
        result: Dict[str, List[str]] = {}
        for r in self.records:
            result.setdefault(r.entity_id, []).append(r.attack_type)
        return result

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize the log to a list of dicts for storage."""
        return [
            {
                "entity_id": r.entity_id,
                "attack_type": r.attack_type,
                "event_ids": r.event_ids,
                "timestamps": r.timestamps,
                **r.extra,
            }
            for r in self.records
        ]


class BaseAttacker(ABC):
    """Abstract base class for all attack injectors.

    Contract every concrete attacker must fulfill:
    - inject() accepts a normal_df slice for the target entity and the EntityProfile
    - inject() returns (attack_rows_df, AttackRecord) — it NEVER modifies normal_df
    - All generated rows must conform to AccessLogTraining schema
    - All generated rows must have the correct AnomalyCategory label

    Reference: SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 3 Global Injection Framework.
    """

    # Attacker-type-specific seed offset (override in each subclass).
    SEED_OFFSET: int = 0

    def __init__(self, base_seed: int) -> None:
        """Initialize with a base seed for reproducible generation.

        Args:
            base_seed: The global random seed from GeneratorConfig.
                       Each attacker derives its own seed via SEED_OFFSET.
        """
        self.rng = np.random.default_rng(seed=base_seed + self.SEED_OFFSET)

    @abstractmethod
    def inject(
        self,
        entity_df: pd.DataFrame,
        profile: EntityProfile,
        n_sessions: int,
    ) -> Tuple[pd.DataFrame, AttackRecord]:
        """Inject attack events for a single target entity.

        Args:
            entity_df: All normal events for this entity (read-only).
            profile: The entity's behavioral profile from M03.
            n_sessions: Number of attack sessions/bursts to inject.

        Returns:
            (attack_df, record): attack_df contains only new attack rows
            (never modifications of entity_df); record logs what was injected.
        """

    @staticmethod
    def _rng_uuid(rng: np.random.Generator) -> str:
        """Generate a deterministic UUID-formatted string from the seeded RNG."""
        raw = rng.integers(0, 256, size=16, dtype=np.uint8)
        raw[6] = (raw[6] & 0x0F) | 0x40
        raw[8] = (raw[8] & 0x3F) | 0x80
        h = raw.tobytes().hex()
        return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    @staticmethod
    def _random_foreign_ip(rng: np.random.Generator) -> str:
        """Generate a random foreign IPv4 address."""
        return (
            f"{int(rng.integers(1, 223))}"
            f".{int(rng.integers(0, 256))}"
            f".{int(rng.integers(0, 256))}"
            f".{int(rng.integers(1, 255))}"
        )
