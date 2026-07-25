"""
Data Generator package for synthetic access log generation.

Public interface:
- GeneratorConfig: Configuration model for the generator.
- EntityProfile: Per-entity behavioral profile model.
- DataGenerator: Core generator class (M03 scope — normal events only).
- AttackInjectionConfig: Configuration model for attack injection.
- AttackInjector: M04 attack injection orchestrator.
- InjectionLog: Per-entity injection metadata.
"""

from anomaly_detection.data_generator.config import GeneratorConfig
from anomaly_detection.data_generator.entity_profiles import (
    EntityProfile,
    GeoPoint,
    DeviceRecord,
)
from anomaly_detection.data_generator.generator import DataGenerator
from anomaly_detection.data_generator.injection_config import AttackInjectionConfig
from anomaly_detection.data_generator.attack_injector import AttackInjector
from anomaly_detection.data_generator.attackers.base import InjectionLog

__all__ = [
    "GeneratorConfig",
    "EntityProfile",
    "GeoPoint",
    "DeviceRecord",
    "DataGenerator",
    "AttackInjectionConfig",
    "AttackInjector",
    "InjectionLog",
]
