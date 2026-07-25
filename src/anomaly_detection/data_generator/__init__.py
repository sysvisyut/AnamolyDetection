"""
Data Generator package for synthetic access log generation.

Public interface:
- GeneratorConfig: Configuration model for the generator.
- EntityProfile: Per-entity behavioral profile model.
- DataGenerator: Core generator class (M03 scope — normal events only).
"""

from anomaly_detection.data_generator.config import GeneratorConfig
from anomaly_detection.data_generator.entity_profiles import (
    EntityProfile,
    GeoPoint,
    DeviceRecord,
)
from anomaly_detection.data_generator.generator import DataGenerator

__all__ = [
    "GeneratorConfig",
    "EntityProfile",
    "GeoPoint",
    "DeviceRecord",
    "DataGenerator",
]
