"""
Sequence Detection Model (SDM) Module.

Provides the PyTorch GRU Autoencoder and associated inference engine.
M08 (Phase 14) implementation.
"""

from anomaly_detection.models.sequence_detection.config import DetectionModelConfig
from anomaly_detection.models.sequence_detection.base import GRUAutoencoder
from anomaly_detection.models.sequence_detection.dataset import SequenceDataset, create_dataloader, build_tensor_from_sequence
from anomaly_detection.models.sequence_detection.trainer import DetectionTrainer
from anomaly_detection.models.sequence_detection.inference import SDMInferenceEngine
from anomaly_detection.models.sequence_detection.attribution_interface import AttributionInterface, GRUIntegratedGradientsAttribution

__all__ = [
    "DetectionModelConfig",
    "GRUAutoencoder",
    "SequenceDataset",
    "create_dataloader",
    "build_tensor_from_sequence",
    "DetectionTrainer",
    "SDMInferenceEngine",
    "AttributionInterface",
    "GRUIntegratedGradientsAttribution",
]
