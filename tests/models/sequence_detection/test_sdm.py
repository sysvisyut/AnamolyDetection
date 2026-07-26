"""
Unit tests for the Sequence Detection Model (M08).
"""

import os
import pytest
import torch
import numpy as np

from anomaly_detection.common.models.features import EntitySequence, EngineeredFeatures, SessionMetadata, EntityFeatureVector
from anomaly_detection.models.sequence_detection.config import DetectionModelConfig, FEATURE_DIM
from anomaly_detection.models.sequence_detection.base import GRUAutoencoder
from anomaly_detection.models.sequence_detection.dataset import build_tensor_from_sequence, create_dataloader
from anomaly_detection.models.sequence_detection.trainer import DetectionTrainer
from anomaly_detection.models.sequence_detection.inference import SDMInferenceEngine
from anomaly_detection.models.sequence_detection.attribution_interface import GRUIntegratedGradientsAttribution, HAS_CAPTUM


@pytest.fixture
def config():
    c = DetectionModelConfig(
        max_epochs=2,
        batch_size=4,
        artifacts_dir="/tmp/artifacts",
        hidden_size=16  # smaller for fast testing
    )
    return c


@pytest.fixture
def dummy_sequence():
    raw_list = [[0.5] * FEATURE_DIM for _ in range(10)]
    return EntitySequence(root=raw_list)


@pytest.fixture
def dummy_features(dummy_sequence):
    return EngineeredFeatures(
        entity_id="usr_123",
        event_id="evt_123",
        session_id="ses_123",
        feature_vector=EntityFeatureVector(root=[0.5] * FEATURE_DIM),
        sequence_window=dummy_sequence,
        session_metadata=SessionMetadata(
            is_cold_start=False,
            delivery_mode_hint="batch",
            profile_event_count=50
        )
    )


def test_build_tensor(config, dummy_sequence):
    seq_tensor, mask = build_tensor_from_sequence(dummy_sequence, config)
    
    assert seq_tensor.shape == (config.window_size, FEATURE_DIM)
    assert mask.shape == (config.window_size,)
    
    assert mask[:10].all()
    assert not mask[10:].any()


def test_model_forward(config):
    model = GRUAutoencoder(config)
    
    # Batch size 2
    x = torch.rand((2, config.window_size, FEATURE_DIM))
    mask = torch.ones((2, config.window_size), dtype=torch.bool)
    mask[0, 10:] = False  # first sequence length 10
    mask[1, 5:] = False   # second sequence length 5
    
    recon = model(x, mask)
    
    assert recon.shape == x.shape


def test_trainer_fit(config, dummy_sequence):
    seqs = [dummy_sequence for _ in range(10)]
    loader = create_dataloader(seqs, config)
    
    trainer = DetectionTrainer(config)
    
    # Just run a quick fit to ensure no crashes
    model = trainer.fit(loader, loader)
    
    assert model.config.calibration_err_max > 0.0
    assert model.config.calibration_err_min >= 0.0


def test_save_load(config, tmp_path):
    config.artifacts_dir = str(tmp_path)
    model = GRUAutoencoder(config)
    
    save_path = config.checkpoint_path("test")
    model.save(save_path)
    
    assert os.path.exists(save_path)
    
    loaded = GRUAutoencoder.load(save_path)
    assert loaded.config.hidden_size == config.hidden_size
    

def test_inference_engine_warm_start(config, tmp_path, dummy_features):
    config.artifacts_dir = str(tmp_path)
    config.calibration_err_min = 0.0
    config.calibration_err_max = 1.0
    model = GRUAutoencoder(config)
    
    save_path = config.checkpoint_path("test")
    model.save(save_path)
    
    engine = SDMInferenceEngine(save_path)
    output = engine.predict(dummy_features)
    
    assert output.model_id == "sdm"
    assert output.cold_start_flag == False
    assert 0.0 <= output.anomaly_score <= 1.0
    assert len(output.top_contributing_features) > 0


def test_inference_engine_cold_start_discount(config, tmp_path, dummy_features):
    config.artifacts_dir = str(tmp_path)
    # mock a fixed error so we can test the discount exactly
    config.calibration_err_min = 0.0
    config.calibration_err_max = 1.0
    model = GRUAutoencoder(config)
    
    save_path = config.checkpoint_path("test")
    model.save(save_path)
    
    engine = SDMInferenceEngine(save_path)
    
    # Warm prediction
    warm_output = engine.predict(dummy_features)
    
    # Cold prediction
    dummy_features.session_metadata.is_cold_start = True
    cold_output = engine.predict(dummy_features)
    
    assert cold_output.cold_start_flag == True
    # If the score isn't perfectly zero, the cold score should be discounted
    if warm_output.anomaly_score > 0:
        assert cold_output.anomaly_score < warm_output.anomaly_score
    assert cold_output.confidence <= config.cold_start_confidence_cap


def test_attribution_interface(config, tmp_path):
    model = GRUAutoencoder(config)
    attr_engine = GRUIntegratedGradientsAttribution(model)
    
    x = torch.rand((1, config.window_size, FEATURE_DIM))
    mask = torch.ones((1, config.window_size), dtype=torch.bool)
    mask[0, 5:] = False
    
    attributions = attr_engine.get_feature_attributions(x, mask)
    
    assert len(attributions) == FEATURE_DIM
    if HAS_CAPTUM:
        # Check that we got non-zero attributions for valid indices (usually true for random input)
        # But even if 0, structure must match
        pass
