"""
Inference engine for the Sequence Detection Model.
Handles loading saved artifacts, running the forward pass, and generating DetectionOutput.
"""

from typing import Dict, Optional
import torch

from anomaly_detection.common.logging import setup_logger
from anomaly_detection.common.models.features import EngineeredFeatures, SessionMetadata
from anomaly_detection.common.models.ml_io import DetectionOutput
from anomaly_detection.models.sequence_detection.base import GRUAutoencoder
from anomaly_detection.models.sequence_detection.dataset import build_tensor_from_sequence
from anomaly_detection.models.sequence_detection.config import FEATURE_NAMES

logger = setup_logger("sdm_inference")


class SDMInferenceEngine:
    """
    Wraps the trained GRU Autoencoder to provide the inference API.
    Handles cold-start logic and normalization per ML_PIPELINE.md.
    """
    
    def __init__(self, model_path: str):
        """
        Loads the model from disk and sets it to eval mode.
        """
        self.device = torch.device("cpu")  # Enforce CPU for latency consistency in T1
        
        try:
            self.model = GRUAutoencoder.load(model_path).to(self.device)
            self.model.eval()
            self.config = self.model.config
            logger.info(f"Loaded SDM from {model_path}")
        except FileNotFoundError:
            logger.error(f"SDM artifact not found at {model_path}. Must train first.")
            raise
            
    def _normalize_score(self, raw_score: float) -> float:
        """
        Clips and normalizes the raw reconstruction error to [0, 1] using
        percentiles learned during training.
        """
        err_min = self.config.calibration_err_min
        err_max = self.config.calibration_err_max
        
        if err_max <= err_min:
            # Fallback if calibration failed (e.g. all 0 errors)
            return 0.0
            
        clipped = max(err_min, min(raw_score, err_max))
        normalized = (clipped - err_min) / (err_max - err_min)
        
        return normalized

    @torch.no_grad()
    def predict(self, features: EngineeredFeatures) -> DetectionOutput:
        """
        Runs the full inference pipeline for a single event.
        Returns a DetectionOutput compliant with Boundary F.
        """
        # 1. Build input tensors
        seq_tensor, mask_tensor = build_tensor_from_sequence(features.sequence_window, self.config)
        
        # Add batch dimension
        seq_tensor = seq_tensor.unsqueeze(0).to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).to(self.device)
        
        # 2. Forward pass
        recon = self.model(seq_tensor, mask_tensor)
        
        # 3. Compute masked error per feature
        # err shape: (1, W, F)
        err = (recon - seq_tensor) ** 2
        
        # mask shape: (1, W, 1)
        mask_expanded = mask_tensor.unsqueeze(-1)
        masked_err = err * mask_expanded
        
        # Extract valid timesteps (valid_len, F)
        valid_len = mask_tensor[0].sum().item()
        
        if valid_len > 0:
            valid_errs = masked_err[0, :valid_len, :]
            
            # Mean error across timesteps per feature: shape (F,)
            feature_errors = valid_errs.mean(dim=0)
            
            # Aggregate to single score per ML_PIPELINE.md §3.6
            mean_err = valid_errs.mean().item()
            max_err = valid_errs.max().item()
            raw_score = (
                self.config.raw_score_weight_mean * mean_err + 
                self.config.raw_score_weight_max * max_err
            )
            
            # Top contributing features
            # Get indices of top 5 errors
            top_indices = torch.topk(feature_errors, k=min(5, self.config.feature_dim)).indices.tolist()
            top_features = [FEATURE_NAMES[idx] for idx in top_indices]
            
        else:
            # Fallback for completely empty sequences
            raw_score = 0.0
            top_features = []
            
        # 4. Normalize score
        normalized_score = self._normalize_score(raw_score)
        
        # 5. Apply cold-start discounting
        # COLDSTART_DRIFT_STRATEGY.md §2
        is_cold_start = features.session_metadata.is_cold_start
        confidence = 0.95  # Default high confidence for SDM
        
        # Also check internal padding condition: if mostly padding, force cold-start behavior
        pad_fraction = 1.0 - (valid_len / self.config.window_size)
        if pad_fraction > self.config.cold_start_padding_threshold:
            is_cold_start = True
            
        if is_cold_start:
            normalized_score *= self.config.cold_start_score_factor
            confidence = min(confidence, self.config.cold_start_confidence_cap)
            
        # 6. Construct output
        return DetectionOutput(
            entity_id=features.entity_id,
            event_id=features.event_id,
            model_id="sdm",
            anomaly_score=normalized_score,
            confidence=confidence,
            cold_start_flag=is_cold_start,
            top_contributing_features=top_features
        )
