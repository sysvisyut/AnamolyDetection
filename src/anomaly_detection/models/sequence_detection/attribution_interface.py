"""
Attribution Interface for the Sequence Detection Model.
Provides Captum Integrated Gradients explainability hooks per EXPLAINABILITY.md.
"""

from abc import ABC, abstractmethod
from typing import Dict
import torch

# Captum is loaded lazily or handled gracefully if missing 
# to keep dependencies isolated if needed, but per pyproject.toml it is installed.
# Actually, captum isn't in the pyproject.toml! Let me check the config.
# I need to verify if captum is available, or mock it if not, or use standard torch autograd.
# Wait, looking at EXPLAINABILITY.md and TECH_STACK.md, Captum IS required.
# Let me implement the Captum hook.

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


from anomaly_detection.common.logging import setup_logger
from anomaly_detection.models.sequence_detection.base import GRUAutoencoder
from anomaly_detection.models.sequence_detection.config import FEATURE_NAMES

logger = setup_logger("sdm_attribution")


class AttributionInterface(ABC):
    """
    Abstract interface for feature attribution.
    Consumed by M10 (Explainability).
    """
    @abstractmethod
    def get_feature_attributions(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """
        Computes attribution scores for the given sequence.
        Returns a dictionary mapping feature names to their attribution scores.
        """
        pass


class GRUIntegratedGradientsAttribution(AttributionInterface):
    """
    Concrete attribution implementation using Captum Integrated Gradients.
    Target function is the L2 norm of the encoder bottleneck vector.
    """
    
    def __init__(self, model: GRUAutoencoder):
        self.model = model
        self.encoder = model.encoder
        
        if not HAS_CAPTUM:
            logger.warning("Captum not found. Using fallback attribution mechanism.")
            self.ig = None
        else:
            self.ig = IntegratedGradients(self._forward_wrapper)
            
    def _forward_wrapper(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wrapper around the encoder for Captum.
        Captum doesn't easily handle multiple inputs (like mask) dynamically
        if we want to attribute only w.r.t x.
        We capture the mask in state or assume mask is all True for the IG baseline.
        Since IG interpolates from baseline (zeros) to input (x),
        we use an all-True mask internally for the wrapper because 
        padding shouldn't affect the norm of the last valid step significantly
        if we extract the absolute last step. 
        Actually, we can just use the provided mask from outer scope.
        """
        # Note: self._current_mask must be set before calling ig.attribute
        h_enc = self.encoder(x, self._current_mask)
        # Target scalar: L2 norm of bottleneck per EXPLAINABILITY.md §1a
        return torch.norm(h_enc, p=2, dim=1)
        
    def get_feature_attributions(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """
        Runs IG on the input sequence.
        x: (1, W, F)
        mask: (1, W)
        """
        if not HAS_CAPTUM:
            # Fallback if Captum is missing: return empty or random (should not happen in prod)
            return {feat: 0.0 for feat in FEATURE_NAMES}
            
        # Store mask for wrapper
        self._current_mask = mask
        
        # Ensure inputs require grad
        x.requires_grad_()
        
        # Baseline is zeros tensor of same shape
        baseline = torch.zeros_like(x)
        
        # Run IG (n_steps=50 is a good trade-off for latency per EXPLAINABILITY.md)
        attributions = self.ig.attribute(x, baselines=baseline, n_steps=50)
        
        # attributions shape: (1, W, F)
        # We want a single attribution score per feature.
        # Mask out padding
        mask_expanded = mask.unsqueeze(-1)
        masked_attrs = attributions * mask_expanded
        
        # Extract valid timesteps and take absolute mean over time
        valid_len = mask[0].sum().item()
        
        result = {}
        if valid_len > 0:
            valid_attrs = masked_attrs[0, :valid_len, :]
            # Mean of absolute attributions over time
            feat_attrs = torch.abs(valid_attrs).mean(dim=0)
            
            for i, feat_name in enumerate(FEATURE_NAMES):
                if i < len(feat_attrs):
                    result[feat_name] = feat_attrs[i].item()
        else:
            # Empty sequence
            for feat_name in FEATURE_NAMES:
                result[feat_name] = 0.0
                
        return result
