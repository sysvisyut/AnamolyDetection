"""
GRU Autoencoder for Sequence Detection Model.

Implements the PyTorch sequence anomaly detector defined in ML_PIPELINE.md §3.
Uses a GRU encoder to compress a sequence of 20 feature vectors into a
64-dim bottleneck, and a GRU decoder to reconstruct the original sequence.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

from anomaly_detection.models.sequence_detection.config import DetectionModelConfig


class GRUEncoder(nn.Module):
    """
    Encodes the input sequence into a fixed-size bottleneck vector.
    """
    def __init__(self, config: DetectionModelConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=config.feature_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_encoder_layers,
            batch_first=True,
            dropout=config.dropout if config.num_encoder_layers > 1 else 0.0
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, window_size, feature_dim)
        mask: (batch_size, window_size) boolean mask where True = valid, False = padding
        
        Returns bottleneck vector h_enc of shape (batch_size, hidden_size).
        Extracts the hidden state at the last valid timestep for each sequence.
        """
        # GRU outputs: (batch_size, seq_len, hidden_size)
        out, _ = self.gru(x)
        
        # Determine the last valid index per sequence
        # mask is True for valid, False for padding.
        # Summing True values along dim 1 gives the count.
        # Index is count - 1. Shape: (batch_size,)
        seq_lengths = mask.sum(dim=1).long()
        last_indices = torch.clamp(seq_lengths - 1, min=0)
        
        # Gather the hidden state at the last valid timestep
        batch_size = x.size(0)
        batch_indices = torch.arange(batch_size, device=x.device)
        
        h_enc = out[batch_indices, last_indices, :]
        return h_enc


class GRUDecoder(nn.Module):
    """
    Decodes the bottleneck vector back into a reconstructed sequence.
    """
    def __init__(self, config: DetectionModelConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=config.hidden_size,  # input is repeated bottleneck
            hidden_size=config.hidden_size,
            num_layers=config.num_decoder_layers,
            batch_first=True,
            dropout=config.dropout if config.num_decoder_layers > 1 else 0.0
        )
        # Project hidden state back to original feature dimension
        self.output_layer = nn.Linear(config.hidden_size, config.feature_dim)
        
    def forward(self, h_enc: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        h_enc: (batch_size, hidden_size)
        seq_len: integer representing the window size to reconstruct (W=20)
        
        Returns reconstructed sequence (batch_size, window_size, feature_dim).
        """
        batch_size = h_enc.size(0)
        
        # Repeat bottleneck vector W times
        # Shape: (batch_size, seq_len, hidden_size)
        repeated_h = h_enc.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        out, _ = self.gru(repeated_h)
        
        # Project back to feature space
        # Shape: (batch_size, seq_len, feature_dim)
        reconstruction = self.output_layer(out)
        
        return reconstruction


class GRUAutoencoder(nn.Module):
    """
    Full Sequence Detection Model (SDM).
    
    Combines GRUEncoder and GRUDecoder.
    """
    def __init__(self, config: DetectionModelConfig):
        super().__init__()
        self.config = config
        self.encoder = GRUEncoder(config)
        self.decoder = GRUDecoder(config)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training and inference.
        
        x: (batch_size, window_size, feature_dim)
        mask: (batch_size, window_size) boolean mask
        
        Returns:
            reconstruction: (batch_size, window_size, feature_dim)
        """
        seq_len = x.size(1)
        
        # 1. Encode to bottleneck
        h_enc = self.encoder(x, mask)
        
        # 2. Decode to reconstruction
        reconstruction = self.decoder(h_enc, seq_len)
        
        return reconstruction
    
    def save(self, filepath: str) -> None:
        """Saves model state and config to disk."""
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict()
        }
        torch.save(state, filepath)
        
    @classmethod
    def load(cls, filepath: str) -> "GRUAutoencoder":
        """Loads model state and config from disk."""
        state = torch.load(filepath, map_location="cpu", weights_only=True)
        config = DetectionModelConfig.from_dict(state["config"])
        
        model = cls(config)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        return model
