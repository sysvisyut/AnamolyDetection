"""
Dataset and DataLoader utilities for the Sequence Detection Model.
"""

from typing import List, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader

from anomaly_detection.common.models.features import EntitySequence
from anomaly_detection.models.sequence_detection.config import DetectionModelConfig


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequences.
    
    Holds input tensors and their corresponding masks.
    Labels are not required for unsupervised autoencoder training,
    but can optionally be stored for evaluation purposes.
    """
    
    def __init__(self, sequences: torch.Tensor, masks: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Args:
            sequences: (N, W, F) float tensor of sequences
            masks: (N, W) boolean tensor of valid indices
            labels: (N,) optional tensor of labels (0=normal, 1=anomaly) for evaluation
        """
        super().__init__()
        self.sequences = sequences
        self.masks = masks
        self.labels = labels
        
        assert self.sequences.size(0) == self.masks.size(0), "Sequence and mask counts must match"
        if self.labels is not None:
            assert self.sequences.size(0) == self.labels.size(0), "Sequence and label counts must match"

    def __len__(self) -> int:
        return self.sequences.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.labels is not None:
            return self.sequences[idx], self.masks[idx], self.labels[idx]
        return self.sequences[idx], self.masks[idx], torch.tensor(-1, dtype=torch.long)


def build_tensor_from_sequence(seq: EntitySequence, config: DetectionModelConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a single EntitySequence into padded tensor and mask.
    
    Args:
        seq: EntitySequence containing list of feature vectors.
        config: DetectionModelConfig containing window_size and feature_dim.
        
    Returns:
        padded_seq: (W, F) float tensor
        mask: (W,) boolean tensor where True = valid, False = padding
    """
    W = config.window_size
    F = config.feature_dim
    
    raw_list = seq.root
    seq_len = len(raw_list)
    
    # Ensure we don't exceed window size (should not happen per FE, but defend)
    if seq_len > W:
        raw_list = raw_list[-W:]
        seq_len = W
        
    padded_seq = torch.zeros((W, F), dtype=torch.float32)
    mask = torch.zeros(W, dtype=torch.bool)
    
    if seq_len > 0:
        # Pydantic validates inner types, but ensure we have float tensors
        valid_tensor = torch.tensor(raw_list, dtype=torch.float32)
        padded_seq[:seq_len, :] = valid_tensor
        mask[:seq_len] = True
        
    return padded_seq, mask


def create_dataloader(
    sequences: List[EntitySequence],
    config: DetectionModelConfig,
    labels: Optional[List[int]] = None,
    shuffle: bool = False
) -> DataLoader:
    """
    Factory to create a DataLoader from a list of EntitySequence objects.
    
    Args:
        sequences: List of EntitySequence objects.
        config: Model configuration.
        labels: Optional list of integer labels (0 or 1).
        shuffle: Whether to shuffle the DataLoader.
        
    Returns:
        Configured DataLoader yielding (batch_seq, batch_mask, batch_labels).
    """
    N = len(sequences)
    W = config.window_size
    F = config.feature_dim
    
    all_seqs = torch.zeros((N, W, F), dtype=torch.float32)
    all_masks = torch.zeros((N, W), dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        s_tensor, m_tensor = build_tensor_from_sequence(seq, config)
        all_seqs[i] = s_tensor
        all_masks[i] = m_tensor
        
    t_labels = None
    if labels is not None:
        t_labels = torch.tensor(labels, dtype=torch.long)
        
    dataset = SequenceDataset(all_seqs, all_masks, t_labels)
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0  # For stability across OS
    )
