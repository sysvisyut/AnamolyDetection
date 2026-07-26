"""
Training loop for the Sequence Detection Model.
Implements early stopping, masked MSE loss, and percentile calibration.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from anomaly_detection.common.logging import setup_logger
from anomaly_detection.models.sequence_detection.config import DetectionModelConfig
from anomaly_detection.models.sequence_detection.base import GRUAutoencoder

logger = setup_logger("sdm_trainer")


class DetectionTrainer:
    """
    Trains the GRU Autoencoder on normal sequences.
    Computes and saves score calibration percentiles at the end of training.
    """
    
    def __init__(self, config: DetectionModelConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = GRUAutoencoder(config).to(self.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            min_lr=config.lr_min
        )
        
    def _masked_mse_loss(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes Mean Squared Error only on valid (unpadded) sequence steps.
        
        recon: (B, W, F)
        target: (B, W, F)
        mask: (B, W) boolean
        """
        # compute squared error everywhere
        # Shape: (B, W, F)
        squared_err = (recon - target) ** 2
        
        # Mask out padding: expand mask to (B, W, 1) and broadcast
        # Shape of mask_expanded: (B, W, 1)
        mask_expanded = mask.unsqueeze(-1)
        
        # apply mask
        masked_err = squared_err * mask_expanded
        
        # mean over valid elements
        # valid element count = mask.sum() * F
        valid_elements = mask.sum() * self.config.feature_dim
        
        if valid_elements > 0:
            return masked_err.sum() / valid_elements
        else:
            return torch.tensor(0.0, device=recon.device, requires_grad=True)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        batches = 0
        
        for seqs, masks, _ in dataloader:
            seqs = seqs.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            recon = self.model(seqs, masks)
            loss = self._masked_mse_loss(recon, seqs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        return total_loss / max(1, batches)
        
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Runs evaluation without parameter updates."""
        self.model.eval()
        total_loss = 0.0
        batches = 0
        
        for seqs, masks, _ in dataloader:
            seqs = seqs.to(self.device)
            masks = masks.to(self.device)
            
            recon = self.model(seqs, masks)
            loss = self._masked_mse_loss(recon, seqs, masks)
            
            total_loss += loss.item()
            batches += 1
            
        return total_loss / max(1, batches)
        
    @torch.no_grad()
    def calibrate(self, dataloader: DataLoader) -> None:
        """
        Runs full dataset through trained model to establish error percentiles.
        Updates config with calibration_err_min and calibration_err_max.
        """
        self.model.eval()
        all_errors = []
        
        for seqs, masks, _ in dataloader:
            seqs = seqs.to(self.device)
            masks = masks.to(self.device)
            
            recon = self.model(seqs, masks)
            
            # error per timestep per feature: (B, W, F)
            err = (recon - seqs) ** 2
            
            # mask invalid: (B, W, 1)
            mask_expanded = masks.unsqueeze(-1)
            masked_err = err * mask_expanded
            
            # For each sequence, calculate the combined error score
            # per ML_PIPELINE.md §3.6: combined = mean(valid) + 0.3 * max(valid)
            for i in range(seqs.size(0)):
                valid_len = masks[i].sum().item()
                if valid_len > 0:
                    # extract valid timesteps for this sequence: (valid_len, F)
                    valid_errs = masked_err[i, :valid_len, :]
                    
                    # mean over time and features
                    mean_err = valid_errs.mean().item()
                    
                    # max over time and features
                    max_err = valid_errs.max().item()
                    
                    combined = (
                        self.config.raw_score_weight_mean * mean_err + 
                        self.config.raw_score_weight_max * max_err
                    )
                    all_errors.append(combined)
                    
        if not all_errors:
            logger.warning("No valid sequences during calibration. Using default bounds.")
            self.model.config.calibration_err_min = 0.0
            self.model.config.calibration_err_max = 1.0
            return
            
        # compute percentiles
        arr = np.array(all_errors)
        self.model.config.calibration_err_min = float(np.percentile(arr, 1))
        self.model.config.calibration_err_max = float(np.percentile(arr, 99))
        
        logger.info(
            f"Calibration complete: min(1st)={self.model.config.calibration_err_min:.5f}, "
            f"max(99th)={self.model.config.calibration_err_max:.5f}"
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> GRUAutoencoder:
        """
        Trains the model with early stopping.
        Calibrates on train_loader after best weights are restored.
        """
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_state = None
        
        logger.info(f"Starting SDM training. Max epochs: {self.config.max_epochs}")
        
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.debug(
                f"Epoch {epoch:03d} | Train MSE: {train_loss:.5f} | "
                f"Val MSE: {val_loss:.5f} | LR: {current_lr:.2e}"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
                
        logger.info(f"Training complete. Best Val MSE: {best_val_loss:.5f}")
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        # Calibrate score thresholds
        logger.info("Calibrating reconstruction score thresholds on training set...")
        self.calibrate(train_loader)
        
        return self.model
