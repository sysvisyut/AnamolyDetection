#!/usr/bin/env python3
"""
CLI Script to train the Sequence Detection Model.

Reads generated sequences (from data_generator or Feature Engineering)
and trains the GRU Autoencoder per entity type.
"""

import argparse
import sys
from typing import List, Dict

from anomaly_detection.common.logging import setup_logger
from anomaly_detection.common.models.features import EntitySequence
from anomaly_detection.common.models.enums import EntityType
from anomaly_detection.models.sequence_detection.config import DetectionModelConfig
from anomaly_detection.models.sequence_detection.dataset import create_dataloader
from anomaly_detection.models.sequence_detection.trainer import DetectionTrainer

logger = setup_logger("train_detector")


def generate_dummy_data_for_training(num_sequences: int = 100, window_size: int = 20) -> List[EntitySequence]:
    """
    Generates dummy normal sequences for testing the training loop
    if real data is not available.
    """
    import random
    sequences = []
    
    # Need 24 dims
    F = 24
    
    for _ in range(num_sequences):
        seq_len = random.randint(5, window_size)
        raw_seq = []
        for _ in range(seq_len):
            # Normal distribution around 0.5
            vec = [max(0.0, min(1.0, random.gauss(0.5, 0.1))) for _ in range(F)]
            raw_seq.append(vec)
            
        sequences.append(EntitySequence(root=raw_seq))
        
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Train Sequence Detection Model")
    parser.add_argument("--entity-type", type=str, choices=["user", "service_account", "edge_device", "all"], 
                        default="all", help="Which model to train")
    parser.add_argument("--dummy-data", action="store_true", help="Use dummy data (for testing)")
    
    args = parser.parse_args()
    
    types_to_train = [e.value for e in EntityType] if args.entity_type == "all" else [args.entity_type]
    
    config = DetectionModelConfig()
    
    for e_type in types_to_train:
        logger.info(f"--- Training SDM for {e_type} ---")
        
        if args.dummy_data:
            logger.info("Generating dummy training data...")
            train_seqs = generate_dummy_data_for_training(500, config.window_size)
            val_seqs = generate_dummy_data_for_training(100, config.window_size)
        else:
            # Here you would load real sequences from Parquet files
            # via the Feature Engineering pipeline.
            # Since FE pipeline is already built, this script can be expanded later.
            logger.error("Real data loading not yet wired in this script. Use --dummy-data to test.")
            sys.exit(1)
            
        train_loader = create_dataloader(train_seqs, config, shuffle=True)
        val_loader = create_dataloader(val_seqs, config, shuffle=False)
        
        trainer = DetectionTrainer(config)
        model = trainer.fit(train_loader, val_loader)
        
        save_path = config.checkpoint_path(e_type)
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
