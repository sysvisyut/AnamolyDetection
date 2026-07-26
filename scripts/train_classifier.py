#!/usr/bin/env python3
"""
CLI script to train the Anomaly Classifier (M08).
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
import numpy as np

from src.classification.config import ClassifierConfig
from src.classification.trainer import ClassifierTrainer
from src.classification.dataset import LABEL_TO_INT, INT_TO_LABEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Train M08 Anomaly Classifier")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data (parquet)")
    parser.add_argument("--val-data", type=str, help="Path to validation data (parquet)")
    parser.add_argument("--model-out", type=str, default="data/models/classifier_model.txt", help="Path to save model")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE")
    args = parser.parse_args()

    config = ClassifierConfig(model_path=args.model_out)
    trainer = ClassifierTrainer(config)
    
    logging.info(f"Loading training data from {args.train_data}...")
    train_df = pd.read_parquet(args.train_data)
    
    val_df = None
    if args.val_data:
        logging.info(f"Loading validation data from {args.val_data}...")
        val_df = pd.read_parquet(args.val_data)
        
    logging.info("Starting training...")
    classifier, evals_result = trainer.train(
        train_df, 
        val_df=val_df, 
        apply_smote=not args.no_smote
    )
    
    # Save the model
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(out_path))
    logging.info(f"Model saved to {out_path}")

    # Output confusion matrix on train (or val) for verification
    # Normally we would use a separate evaluation script, but this is a quick check
    from sklearn.metrics import confusion_matrix, classification_report
    from src.classification.dataset import prepare_training_data
    
    logging.info("Generating evaluation metrics on training set...")
    X_train, y_train, _, _ = prepare_training_data(train_df, config, apply_smote=False)
    preds = classifier.model.predict(X_train)
    y_pred = np.argmax(preds, axis=1)
    
    labels = list(range(len(INT_TO_LABEL)))
    target_names = [INT_TO_LABEL[i] for i in labels]
    
    cm = confusion_matrix(y_train, y_pred, labels=labels)
    logging.info("\nConfusion Matrix:\n%s", cm)
    logging.info("\nClassification Report:\n%s", classification_report(y_train, y_pred, labels=labels, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()
