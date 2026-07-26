#!/usr/bin/env python3
"""
CLI entry point for running the M15 Model Evaluation.
"""

import argparse
import sys
import logging
import os
from pathlib import Path

from anomaly_detection.common.logging import setup_logger
from anomaly_detection.evaluation.evaluator import Evaluator
from anomaly_detection.evaluation.report_generator import generate_report

logger = setup_logger("run_evaluation")


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline evaluation.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID of the dataset to evaluate.")
    parser.add_argument("--raw-data-dir", type=str, default="data/raw", help="Directory containing raw data.")
    parser.add_argument("--labeled-data-dir", type=str, default="data/labeled", help="Directory containing labeled data.")
    parser.add_argument("--output-dir", type=str, default="docs", help="Output directory for the markdown report.")
    
    args = parser.parse_args()
    
    raw_data_path = Path(args.raw_data_dir) / f"synthetic_logs_{args.run_id}.parquet"
    labels_path = Path(args.labeled_data_dir) / f"labels_{args.run_id}.parquet"
    
    if not raw_data_path.exists():
        logger.error(f"Raw data file not found: {raw_data_path}")
        sys.exit(1)
        
    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        sys.exit(1)
        
    logger.info(f"Starting evaluation for run ID: {args.run_id}")
    logger.info(f"Using raw data: {raw_data_path}")
    logger.info(f"Using labels: {labels_path}")
    
    evaluator = Evaluator(
        run_id=args.run_id,
        raw_data_path=str(raw_data_path),
        labels_path=str(labels_path)
    )
    
    try:
        metrics = evaluator.evaluate()
        logger.info("Evaluation metrics computed successfully.")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        sys.exit(1)
        
    report_path = generate_report(metrics, args.output_dir, args.run_id)
    logger.info(f"Evaluation complete. Report generated at: {report_path}")

if __name__ == "__main__":
    main()
