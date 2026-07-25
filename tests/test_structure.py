"""
Tests to verify the project directory structure matches PROJECT_STRUCTURE.md.

@pytest.mark.tier1
"""

import os
from pathlib import Path
import pytest


@pytest.mark.tier1
def test_directory_structure() -> None:
    """
    Verify that the essential directory structure exists as defined in
    PROJECT_STRUCTURE.md.
    """
    root_dir = Path(__file__).parent.parent
    
    expected_directories = [
        "config",
        "data/raw",
        "data/labeled",
        "data/processed",
        "data/profiles",
        "src/anomaly_detection/common",
        "src/anomaly_detection/data_generator",
        "src/anomaly_detection/streaming",
        "src/anomaly_detection/feature_engineering",
        "src/anomaly_detection/stores/backends",
        "src/anomaly_detection/models/behavioral_profiling/artifacts",
        "src/anomaly_detection/models/sequence_detection/artifacts",
        "src/anomaly_detection/models/fusion",
        "src/anomaly_detection/classifier/artifacts",
        "src/anomaly_detection/explainability",
        "src/anomaly_detection/api/routers",
        "src/anomaly_detection/cold_start",
        "src/anomaly_detection/drift",
        "src/anomaly_detection/evaluation",
        "src/dashboard/scripts",
        "src/dashboard/styles",
        "src/dashboard/assets",
        "scripts",
        "notebooks",
        "docs/report",
        "docs/presentation",
        "tests",
    ]
    
    for directory in expected_directories:
        dir_path = root_dir / directory
        assert dir_path.is_dir(), f"Expected directory not found: {directory}"

    # Also verify some key files created by M01
    expected_files = [
        "pyproject.toml",
        ".env.example",
        "config/default.yaml",
        "src/anomaly_detection/common/config.py",
        "src/anomaly_detection/common/logging.py",
        "src/anomaly_detection/common/exceptions.py",
    ]

    for file_path in expected_files:
        full_path = root_dir / file_path
        assert full_path.is_file(), f"Expected file not found: {file_path}"
