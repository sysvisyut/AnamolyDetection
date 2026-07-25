"""
Custom exception hierarchy for the anomaly detection project.

This module provides base exceptions and specific error types
to allow structured error handling across the application without
relying on generic Python exceptions.
"""

class AnomalyDetectionError(Exception):
    """Base exception for all anomaly detection errors."""
    pass

class ConfigurationError(AnomalyDetectionError):
    """Raised when application configuration is invalid or missing."""
    pass

class DataValidationError(AnomalyDetectionError):
    """Raised when boundary data fails validation."""
    pass

class ModelNotTrainedError(AnomalyDetectionError):
    """Raised when inference is attempted on an untrained model."""
    pass
