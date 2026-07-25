"""
Tests for the centralized logging configuration.

@pytest.mark.tier1
"""

import logging
import pytest

from anomaly_detection.common.logging import setup_logger


@pytest.mark.tier1
def test_setup_logger_initialization() -> None:
    """Test that setup_logger initializes and returns a valid logger."""
    logger_name = "test_logger_init"
    logger = setup_logger(logger_name)
    
    assert logger.name == logger_name
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 1
    assert logger.propagate is False


@pytest.mark.tier1
def test_setup_logger_idempotent() -> None:
    """Test that setup_logger does not add multiple handlers if called multiple times."""
    logger_name = "test_logger_idempotent"
    logger1 = setup_logger(logger_name)
    logger2 = setup_logger(logger_name)
    
    assert logger1 is logger2
    assert len(logger2.handlers) == 1


@pytest.mark.tier1
def test_logger_output(capsys: pytest.CaptureFixture) -> None:
    """Test that the logger produces structured output to stdout."""
    logger_name = "test_logger_output"
    logger = setup_logger(logger_name)
    
    # We force the level to INFO just to be sure it logs the info message
    logger.setLevel(logging.INFO)
    logger.handlers[0].setLevel(logging.INFO)
    
    test_message = "This is a test log message."
    logger.info(test_message)
    
    captured = capsys.readouterr()
    
    assert test_message in captured.out
    assert "INFO" in captured.out
    assert logger_name in captured.out
