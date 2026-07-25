"""
Centralized logging configuration module.

Provides structured logging setup that all future modules will import.
Ensures consistent log formats, levels, and output destinations across
the entire application.
"""

import logging
import sys

from anomaly_detection.common.config import settings


def setup_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger with the specified name.
    
    The logger uses a standard structured format and its log level is
    controlled by the application configuration (settings.log_level).

    Args:
        name: The name of the module or component requesting the logger.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if setup_logger is called multiple times
    if not logger.handlers:
        level_name = settings.log_level.upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent log messages from propagating to the root logger to avoid duplication
        logger.propagate = False

    return logger
