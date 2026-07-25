"""
Configuration management module.

Loads environment variables and YAML configurations, providing a centralized,
validated Settings object used throughout the application. Uses pydantic-settings
for declarative validation.
"""

from typing import Any, Dict

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    Provides validation and default values for all required configuration.
    """

    # --- Core API Settings ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # --- Storage Settings ---
    db_path: str = "data/profiles/store.db"
    redis_url: str = "redis://localhost:6379/0"

    # --- ML Pipeline & Paths ---
    model_artifacts_dir: str = "models/behavioral_profiling/artifacts/"
    data_dir: str = "data/"
    min_profile_events: int = 10

    # --- Streaming Settings ---
    streaming_compression_factor: float = 10.0
    streaming_queue_size: int = 1000

    # --- Secrets ---
    secret_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_yaml_config(filepath: str = "config/default.yaml") -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        filepath: Path to the YAML configuration file.

    Returns:
        A dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


# Global settings instance
settings = Settings()
