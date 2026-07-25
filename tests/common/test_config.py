"""
Tests for the configuration module.

@pytest.mark.tier1
"""

import os
import pytest
from pydantic import ValidationError

from anomaly_detection.common.config import Settings, load_yaml_config


@pytest.mark.tier1
def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings correctly loads and overrides from environment variables."""
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("SECRET_KEY", "test_secret_123")
    
    settings = Settings()
    
    assert settings.api_port == 9000
    assert settings.debug is True
    assert settings.secret_key == "test_secret_123"


@pytest.mark.tier1
def test_settings_missing_required_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings raises a ValidationError if a required field (secret_key) is missing."""
    monkeypatch.delenv("SECRET_KEY", raising=False)
    
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    
    assert "secret_key" in str(exc_info.value)
    assert "Field required" in str(exc_info.value)


@pytest.mark.tier1
def test_load_yaml_config_success(tmp_path: pytest.TempPathFactory) -> None:
    """Test that a valid YAML configuration can be loaded."""
    # Create a temporary YAML file
    yaml_content = "app_name: anomaly_detection\nversion: '1.0'"
    test_yaml_path = tmp_path / "test_config.yaml"
    test_yaml_path.write_text(yaml_content)
    
    config = load_yaml_config(str(test_yaml_path))
    
    assert config["app_name"] == "anomaly_detection"
    assert config["version"] == "1.0"


@pytest.mark.tier1
def test_load_yaml_config_file_not_found() -> None:
    """Test that FileNotFoundError is raised for non-existent YAML file."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("non_existent_file.yaml")
