"""
Global pytest configuration and shared fixtures for the anomaly detection project.
"""

import os
import pytest

# Set environment variables required for Pydantic Settings instantiation during import
os.environ["SECRET_KEY"] = "test_secret_for_collection"
os.environ["ENVIRONMENT"] = "testing"
os.environ["DB_PATH"] = ":memory:"


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Automatically set environment variables for the test environment.
    This fixture ensures that tests don't inadvertently connect to
    production stores or use production secrets.
    """
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("DB_PATH", ":memory:")
    monkeypatch.setenv("SECRET_KEY", "test_secret_key")
    # Add other mock environment variables needed for testing here
