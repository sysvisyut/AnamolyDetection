"""Tests for the M13 Entities router."""

import pytest
from httpx import AsyncClient, ASGITransport

from anomaly_detection.api.main import create_app
from anomaly_detection.common.models.entities import EntityProfile, EntityHistoryEntry, DriftMetrics
from anomaly_detection.common.models.enums import EntityType, AnomalyCategory

app = create_app()

@pytest.fixture
def mock_profile():
    return EntityProfile(
        entity_id="test_usr_1",
        entity_type=EntityType.USER,
        baseline_vector=[0.0] * 24,
        baseline_std=[0.1] * 24,
        sequence_history=[],
        most_frequent_country="US",
        known_mac_addresses=[],
        known_os_profiles=[],
        known_protocols=[],
        resource_access_counts={},
        command_frequency={},
        event_count=100,
        cold_start_flag=False,
        drift_metrics=DriftMetrics(
            feature_means_history=[],
            last_drift_check="2026-07-26T10:00:00Z",
            drift_severity="low",
            drift_detected_at=None
        ),
        last_updated="2026-07-26T10:00:00Z",
        profile_version=2
    )

@pytest.fixture
def mock_history():
    return [
        EntityHistoryEntry(
            event_id="e1",
            timestamp="2026-07-26T10:00:00Z",
            resource_accessed="file/test",
            auth_outcome="success",
            risk_score=None,
            attack_class=AnomalyCategory.NORMAL,
            has_alert=False
        )
    ]

from unittest.mock import MagicMock

@pytest.fixture
def mock_profile_store(mock_profile):
    store = MagicMock()
    store.get_profile.return_value = mock_profile
    return store

@pytest.fixture
def mock_alert_store(mock_history):
    store = MagicMock()
    store.get_entity_history.return_value = mock_history
    return store

@pytest.fixture
def override_dependencies(mock_profile_store, mock_alert_store):
    from anomaly_detection.api.dependencies import get_profile_store, get_alert_store
    app.dependency_overrides[get_profile_store] = lambda: mock_profile_store
    app.dependency_overrides[get_alert_store] = lambda: mock_alert_store
    yield
    app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_get_entity_status(override_dependencies):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/entities/test_usr_1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_id"] == "test_usr_1"
        assert data["is_cold_start"] is False
        assert data["drift_severity"] == "low"
        assert data["profile_version"] == 2

@pytest.mark.asyncio
async def test_get_entity_status_not_found(override_dependencies, mock_profile_store):
    mock_profile_store.get_profile.return_value = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/entities/not_found/status")
        assert resp.status_code == 404

@pytest.mark.asyncio
async def test_get_entity_history(override_dependencies, mock_alert_store):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/entities/test_usr_1/history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["event_id"] == "e1"

@pytest.mark.asyncio
async def test_get_entity_history_not_implemented(override_dependencies, mock_alert_store):
    del mock_alert_store.get_entity_history
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/entities/test_usr_1/history")
        assert resp.status_code == 501
