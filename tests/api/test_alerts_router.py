"""Tests for the M13 Alerts router."""

import pytest
from httpx import AsyncClient, ASGITransport
import asyncio

from anomaly_detection.api.main import create_app
from anomaly_detection.common.models.alerts import AlertSummary, Alert, RiskScore, Explanation
from anomaly_detection.common.models.enums import AnomalyCategory

app = create_app()

@pytest.fixture
def mock_alerts():
    return [
        AlertSummary(
            alert_id="a1",
            entity_id="test_usr_1",
            timestamp="2026-07-26T10:00:00Z",
            risk_score=95,
            risk_tier="critical",
            attack_class=AnomalyCategory.BRUTE_FORCE,
            classification_confidence=0.9,
            cold_start_flag=False,
            human_readable_explanation="Alert 1"
        ),
        AlertSummary(
            alert_id="a2",
            entity_id="test_usr_2",
            timestamp="2026-07-26T09:00:00Z",
            risk_score=80,
            risk_tier="high",
            attack_class=AnomalyCategory.LATERAL_MOVEMENT,
            classification_confidence=0.8,
            cold_start_flag=False,
            human_readable_explanation="Alert 2"
        )
    ]

@pytest.fixture
def mock_alert_detail():
    return Alert(
        alert_id="a1",
        entity_id="test_usr_1",
        event_id="e1",
        session_id="s1",
        timestamp="2026-07-26T10:00:00Z",
        detected_at="2026-07-26T10:01:00Z",
        risk=RiskScore(risk_score=95, risk_tier="critical"),
        explanation=Explanation(
            human_readable_explanation="Alert 1", 
            feature_attributions=[
                {
                    "feature_name": "f1",
                    "feature_value": 1.0,
                    "attribution_score": 0.9,
                    "direction": "toward_anomaly",
                    "source_model": "bpm",
                    "human_label": "Feature 1"
                }
            ]
        ),
        attack_class=AnomalyCategory.BRUTE_FORCE,
        classification_confidence=0.9,
        fused_score=0.95,
        bpm_score=0.9,
        sdm_score=0.9,
        cold_start_flag=False,
        raw_event_snapshot={}
    )

from unittest.mock import MagicMock

@pytest.fixture
def mock_alert_store(mock_alerts, mock_alert_detail):
    store = MagicMock()
    
    def get_alerts_mock(page=1, page_size=50, risk_tier=None, attack_class=None, entity_id=None, since=None, until=None):
        filtered = mock_alerts
        if risk_tier:
            filtered = [a for a in filtered if a.risk_tier in risk_tier]
        if attack_class:
            filtered = [a for a in filtered if a.attack_class.value in attack_class]
        if entity_id:
            filtered = [a for a in filtered if a.entity_id == entity_id]
        
        # Sort logic: risk_score DESC, timestamp DESC
        filtered = sorted(filtered, key=lambda x: (x.risk_score, x.timestamp), reverse=True)
        
        start = (page - 1) * page_size
        return filtered[start:start+page_size], len(filtered)
        
    store.get_alerts.side_effect = get_alerts_mock
    store.get_alert.return_value = mock_alert_detail
    return store


@pytest.fixture
def override_dependencies(mock_alert_store):
    from anomaly_detection.api.dependencies import get_alert_store
    app.dependency_overrides[get_alert_store] = lambda: mock_alert_store
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_alerts_pagination(override_dependencies):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Request page 1, size 1
        resp = await client.get("/api/v1/alerts?page=1&page_size=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 2
        assert len(data["alerts"]) == 1
        assert data["alerts"][0]["alert_id"] == "a1" # highest risk score

        # Request page 2, size 1
        resp = await client.get("/api/v1/alerts?page=2&page_size=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["alerts"]) == 1
        assert data["alerts"][0]["alert_id"] == "a2"

@pytest.mark.asyncio
async def test_get_alerts_filters(override_dependencies):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Filter by risk tier
        resp = await client.get("/api/v1/alerts?risk_tier=critical")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["alerts"]) == 1
        assert data["alerts"][0]["alert_id"] == "a1"
        
        # Filter by entity
        resp = await client.get("/api/v1/alerts?entity_id=test_usr_2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["alerts"]) == 1
        assert data["alerts"][0]["alert_id"] == "a2"

@pytest.mark.asyncio
async def test_get_alert_detail(override_dependencies):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/alerts/a1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alert_id"] == "a1"

@pytest.mark.asyncio
async def test_get_alert_not_found(override_dependencies, mock_alert_store):
    mock_alert_store.get_alert.return_value = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/alerts/not_found")
        assert resp.status_code == 404

@pytest.mark.asyncio
async def test_sse_stream_push():
    # We can inject an alert into the queue in the background
    summary = AlertSummary(
        alert_id="sse_1",
        timestamp="2026-07-26T10:00:00Z",
        entity_id="test_usr_001",
        risk_score=95,
        risk_tier="critical",
        attack_class=AnomalyCategory.BRUTE_FORCE,
        classification_confidence=0.9,
        cold_start_flag=False,
        human_readable_explanation="Alert 1"
    )

    # We must mock the queue so it yields one item and then stops the generator
    # otherwise ASGITransport hangs waiting for the generator to finish or buffer
    from unittest.mock import AsyncMock
    mock_queue = AsyncMock()
    mock_queue.get.side_effect = [summary, Exception("Stop iteration for test")]
    
    from anomaly_detection.api.dependencies import get_alert_stream_queue
    app.dependency_overrides[get_alert_stream_queue] = lambda: mock_queue
    
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            async with client.stream("GET", "/api/v1/stream/alerts") as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        assert "sse_1" in data
                        break # test passed if we got one event
    except Exception as e:
        if str(e) != "Stop iteration for test":
            raise
                    
    app.dependency_overrides.clear()
