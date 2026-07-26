"""FastAPI dependency injection wiring."""

from fastapi import Request
from anomaly_detection.stores.alert_store import AbstractAlertStore
from src.profiling.profile_store import ProfileStore
from src.orchestrator.pipeline import InferencePipeline


def get_orchestrator(request: Request) -> InferencePipeline:
    """Retrieve the initialized InferencePipeline from app state."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if not orchestrator:
        raise RuntimeError("Orchestrator is not initialized in app state.")
    return orchestrator


def get_alert_store(request: Request) -> AbstractAlertStore:
    """Retrieve the initialized AlertStore from app state."""
    alert_store = getattr(request.app.state, "alert_store", None)
    if not alert_store:
        raise RuntimeError("AlertStore is not initialized in app state.")
    return alert_store


def get_profile_store(request: Request) -> ProfileStore:
    """Retrieve the initialized ProfileStore from app state."""
    profile_store = getattr(request.app.state, "profile_store", None)
    if not profile_store:
        raise RuntimeError("ProfileStore is not initialized in app state.")
    return profile_store
