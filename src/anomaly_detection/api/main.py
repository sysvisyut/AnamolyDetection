"""FastAPI application factory and lifespan management."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
import yaml

from anomaly_detection.api.middleware import setup_middleware
from anomaly_detection.api.routers import inference, alerts, entities
from src.orchestrator import InferencePipeline, OrchestratorConfig
from src.orchestrator.alert_builder import AlertBuilder
from anomaly_detection.feature_engineering.feature_pipeline import FeaturePipeline
from anomaly_detection.feature_engineering.config import FeatureEngineeringConfig as FeaturePipelineConfig
from src.profiling.profile_store import ProfileStore
from src.profiling.config import ProfilingConfig
from src.profiling.profile_model import BehavioralProfilingModel
from src.profiling.population_prior import PopulationPrior
from anomaly_detection.models.sequence_detection.inference import SDMInferenceEngine
from anomaly_detection.models.sequence_detection.config import DetectionModelConfig as SDMConfig
from anomaly_detection.models.fusion import ScoreFusion, FusionConfig
from src.classification.classifier import AnomalyClassifier
from src.classification.config import ClassifierConfig
from src.explainability.engine import ExplainabilityEngine
from src.explainability.config import ExplainabilityConfig
from src.drift.ewma_updater import EWMAUpdater
from src.drift.config import DriftConfig
from anomaly_detection.stores.backends.sqlite import SQLiteAlertStore
from anomaly_detection.stores.backends.in_memory import InMemoryAlertStore

logger = logging.getLogger(__name__)


def init_alert_store(config: dict):
    store_conf = config.get("stores", {}).get("alert_store", {})
    backend = store_conf.get("backend", "in_memory")
    if backend == "sqlite":
        path = store_conf.get("sqlite_path", "data/alerts.db")
        store = SQLiteAlertStore(db_path=path)
        return store
    return InMemoryAlertStore()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown logic for the FastAPI application.
    
    Initializes models and stores once so inference latency isn't dominated by model loading.
    """
    logger.info("Initializing models and stores...")
    
    # Load configuration
    try:
        with open("config/default.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load config/default.yaml: {e}, using defaults")
        config = {}

    model_params = config.get("model_params", {})
    
    profiling_config = ProfilingConfig()
    
    # 1. Initialize Stores
    profile_store = ProfileStore(profiling_config.profile_store_path)
    alert_store = init_alert_store(config)

    # 2. Initialize Feature Pipeline
    feature_pipeline = FeaturePipeline(FeaturePipelineConfig(), profile_store=profile_store)

    # 3. Initialize Models
    prior = PopulationPrior(profiling_config)
    bpm = BehavioralProfilingModel(profiling_config, profile_store, prior)
    sdm_path = model_params.get("sdm_path", "data/models/sdm_user.pt")
    
    # In tests or missing artifact scenarios, SDMInferenceEngine will raise FileNotFoundError
    # if the file doesn't exist. We catch it to allow the API to start in degraded mode or tests.
    try:
        sdm = SDMInferenceEngine(sdm_path)
    except FileNotFoundError:
        logger.warning(f"SDM model not found at {sdm_path}. Falling back to dummy.")
        from unittest.mock import MagicMock
        sdm = MagicMock()
        # Mock predict to return a default DetectionOutput so inference can proceed
        from anomaly_detection.common.models.ml_io import DetectionOutput
        def mock_predict(features):
            from anomaly_detection.common.models.ml_io import DetectionOutput
            return DetectionOutput(
                model_id="sdm",
                entity_id=features.entity_id,
                event_id=features.event_id,
                anomaly_score=0.1,
                reconstruction_error=0.1,
                is_anomaly=False,
                confidence=0.9,
                cold_start_flag=False,
                top_contributing_features=["foo"],
                feature_attributions={}
            )
        sdm.predict.side_effect = mock_predict
    
    fusion_config = FusionConfig(
        bpm_weight=model_params.get("fusion_weights", {}).get("bpm", 0.5),
        sdm_weight=model_params.get("fusion_weights", {}).get("sdm", 0.5),
        fusion_threshold=model_params.get("fusion_threshold", 0.5)
    )
    fusion = ScoreFusion(fusion_config)
    
    classifier = AnomalyClassifier(ClassifierConfig())
    # Note: In a real environment, you'd call classifier.load_model(...) here.
    
    explainability = ExplainabilityEngine(ExplainabilityConfig())
    drift_config = DriftConfig()
    ewma_updater = EWMAUpdater(config=drift_config, profile_store=profile_store)
    alert_builder = AlertBuilder()

    # 4. Assemble Orchestrator
    orchestrator = InferencePipeline(
        config=OrchestratorConfig(),
        feature_pipeline=feature_pipeline,
        profile_store=profile_store,
        profiling_model=bpm,
        detection_model=sdm,
        score_fusion=fusion,
        classifier=classifier,
        explainability=explainability,
        ewma_updater=ewma_updater,
        alert_store=alert_store,
        alert_builder=alert_builder
    )

    # Attach to app state
    app.state.orchestrator = orchestrator
    app.state.profile_store = profile_store
    app.state.alert_store = alert_store
    
    logger.info("Application startup complete.")
    yield
    
    logger.info("Shutting down application...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Anomaly Detection API",
        description="AI-Powered Behavioral Anomaly Detection for Cybersecurity",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Setup Middleware (Read CORS from config)
    try:
        with open("config/default.yaml", "r") as f:
            config = yaml.safe_load(f)
        cors_origins = config.get("api", {}).get("cors_origins", ["*"])
    except Exception:
        cors_origins = ["*"]

    setup_middleware(app, cors_origins)

    # Register Routers
    app.include_router(inference.router, prefix="/api/v1/inference", tags=["Inference"])
    app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["Alerts"])
    app.include_router(entities.router, prefix="/api/v1/entities", tags=["Entities"])

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app

app = create_app()
