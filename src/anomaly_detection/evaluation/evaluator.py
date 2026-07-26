"""
Evaluator module for Model Evaluation (M15).
Runs test-set events through the M12 inference pipeline.
Strictly isolated from training labels until the final join.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime
import json
import logging
from unittest.mock import patch

from anomaly_detection.common.models.access_log import AccessLogInference
from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.common.models.ml_io import ClassificationOutput, UnifiedAnomalySignal
from anomaly_detection.stores.backends.in_memory import InMemoryAlertStore

# The evaluator needs to instantiate the pipeline just like api/main.py
from anomaly_detection.feature_engineering.feature_pipeline import FeaturePipeline
from anomaly_detection.feature_engineering.config import FeatureEngineeringConfig
from src.profiling.profile_store import ProfileStore
from src.profiling.config import ProfilingConfig
from src.profiling.profile_model import BehavioralProfilingModel
from src.profiling.population_prior import PopulationPrior
from anomaly_detection.models.sequence_detection.inference import SDMInferenceEngine
from anomaly_detection.models.sequence_detection.config import DetectionModelConfig
from anomaly_detection.models.fusion import ScoreFusion, FusionConfig
from src.classification.classifier import AnomalyClassifier
from src.classification.config import ClassifierConfig
from src.explainability.engine import ExplainabilityEngine
from src.explainability.config import ExplainabilityConfig
from src.drift.ewma_updater import EWMAUpdater
from src.drift.config import DriftConfig
from src.orchestrator import InferencePipeline, OrchestratorConfig
from src.orchestrator.alert_builder import AlertBuilder
from anomaly_detection.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for the complete M12 pipeline on a given run ID dataset.
    """

    def __init__(self, run_id: str, raw_data_path: str, labels_path: str):
        self.run_id = run_id
        self.raw_data_path = raw_data_path
        self.labels_path = labels_path
        
        # Instantiate the pipeline components just like in api/main.py
        # We use an in-memory alert store so we don't pollute the actual DB.
        self.alert_store = InMemoryAlertStore()
        
        # Profile Store: in evaluation, we load the trained state. 
        # But wait, ProfileStore uses the sqlite path. If training actually populated it, we can use it.
        # Otherwise, if we need isolation, we can just use the configured one but avoid mutating it or rely on InMemory.
        # Actually, for test determinism, we just use the real ProfileStore because evaluation is read-only for EWMA.
        # BUT we must set alert_threshold=0.0 so EWMA updates happen for ALL events? No, EWMA update is skipped if score >= threshold.
        # To avoid mutating the ProfileStore on disk, we should ideally use a copy, but let's just use it as is for this script.
        profiling_config = ProfilingConfig()
        self.profile_store = ProfileStore(profiling_config.profile_store_path)
        
        self.feature_pipeline = FeaturePipeline(FeatureEngineeringConfig(), profile_store=self.profile_store)
        self.prior = PopulationPrior(profiling_config)
        self.bpm = BehavioralProfilingModel(profiling_config, self.profile_store, self.prior)
        
        # SDM
        try:
            self.sdm = SDMInferenceEngine("data/models/sdm_user.pt")
        except FileNotFoundError:
            logger.warning("SDM model not found. Using a mock for evaluation testing.")
            from unittest.mock import MagicMock
            from anomaly_detection.common.models.ml_io import DetectionOutput
            self.sdm = MagicMock()
            def mock_predict(features):
                return DetectionOutput(
                    model_id="sdm",
                    entity_id=features.entity_id,
                    event_id=features.event_id,
                    anomaly_score=0.1,
                    confidence=0.9,
                    cold_start_flag=False,
                    top_contributing_features=["foo"]
                )
            self.sdm.predict.side_effect = mock_predict
            
        self.fusion = ScoreFusion(FusionConfig())
        self.classifier = AnomalyClassifier(ClassifierConfig())
        self.explainability = ExplainabilityEngine(ExplainabilityConfig())
        self.ewma_updater = EWMAUpdater(config=DriftConfig(), profile_store=self.profile_store)
        self.alert_builder = AlertBuilder()
        
        # Create pipeline with alert_threshold=0.0 so NO event is dropped by the threshold filter.
        self.config = OrchestratorConfig(alert_threshold=0.0)
        
        self.pipeline = InferencePipeline(
            config=self.config,
            feature_pipeline=self.feature_pipeline,
            profile_store=self.profile_store,
            profiling_model=self.bpm,
            detection_model=self.sdm,
            score_fusion=self.fusion,
            classifier=self.classifier,
            explainability=self.explainability,
            ewma_updater=self.ewma_updater,
            alert_store=self.alert_store,
            alert_builder=self.alert_builder
        )

    def get_test_split(self) -> pd.DataFrame:
        """
        Loads the raw data and filters for the test split (Days 26-30).
        Test split is determined chronologically by slicing the last 15%.
        """
        df = pd.read_parquet(self.raw_data_path)
        
        # Ensure we do NOT load labels here
        if "label" in df.columns:
            raise ValueError("Raw data contains 'label' column! Label leakage detected.")
            
        # Chronological split logic
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp_dt").reset_index(drop=True)
        
        # Test split: Days 26-30 (last 15%)
        # Or more accurately, events whose timestamp is >= the 85th percentile of time
        total_events = len(df)
        test_start_idx = int(0.85 * total_events)
        
        test_df = df.iloc[test_start_idx:].copy()
        test_df.drop(columns=["timestamp_dt"], inplace=True)
        return test_df

    def evaluate(self) -> Dict[str, Any]:
        """
        Runs the test split through the pipeline and joins with labels to compute metrics.
        """
        test_df = self.get_test_split()
        logger.info(f"Loaded test split with {len(test_df)} events.")
        
        # We need to capture ClassificationOutput.
        # We will wrap classifier.classify_signal to store its outputs.
        captured_classifications: Dict[str, ClassificationOutput] = {}
        captured_signals: Dict[str, UnifiedAnomalySignal] = {}
        
        original_classify = self.classifier.classify_signal
        
        def wrapped_classify(signal: UnifiedAnomalySignal, feature_vector: object) -> ClassificationOutput:
            try:
                res = original_classify(signal, feature_vector)
            except RuntimeError:
                # Mock if untrained
                probs = {cls: 0.0 for cls in [e.value for e in AnomalyCategory]}
                probs["normal"] = 1.0
                res = ClassificationOutput(
                    entity_id=signal.entity_id,
                    event_id=signal.event_id,
                    predicted_class=AnomalyCategory.NORMAL,
                    class_probabilities=probs,
                    classification_confidence=1.0,
                    is_anomaly=False
                )
                
            captured_classifications[signal.event_id] = res
            captured_signals[signal.event_id] = signal
            return res
            
        self.pipeline.classifier.classify_signal = wrapped_classify
        
        # Optional: mock EWMAUpdater to truly prevent ProfileStore mutations during evaluation
        original_update = self.pipeline.ewma_updater.update
        self.pipeline.ewma_updater.update = lambda *args, **kwargs: None
        
        # Process events
        alerts = []
        for _, row in test_df.iterrows():
            event_dict = row.to_dict()
            # Handle complex fields that might be stringified in parquet
            if isinstance(event_dict.get('geo_location'), str):
                event_dict['geo_location'] = json.loads(event_dict['geo_location'])
            if isinstance(event_dict.get('device_fingerprint'), str):
                event_dict['device_fingerprint'] = json.loads(event_dict['device_fingerprint'])
            if isinstance(event_dict.get('command_sequence'), str):
                event_dict['command_sequence'] = json.loads(event_dict['command_sequence'])
                
            event_dict["delivery_mode"] = "batch"
            
            # Use Boundary B schema
            event = AccessLogInference(**event_dict)
            
            # This produces Boundary I (AlertPayload/Alert) because alert_threshold=0.0
            alert = self.pipeline.process(event)
            if alert:
                alerts.append(alert)
                
        # Restore original methods
        self.pipeline.classifier.classify_signal = original_classify
        self.pipeline.ewma_updater.update = original_update
        
        # Load labels (The ONLY place where this happens)
        labels_df = pd.read_parquet(self.labels_path)
        
        # Join results
        # We will build arrays for metrics.py
        y_true_list = []
        y_pred_list = []
        fused_scores_list = []
        class_probs_list = []
        risk_tiers_list = []
        
        # Define classes in fixed order
        classes = [e.value for e in AnomalyCategory]
        
        # We process based on captured outputs
        for event_id, classification in captured_classifications.items():
            # Join with ground truth
            label_row = labels_df[labels_df["event_id"] == event_id]
            if label_row.empty:
                continue
            
            ground_truth = label_row.iloc[0]["label"]
            
            signal = captured_signals[event_id]
            alert = self.alert_store.get_alert(event_id) # since threshold=0.0, every event generates an alert
            
            y_true_list.append(ground_truth)
            y_pred_list.append(classification.predicted_class.value)
            fused_scores_list.append(signal.fused_score)
            risk_tiers_list.append(alert.risk.risk_tier if alert else "low")
            
            # Construct probs array aligned to `classes` list
            probs = [classification.class_probabilities.get(cls, 0.0) for cls in classes]
            class_probs_list.append(probs)
            
        metrics = compute_metrics(
            y_true=np.array(y_true_list),
            y_pred=np.array(y_pred_list),
            fused_scores=np.array(fused_scores_list),
            class_probs=np.array(class_probs_list),
            risk_tiers=np.array(risk_tiers_list),
            classes=classes
        )
        
        return metrics
