"""Inference router for processing events."""

from typing import List, Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
import logging

from anomaly_detection.common.models.access_log import AccessLogInference
from anomaly_detection.common.models.alerts import AlertSummary
from anomaly_detection.api.dependencies import get_orchestrator
from src.orchestrator.pipeline import InferencePipeline

logger = logging.getLogger(__name__)

router = APIRouter()


class InferenceEventPayload(AccessLogInference):
    """Wrapper around AccessLogInference to explicitly reject labels."""
    
    @model_validator(mode="before")
    @classmethod
    def reject_labels(cls, data: Any) -> Any:
        if isinstance(data, dict) and "label" in data:
            raise ValueError("Input events must not contain a ground-truth 'label' field.")
        return data


class InferenceBatchRequest(BaseModel):
    """Batch payload of inference events."""
    events: List[InferenceEventPayload] = Field(..., min_length=1, max_length=1000)


class InferenceBatchResponse(BaseModel):
    """Lightweight response indicating generated alerts."""
    status: str = Field(default="success")
    processed_count: int
    alerts: List[AlertSummary]


@router.post(
    "/events",
    response_model=InferenceBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Process a batch of events through the inference pipeline"
)
async def process_inference_events(
    request: InferenceBatchRequest,
    orchestrator: InferencePipeline = Depends(get_orchestrator),
) -> InferenceBatchResponse:
    """
    Run a batch of label-stripped boundary-B-shaped events through the full
    orchestrator pipeline. Returns a lightweight list of alerts generated.
    """
    alerts_generated: List[AlertSummary] = []
    
    for event in request.events:
        try:
            alert = orchestrator.process(event)
            if alert:
                # Convert the full Alert (Boundary K) into the lightweight AlertSummary
                # to avoid a massive response body, while keeping the primary keys.
                summary = AlertSummary.model_validate(alert)
                alerts_generated.append(summary)
        except Exception as e:
            # We catch exceptions per event to allow batch to continue if needed,
            # or we could fail the whole batch. The requirements state: 
            # "Errors from any pipeline stage (e.g. a malformed feature vector) must surface as a proper 4xx/5xx"
            # Let's fail the batch since that's standard for atomic endpoints.
            logger.error(f"Pipeline error processing event {event.event_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing event {event.event_id}: {str(e)}"
            )

    return InferenceBatchResponse(
        processed_count=len(request.events),
        alerts=alerts_generated
    )
