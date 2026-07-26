"""Stream Router (M13)."""

import asyncio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from anomaly_detection.api.dependencies import get_alert_stream_queue
from anomaly_detection.common.models.alerts import AlertSummary

router = APIRouter()

@router.get("/alerts")
async def stream_alerts(
    queue: asyncio.Queue | None = Depends(get_alert_stream_queue)
):
    """
    Server-Sent Events (SSE) endpoint pushing new alerts live.
    """
    if queue is None:
        raise HTTPException(status_code=501, detail="Streaming not configured")

    async def event_generator():
        while True:
            summary: AlertSummary = await queue.get()
            data = summary.model_dump_json()
            yield f"data: {data}\n\n"
            queue.task_done()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
