"""FastAPI middleware and global exception handlers."""

import logging
from typing import Callable, Any, Awaitable
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import time

logger = logging.getLogger(__name__)


def setup_middleware(app: FastAPI, cors_origins: list[str]) -> None:
    """Register all middleware and global exception handlers to the FastAPI app."""
    
    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Request Logging Middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Any]]) -> Any:
        start_time = time.time()
        
        response = None
        try:
            response = await call_next(request)
        finally:
            process_time = time.time() - start_time
            status_code = response.status_code if response else 500
            
            logger.info(
                "API Request",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": status_code,
                    "process_time_ms": round(process_time * 1000, 2),
                    "client_ip": request.client.host if request.client else None,
                },
            )
        return response

    # 3. Global Exception Handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Prevent tracebacks from leaking in 500 responses."""
        logger.error(
            "Unhandled server error",
            exc_info=exc,
            extra={"method": request.method, "url": str(request.url)}
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred while processing the request."
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Format Pydantic validation errors consistently."""
        logger.warning(
            "Request validation error",
            extra={"method": request.method, "url": str(request.url), "errors": exc.errors()}
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "details": jsonable_encoder(exc.errors())
            },
        )
