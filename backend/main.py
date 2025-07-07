from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import os

from core.config import settings
from core.logger import logger
from core.exceptions import AIVideoSlicerException, get_user_friendly_message
from api.routes import router as api_router
from api.video_routes import router as video_router
from api.websocket import websocket_router
from core.background_tasks import startup_background_tasks, shutdown_background_tasks

# Initialize Sentry for error tracking
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration

if settings.sentry_enabled and settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        release=settings.sentry_release,
        sample_rate=settings.sentry_sample_rate,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        integrations=[
            FastApiIntegration(
                transaction_style="endpoint",
                failed_request_status_codes=[500, 502, 503, 504],
            ),
            AsyncioIntegration(),
        ],
        before_send=lambda event, hint: event if settings.sentry_enabled else None,
    )
    logger.info("Sentry initialized for error tracking")

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered video editing with face recognition and script-driven assembly",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(video_router, prefix="/api/video")
app.include_router(websocket_router, prefix="/api/video")

# Event handlers
@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    await startup_background_tasks()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop background tasks on application shutdown."""
    await shutdown_background_tasks()

# Static file serving for uploads/downloads
if os.path.exists(settings.output_dir):
    app.mount("/downloads", StaticFiles(directory=settings.output_dir), name="downloads")

# Global exception handler
@app.exception_handler(AIVideoSlicerException)
async def custom_exception_handler(request, exc: AIVideoSlicerException):
    # Add request context to Sentry
    sentry_sdk.set_context("request", {
        "url": str(request.url),
        "method": request.method,
        "headers": dict(request.headers),
        "client_ip": request.client.host if request.client else None
    })
    
    # Add user context if available (from session or auth)
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        sentry_sdk.set_user({"id": session_id})
    
    # Add custom tags
    sentry_sdk.set_tag("error_type", "application_error")
    sentry_sdk.set_tag("error_code", exc.error_code)
    
    # Capture the exception (already done in AIVideoSlicerException but adding context)
    sentry_sdk.capture_exception(exc)
    
    logger.error(f"Application error: {exc.message}")
    
    return JSONResponse(
        status_code=400,
        content={
            "error": True,
            "message": get_user_friendly_message(exc.error_code) if exc.error_code else exc.message,
            "error_code": exc.error_code,
            "type": type(exc).__name__
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    # Add comprehensive request context to Sentry
    sentry_sdk.set_context("request", {
        "url": str(request.url),
        "method": request.method,
        "headers": dict(request.headers),
        "client_ip": request.client.host if request.client else None
    })
    
    # Add user context if available
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        sentry_sdk.set_user({"id": session_id})
    
    # Add tags for unhandled exceptions
    sentry_sdk.set_tag("error_type", "unhandled_exception")
    sentry_sdk.set_tag("exception_name", type(exc).__name__)
    
    # Capture unhandled exceptions
    sentry_sdk.capture_exception(exc)
    
    logger.error(f"Unexpected error: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "An unexpected error occurred. Please try again.",
            "type": "InternalServerError"
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "test_mode": settings.test_mode_enabled
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI Video Slicer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }



# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.PROJECT_NAME} API server")
    # Cleanup logic would go here

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 