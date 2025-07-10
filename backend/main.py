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

# ADD this middleware at the top to see if requests are reaching the server:

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f">>> REQUEST: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"<<< RESPONSE: {response.status_code}")
    return response

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(video_router, prefix="/api/video")
app.include_router(websocket_router, prefix="/api/video")

# Event handlers
@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    # await startup_background_tasks()  # COMMENT THIS OUT TEMPORARILY
    logger.info("Startup event completed (background tasks disabled for testing)")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop background tasks on application shutdown."""
    # await shutdown_background_tasks()  # COMMENT THIS OUT TOO
    pass

# Static file serving for uploads/downloads
if os.path.exists(settings.output_dir):
    app.mount("/downloads", StaticFiles(directory=settings.output_dir), name="downloads")

# Global exception handler
@app.exception_handler(AIVideoSlicerException)
async def custom_exception_handler(request, exc: AIVideoSlicerException):
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
    logger.error(f"Unexpected error: {str(exc)}")
    logger.error(f"Error type: {type(exc).__name__}")
    import traceback
    logger.error(f"Full traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": f"Server error: {str(exc)}",
            "type": type(exc).__name__,
            "detail": str(exc)
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Health check called!")
    return {
        "status": "MODIFIED_WORKING",  # Changed from "healthy"
        "service": settings.PROJECT_NAME,
        "version": "1.0.0", 
        "test_mode": settings.test_mode_enabled,
        "timestamp": "2025-07-10-WORKING"  # Add this line
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