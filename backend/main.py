"""
AtiendeSeñas API - Main FastAPI application

This is the entry point for the backend service that handles:
- Sign language video translation (VideoMAE)
- Chatbot integration
- Full pipeline orchestration

To run the development server:
    uvicorn backend.main:app --reload --port 8000

Then visit:
    - http://localhost:8000/docs (Swagger UI)
    - http://localhost:8000/api/health (Health check)
    - http://localhost:8000/api/debug/settings (Configuration debug)
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI, Request, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.config import Settings
from backend.dependencies import get_settings


# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hola, ¿cómo estás?"
            }
        }


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="AtiendeSeñas API",
    description="API for Chilean Sign Language translation and chatbot integration",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================================
# CORS Configuration
# ============================================================================

def setup_cors(application: FastAPI, settings: Settings) -> None:
    """
    Configure CORS middleware to allow frontend requests.

    Args:
        application: FastAPI app instance
        settings: Application settings with CORS origins
    """
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
        allow_headers=["*"],  # Allows all headers
    )


# Initialize CORS with settings
setup_cors(app, get_settings())


# ============================================================================
# Global Exception Handler
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for uncaught errors.

    Args:
        request: The incoming request
        exc: The exception that was raised

    Returns:
        JSONResponse with error details
    """
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "error_type": type(exc).__name__,
            "path": str(request.url),
        },
    )


# ============================================================================
# API Routes
# ============================================================================

@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify the service is running.

    Returns:
        Dict with status, service name, and version
    """
    return {
        "status": "ok",
        "service": "AtiendeSeñas API",
        "version": "0.1.0",
    }


@app.get("/api/debug/settings")
async def debug_settings(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    """
    Debug endpoint to view current configuration.
    Useful during development to verify settings are loaded correctly.

    Args:
        settings: Injected application settings

    Returns:
        Dict with current configuration values
    """
    return {
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "cors_origins": settings.BACKEND_CORS_ORIGINS,
    }


# ============================================================================
# Translation & Pipeline Endpoints
# ============================================================================

@app.post("/api/translate")
async def translate_sign_language(
    video: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Translate sign language video to text using VideoMAE model.

    Currently returns dummy data. Will be replaced with actual model inference.

    Args:
        video: Uploaded video file (multipart/form-data)

    Returns:
        Dict with detected sign and confidence score
    """
    try:
        logger.info(f"POST /api/translate called - filename: {video.filename}, content_type: {video.content_type}")

        # Validate file type (optional but recommended)
        if video.content_type not in ["video/mp4", "video/avi", "video/mov", "video/webm"]:
            logger.warning(f"Invalid content type: {video.content_type}")

        # TODO: Replace with actual VideoMAE model inference
        # For now, return dummy data
        return {
            "sign": "placeholder_sign",
            "confidence": 0.0
        }

    except Exception as e:
        logger.error(f"Error in /api/translate: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )


@app.post("/api/chat")
async def chat(request: ChatRequest) -> Dict[str, str]:
    """
    Send message to chatbot and get response.

    Currently returns dummy data. Will be replaced with actual chatbot integration.

    Args:
        request: ChatRequest with 'text' field

    Returns:
        Dict with chatbot response

    Raises:
        HTTPException: If 'text' field is missing or empty
    """
    try:
        # Validation happens automatically via Pydantic
        if not request.text or request.text.strip() == "":
            logger.warning("POST /api/chat called with empty text")
            raise HTTPException(
                status_code=400,
                detail="Text field cannot be empty"
            )

        logger.info(f"POST /api/chat called - text: {request.text[:50]}...")

        # TODO: Replace with actual chatbot API call (OpenAI/Voiceflow/Globot)
        # For now, return dummy data
        return {
            "response": "placeholder_response"
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in /api/chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@app.post("/api/full-pipeline")
async def full_pipeline(
    video: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Complete pipeline: video translation + chatbot response.

    This endpoint orchestrates the full flow:
    1. Translate sign language video to text
    2. Send translation to chatbot
    3. Return both results

    Currently returns dummy data. Will be replaced with actual pipeline.

    Args:
        video: Uploaded video file (multipart/form-data)

    Returns:
        Dict with sign, confidence, and chat response
    """
    try:
        logger.info(f"POST /api/full-pipeline called - filename: {video.filename}")

        # TODO: Step 1 - Translate video using VideoMAE model
        # translation_result = await translate_video(video)

        # TODO: Step 2 - Send translation to chatbot
        # chat_result = await send_to_chatbot(translation_result["sign"])

        # For now, return dummy data
        return {
            "sign": "placeholder_sign",
            "confidence": 0.0,
            "chat_response": "placeholder_chat_response"
        }

    except Exception as e:
        logger.error(f"Error in /api/full-pipeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing full pipeline: {str(e)}"
        )


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event() -> None:
    """
    Run tasks on application startup.
    Currently just logs that the server is ready.
    """
    settings = get_settings()
    print(f"\n{'='*60}")
    print(f"{settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"{'='*60}")
    print(f"Server running on: http://localhost:8000")
    print(f"API docs available at: http://localhost:8000/docs")
    print(f"Health check: http://localhost:8000/api/health")
    print(f"Debug settings: http://localhost:8000/api/debug/settings")
    print(f"{'='*60}\n")
