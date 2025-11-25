"""
FastAPI dependencies for AtiendeSeñas API.
Provides reusable dependencies for route handlers.
"""

from backend.config import get_settings as _get_settings, Settings


def get_settings() -> Settings:
    """
    FastAPI dependency for accessing application settings.

    Usage:
        @app.get("/some-route")
        def route(settings: Settings = Depends(get_settings)):
            return settings.APP_NAME

    Returns:
        Settings: Application settings instance
    """
    return _get_settings()
