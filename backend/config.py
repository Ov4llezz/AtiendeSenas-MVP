"""
Configuration module for AtiendeSeñas API.
Uses Pydantic for settings management with singleton pattern.
"""

from typing import List
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with sensible defaults for development.
    Can be overridden via environment variables.
    """

    # Application metadata
    APP_NAME: str = "AtiendeSeñas API"
    APP_VERSION: str = "0.1.0"

    # CORS configuration - allows frontend to communicate with backend
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:5173"]

    class Config:
        """Pydantic config class"""
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with singleton pattern.
    Uses lru_cache to ensure settings are only loaded once.

    Returns:
        Settings: Application settings instance
    """
    return Settings()
