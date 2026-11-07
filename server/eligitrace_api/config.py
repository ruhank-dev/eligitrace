"""API settings live in the project-wide Pydantic Settings (settings/config.py)."""

from settings.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
