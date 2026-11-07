from fastapi import APIRouter

from server.eligitrace_api.schemas.health import HealthOut
from server.eligitrace_api.services.model_registry import registry

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthOut, summary="Model status, device and memory")
def health() -> dict:
    return registry.health()
