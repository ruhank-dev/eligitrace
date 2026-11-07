from pydantic import BaseModel


class HealthOut(BaseModel):
    status: str  # loading | ready | error | disabled
    detail: str | None
    device: str | None
    dtype: str | None
    target_model: str
    draft_model: str
    adapter: str | None
    adapter_sha256: str | None
    draft_loaded: bool
    gpu: str | None
    vram_used_gb: float | None
    vram_total_gb: float | None
    busy: bool
    versions: dict
