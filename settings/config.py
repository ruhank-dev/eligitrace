"""Central configuration. The ONLY module that reads environment variables (prefix ELIGITRACE_)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_prefix="ELIGITRACE_", extra="ignore"
    )

    # Data
    dataset_id: str = "ncbi/TrialGPT-Criterion-Annotations"
    dataset_revision: str = "1cfcafde94a1560a33b4addc1638664fe28fc059"
    seed: int = 42

    # Models
    target_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_repo_id: str = "ruhank-dev/eligitrace-qwen2.5-1.5b-lora"
    adapter_local_dir: Path = REPO_ROOT / "models" / "eligitrace-lora"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"  # fp16 GEMM is ~5x slower on GTX 16xx
    max_new_tokens: int = 64

    # App
    db_url: str = f"sqlite:///{(REPO_ROOT / 'app_data' / 'eligitrace.sqlite3').as_posix()}"
    host: str = "127.0.0.1"
    port: int = 8000
    load_model_on_startup: bool = True
    clinicaltrials_base_url: str = "https://clinicaltrials.gov/api/v2/studies"

    # Paths
    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    results_dir: Path = REPO_ROOT / "results"


@lru_cache
def get_settings() -> Settings:
    return Settings()
