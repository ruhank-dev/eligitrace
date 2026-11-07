"""Singleton registry: loads target (+LoRA, unmerged) and draft once; serialises GPU work.

Pattern: Singleton + single-worker executor. One GPU job runs at a time, so concurrent requests
queue instead of fighting for 6 GB of VRAM.
"""

import importlib.metadata as md
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

from evaluation.predict import adapter_sha256
from modeling.loader import LoadedModel, load_model
from server.eligitrace_api.config import get_settings


class ModelNotReady(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self) -> None:
        self.status = "idle"
        self.detail: str | None = None
        self.target: LoadedModel | None = None
        self.draft: LoadedModel | None = None
        self.adapter_path: Path | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self._busy = 0
        self._lock = threading.Lock()

    def start_loading(self) -> None:
        self.status = "loading"
        self.submit(self._load)

    def _load(self) -> None:
        s = get_settings()
        try:
            ad = s.adapter_local_dir
            self.adapter_path = ad if (ad / "adapter_config.json").exists() else None
            self.target = load_model(s.target_model_id, adapter=self.adapter_path)
            self.draft = load_model(s.draft_model_id)
            self.status = "ready"
            self.detail = None if self.adapter_path else (
                "Fine-tuned adapter not found - running the BASE model. Run `task download-weights`.")
        except Exception as e:  # surfaced via /api/health; never crash the server
            self.status, self.detail = "error", f"{type(e).__name__}: {str(e)[:300]}"

    def require(self) -> LoadedModel:
        if self.status != "ready" or self.target is None:
            raise ModelNotReady(self.detail or f"model is {self.status}")
        return self.target

    def submit(self, fn, *args, **kwargs) -> Future:
        def wrapped():
            with self._lock:
                self._busy += 1
            try:
                return fn(*args, **kwargs)
            finally:
                with self._lock:
                    self._busy -= 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return self._executor.submit(wrapped)

    @property
    def busy(self) -> bool:
        return self._busy > 0

    def health(self) -> dict:
        s = get_settings()
        cuda = torch.cuda.is_available()
        free, total = torch.cuda.mem_get_info() if cuda else (0, 0)
        t = self.target
        return {
            "status": self.status, "detail": self.detail,
            "device": str(t.device) if t else None,
            "dtype": str(t.dtype).replace("torch.", "") if t else None,
            "target_model": s.target_model_id, "draft_model": s.draft_model_id,
            "adapter": self.adapter_path.name if self.adapter_path else None,
            "adapter_sha256": adapter_sha256(self.adapter_path) if self.adapter_path else None,
            "draft_loaded": self.draft is not None,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "vram_used_gb": round((total - free) / 2**30, 2) if cuda else None,
            "vram_total_gb": round(total / 2**30, 2) if cuda else None,
            "busy": self.busy,
            "versions": {p: md.version(p) for p in ("torch", "transformers", "peft")},
        }


registry = ModelRegistry()
