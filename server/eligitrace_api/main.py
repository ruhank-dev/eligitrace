"""EligiTrace API: app creation, error mapping, routers and the static web UI. No business logic here."""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.eligitrace_api.config import get_settings
from server.eligitrace_api.database import init_db
from server.eligitrace_api.routers import health, lab, screening, trials
from server.eligitrace_api.services.model_registry import ModelNotReady, registry
from server.eligitrace_api.services.screening_service import InputTooLong
from server.eligitrace_api.services.trial_fetch import TrialFetchError

WEB = get_settings().repo_root / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if get_settings().load_model_on_startup:
        registry.start_loading()  # background; /api/health reports progress
    yield


app = FastAPI(title="EligiTrace API", version="1.0.0", lifespan=lifespan,
              description="Evidence-traced clinical-trial eligibility screening. "
                          "Educational prototype - not a clinical enrollment decision.")


def _err(status: int, code: str, msg: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "detail": msg})


@app.exception_handler(ModelNotReady)
async def _not_ready(_: Request, e: ModelNotReady):
    return _err(503, "model_not_ready", f"The model is not ready: {e}")


@app.exception_handler(InputTooLong)
async def _too_long(_: Request, e: InputTooLong):
    return _err(422, "input_too_long", str(e))


@app.exception_handler(TrialFetchError)
async def _upstream(_: Request, e: TrialFetchError):
    return _err(502, "trial_fetch_failed", str(e))


@app.exception_handler(torch.OutOfMemoryError)
async def _oom(_: Request, e: Exception):
    torch.cuda.empty_cache()
    return _err(507, "gpu_out_of_memory", "GPU memory exhausted. Use a shorter input or fewer tokens and retry.")


@app.exception_handler(ValueError)
async def _value(_: Request, e: ValueError):
    return _err(422, "invalid_input", str(e)[:300])


@app.exception_handler(Exception)
async def _unexpected(_: Request, e: Exception):  # never leak stack traces / paths
    return _err(500, "internal_error", f"Unexpected {type(e).__name__}. The request was not completed.")


for r in (health.router, screening.router, trials.router, lab.router):
    app.include_router(r)

app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


@app.get("/lab", include_in_schema=False)
def lab_page():
    return FileResponse(WEB / "lab.html")
