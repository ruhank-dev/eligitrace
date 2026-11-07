"""Inference Lab: live decoding + attention comparisons, persisted as lab runs."""

import json

from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from server.eligitrace_api.config import get_settings
from server.eligitrace_api.database import get_db
from server.eligitrace_api.repositories import lab_repo
from server.eligitrace_api.schemas.lab import AttentionLabIn, DecodeLabIn, LabRunOut
from server.eligitrace_api.services import lab_service
from server.eligitrace_api.services.model_registry import registry

router = APIRouter(prefix="/api/lab", tags=["inference lab"])


def _out(run) -> dict:
    return {"id": run.id, "kind": run.kind, "created_at": run.created_at.isoformat(), "params": run.params,
            "result": run.result}


@router.post("/decode", response_model=LabRunOut, summary="Run no-cache / KV-cached / speculative decoding live")
async def decode(body: DecodeLabIn, db: Session = Depends(get_db)) -> dict:
    result = await run_in_threadpool(lambda: registry.submit(lab_service.run_decode, body).result())
    return _out(lab_repo.add(db, "decode", body.model_dump(), result))


@router.post("/attention", response_model=LabRunOut, summary="Compare reference vs tiled vs PyTorch attention live")
async def attention(body: AttentionLabIn, db: Session = Depends(get_db)) -> dict:
    result = await run_in_threadpool(lambda: registry.submit(lab_service.run_attention, body).result())
    return _out(lab_repo.add(db, "attention", body.model_dump(), result))


@router.get("/runs", response_model=list[LabRunOut], summary="Recent lab runs")
def runs(kind: str | None = Query(None, pattern="^(decode|attention)$"), limit: int = Query(20, ge=1, le=100),
         db: Session = Depends(get_db)) -> list[dict]:
    return [_out(r) for r in lab_repo.list_recent(db, kind, limit)]


@router.get("/offline", summary="Offline benchmark summaries from results/ (full repeated runs)")
def offline() -> dict:
    root = get_settings().results_dir
    out = {}
    for name, rel in {"kvcache": "kvcache/finetuned/equality.json", "speculative": "speculative/finetuned/summary.json",
                      "attention": "attention/summary.json"}.items():
        f = root / rel
        out[name] = json.loads(f.read_text()) if f.exists() else None
    for name, rel in {"kvcache_csv": "kvcache/finetuned/benchmark.csv", "speculative_csv": "speculative/finetuned/benchmark.csv",
                      "attention_csv": "attention/benchmark.csv"}.items():
        f = root / rel
        out[name] = f.read_text() if f.exists() else None
    return out
