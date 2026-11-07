"""Facade over segmentation, the model, explanation, aggregation and persistence.

Pattern: Facade (routers call one method; tokenizer/decoder/parser/explainer stay hidden) +
background job (criteria are screened one by one on the GPU worker; the UI polls progress).
"""

import subprocess
from datetime import UTC, datetime

import torch

from evaluation.predict import adapter_sha256
from modeling.screen import screen_criterion
from server.eligitrace_api.database import SessionLocal
from server.eligitrace_api.repositories import screening_repo
from server.eligitrace_api.schemas.screening import ScreeningIn
from server.eligitrace_api.services.aggregation import aggregate
from server.eligitrace_api.services.model_registry import registry
from server.eligitrace_api.services.sentence_splitter import segment

DISCLAIMER = ("Educational screening prototype - NOT a clinical enrollment decision. Results are model "
              "predictions per criterion; the overall outcome is a rule-based aggregation, not a ground-truth label.")
MAX_PROMPT_TOKENS = 3000


class InputTooLong(ValueError):
    pass


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, timeout=3).strip()
    except Exception:
        return "unknown"


def start(payload: ScreeningIn) -> str:
    lm = registry.require()
    sentences, _ = segment(payload.patient_note)
    if not sentences:
        raise InputTooLong("The note contains no sentences.")
    n_tok = len(lm.tokenizer("\n".join(sentences.values()))["input_ids"])
    if n_tok > MAX_PROMPT_TOKENS:
        raise InputTooLong(f"The note is {n_tok} tokens; the limit is {MAX_PROMPT_TOKENS}. Shorten it.")
    with SessionLocal() as db:
        sc = screening_repo.create(
            db, note=payload.patient_note, sentences=sentences,
            criteria=[c.model_dump() for c in payload.criteria], options=payload.options.model_dump(),
            trial_title=payload.trial_title, trial_id=payload.trial_id, patient_label=payload.patient_label)
        sid = sc.id
    registry.submit(_run, sid)
    return sid


def _run(sid: str) -> None:
    with SessionLocal() as db:
        sc = screening_repo.get(db, sid)
        screening_repo.set_status(db, sc, "running")
        lm, opts = registry.target, sc.options
        sentences = {int(k): v for k, v in sc.sentences.items()}
        draft = registry.draft if opts["decoding"] == "speculative" else None
        try:
            for row in sc.results:
                try:
                    res = screen_criterion(lm, sentences, row.criterion_type, row.criterion_text,
                                           decoding=opts["decoding"], gamma=opts["gamma"], draft=draft,
                                           explain_mode=opts["explanation"])
                    screening_repo.save_result(db, row, res, "done")
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    screening_repo.save_result(db, row, {"error": "GPU out of memory for this criterion"}, "error")
            flat = [{"position": r.position, "criterion_type": r.criterion_type, "criterion_text": r.criterion_text,
                     "label": (r.result or {}).get("label", "invalid")} for r in sc.results]
            prov = {"target_model": lm.model_id, "adapter": registry.adapter_path.name if registry.adapter_path else None,
                    "adapter_sha256": adapter_sha256(registry.adapter_path),
                    "draft_model": registry.draft.model_id if draft else None,
                    "dtype": str(lm.dtype).replace("torch.", ""), "device": str(lm.device),
                    "decoding": {"mode": opts["decoding"], "greedy": True, "max_new_tokens": 64,
                                 "gamma": opts["gamma"] if draft else None},
                    "explanation_mode": opts["explanation"], "git_commit": _commit(),
                    "completed_at": datetime.now(UTC).isoformat(timespec="seconds")}
            screening_repo.finish(db, sc, aggregate(flat), prov)
        except Exception as e:  # job-level failure is stored, never raised into the worker
            screening_repo.set_status(db, sc, "error", f"{type(e).__name__}: {str(e)[:300]}")


def to_out(sc) -> dict:
    results, done = [], 0
    for r in sc.results:
        res = r.result or {}
        done += r.status != "pending"
        results.append({
            "position": r.position, "criterion_type": r.criterion_type, "criterion_text": r.criterion_text,
            "status": r.status, "label": res.get("label"), "evidence": res.get("evidence_text", []),
            "explanation": res.get("explanation"), "explanation_source": res.get("explanation_source"),
            "missing_info": res.get("missing_info", False), "parse_ok": res.get("parse_ok"),
            "unsupported_ids": res.get("unsupported_ids", []), "decoding": res.get("decoding"),
            "decoding_stats": res.get("decoding_stats", {}), "generation_ms": res.get("generation_ms"),
            "explanation_ms": res.get("explanation_ms"), "error": res.get("error")})
    return {"id": sc.id, "created_at": sc.created_at, "status": sc.status, "error": sc.error,
            "progress": {"done": done, "total": sc.n_criteria}, "trial_title": sc.trial_title,
            "trial_id": sc.trial_id, "patient_label": sc.patient_label,
            "sentences": [{"id": int(k), "text": v} for k, v in sorted(sc.sentences.items(), key=lambda x: int(x[0]))],
            "options": sc.options, "results": results, "overall": sc.overall, "provenance": sc.provenance,
            "disclaimer": DISCLAIMER}
