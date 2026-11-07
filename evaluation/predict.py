"""Batch prediction over a split with the project's own KV-cached greedy decoder.

Same prompt, parser and decoding settings for base and fine-tuned models (B4). Results:
results/<tag>/<split>/predictions.jsonl + meta.json (model, adapter hash, decoding config, commit).
"""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from dataprep.formatting import read_jsonl
from evaluation.test_guard import check_split_allowed
from kvcache.decoder import greedy_cached
from modeling.loader import eos_ids, load_model
from modeling.parser import parse_output
from modeling.prompts import render_prompt
from settings.config import get_settings

DECODING = {"strategy": "greedy (argmax)", "decoder": "kvcache.greedy_cached", "max_new_tokens": 64,
            "batch_size": 1, "do_sample": False, "repetition_penalty": 1.0}


def adapter_sha256(path: Path | None) -> str | None:
    if not path:
        return None
    f = Path(path) / "adapter_model.safetensors"
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def predict_one(lm, eos, ex: dict, max_new: int) -> dict:
    msgs = ex["messages"][:-1] if ex["messages"][-1]["role"] == "assistant" else ex["messages"]
    ids = lm.tokenizer(render_prompt(lm.tokenizer, msgs), return_tensors="pt").input_ids.to(lm.device)
    t = time.perf_counter()
    res = greedy_cached(lm.model, ids, max_new, eos)
    text = lm.tokenizer.decode(res.tokens, skip_special_tokens=True)
    valid_ids = {int(k) for k in ex["sentence_ids"]} if "sentence_ids" in ex else _ids_from_prompt(msgs)
    p = parse_output(text, ex["criterion_type"], valid_ids)
    return {"raw": text, "pred_label": p.label, "pred_evidence": p.evidence, "dropped_ids": p.dropped_ids,
            "parse_ok": p.ok, "parse_error": p.error, "prompt_tokens": int(ids.shape[1]),
            "output_tokens": len(res.tokens), "latency_s": round(time.perf_counter() - t, 4)}


def _ids_from_prompt(msgs) -> set[int]:
    note = msgs[-1]["content"].split("Patient note:\n", 1)[1].split("\n\nCriterion type:", 1)[0]
    return {int(line.split(".", 1)[0]) for line in note.splitlines() if line.split(".", 1)[0].isdigit()}


def run_predictions(split: str, adapter: Path | None, tag: str, out_root: Path | None = None,
                    final: bool = False, model_id: str | None = None, limit: int = 0) -> list[dict]:
    s = get_settings()
    check_split_allowed(split, final)
    examples = read_jsonl(s.data_dir / "processed" / f"{split}.jsonl")
    if limit:
        examples = examples[:limit]
    lm = load_model(model_id, adapter=adapter)
    eos = eos_ids(lm.tokenizer, lm.model)
    out_dir = s.results_dir / tag / split if out_root is None else out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            r = {"annotation_id": ex["annotation_id"], "patient_id": ex["patient_id"],
                 "criterion_type": ex["criterion_type"], "gold_label": ex["label"],
                 "gold_evidence": ex["evidence"], **predict_one(lm, eos, ex, DECODING["max_new_tokens"])}
            recs.append(r)
            f.write(json.dumps(r) + "\n")
            if (i + 1) % 25 == 0:
                print(f"[{tag}/{split}] {i + 1}/{len(examples)}", flush=True)
    meta = {"tag": tag, "split": split, "n": len(recs), "model_id": lm.model_id,
            "adapter": str(adapter) if adapter else None, "adapter_sha256": adapter_sha256(adapter),
            "dtype": str(lm.dtype), "device": str(lm.device),
            "gpu": torch.cuda.get_device_name(0) if lm.device.type == "cuda" else None,
            "decoding": DECODING, "git_commit": git_commit(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    del lm
    torch.cuda.empty_cache()
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    recs = run_predictions(a.split, a.adapter, a.tag, final=a.final, limit=a.limit)
    from evaluation.report import summarise, write_report

    write_report(recs, get_settings().results_dir / a.tag / a.split)
    m = summarise(recs)
    print(json.dumps({"accuracy": m["all"]["accuracy"], "macro_f1": m["all"]["macro_f1"],
                      "evidence_micro_f1": m["evidence"]["micro"]["f1"],
                      "parse_fail_rate": m["parse_fail_rate"]}, indent=2))


if __name__ == "__main__":
    main()
