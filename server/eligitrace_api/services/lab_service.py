"""Live inference comparisons for the Inference Lab (runs on the GPU worker; results are real timings)."""

import statistics
import time

import torch

from attention.reference import sdpa_reference
from attention.tiled import tiled_attention
from dataprep.formatting import read_jsonl
from kvcache.decoder import greedy_cached
from kvcache.reference import greedy_no_cache
from modeling.loader import eos_ids
from modeling.prompts import render_prompt
from server.eligitrace_api.config import get_settings
from server.eligitrace_api.schemas.lab import AttentionLabIn, DecodeLabIn
from server.eligitrace_api.services.model_registry import registry
from speculative.decoder import speculative_greedy


def _val_examples() -> list[dict]:
    return read_jsonl(get_settings().data_dir / "processed" / "val.jsonl")


def _peak_reset(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated()
    return 0


def run_decode(p: DecodeLabIn) -> dict:
    lm = registry.require()
    tok, dev = lm.tokenizer, lm.device
    if p.prompt_source == "custom" and p.custom_prompt:
        msgs = [{"role": "user", "content": p.custom_prompt}]
    else:
        ex = _val_examples()[p.example_index % 50]
        msgs = ex["messages"][:-1]
    ids = tok(render_prompt(tok, msgs), return_tensors="pt").input_ids.to(dev)
    eos = set() if p.ignore_eos else eos_ids(tok, lm.model)
    runners = {
        "no_cache": lambda: greedy_no_cache(lm.model, ids, p.max_new_tokens, eos),
        "kv_cached": lambda: greedy_cached(lm.model, ids, p.max_new_tokens, eos),
        "speculative": lambda: speculative_greedy(lm.model, registry.draft.model, ids, p.gamma, p.max_new_tokens, eos),
    }
    runners["kv_cached"]()  # warm-up (kernels, allocator)
    ref = runners["kv_cached"]().tokens
    out = {"prompt_tokens": int(ids.shape[1]), "reference_mode": "kv_cached", "modes": []}
    for mode in p.modes:
        runs, peaks = [], []
        for _ in range(p.repeats):
            base = _peak_reset(dev)
            runs.append(runners[mode]())
            if dev.type == "cuda":
                peaks.append((torch.cuda.max_memory_allocated() - base) / 2**20)
        r0 = runs[0]
        row = {"mode": mode, "output_tokens": len(r0.tokens),
               "prefill_ms": statistics.mean(r.prefill_s for r in runs) * 1e3,
               "decode_ms": statistics.mean(r.decode_s for r in runs) * 1e3,
               "total_ms": statistics.mean(r.total_s for r in runs) * 1e3,
               "tokens_per_s": len(r0.tokens) / statistics.mean(r.total_s for r in runs),
               "peak_extra_mib": statistics.mean(peaks) if peaks else None,
               "tokens_equal_reference": r0.tokens == ref, "repeats": p.repeats}
        if mode == "speculative":
            row.update({"gamma": p.gamma, "acceptance_rate": r0.acceptance_rate, "rounds": r0.rounds,
                        "target_forwards": r0.target_forwards})
        out["modes"].append(row)
    base_ms = next((m["total_ms"] for m in out["modes"] if m["mode"] == "kv_cached"), None)
    for m in out["modes"]:
        m["speedup_vs_kv_cached"] = base_ms / m["total_ms"] if base_ms else None
    out["output_text"] = tok.decode(ref, skip_special_tokens=True)
    return out


def run_attention(p: AttentionLabIn) -> dict:
    if p.heads % p.kv_heads:
        raise ValueError("heads must be a multiple of kv_heads (grouped-query attention)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = getattr(torch, p.dtype)
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(1, p.heads, p.seq_len, p.head_dim, generator=g).to(dev, dt)
    k = torch.randn(1, p.kv_heads, p.seq_len, p.head_dim, generator=g).to(dev, dt)
    v = torch.randn(1, p.kv_heads, p.seq_len, p.head_dim, generator=g).to(dev, dt)
    truth = sdpa_reference(q.double(), k.double(), v.double(), causal=p.causal, compute_dtype=torch.float64)
    impls = {"reference": lambda: sdpa_reference(q, k, v, causal=p.causal),
             "tiled_online_softmax": lambda: tiled_attention(q, k, v, causal=p.causal, block_q=p.block, block_k=p.block),
             "torch_sdpa": lambda: torch.nn.functional.scaled_dot_product_attention(
                 q, k.repeat_interleave(p.heads // p.kv_heads, 1), v.repeat_interleave(p.heads // p.kv_heads, 1),
                 is_causal=p.causal)}
    rows = []
    for name, fn in impls.items():
        o = fn()
        base = _peak_reset(dev)
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            o = fn()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        err = (o.double() - truth).abs()
        rows.append({"impl": name, "ms": statistics.median(ts) * 1e3, "max_abs_err": float(err.max()),
                     "mean_abs_err": float(err.mean()),
                     "peak_extra_mib": (torch.cuda.max_memory_allocated() - base) / 2**20 if dev.type == "cuda" else None})
    return {"device": dev.type, "dtype": p.dtype, "truth": "float64 reference", "rows": rows,
            "score_matrix_mib_if_materialised": p.heads * p.seq_len * p.seq_len * 4 / 2**20}
