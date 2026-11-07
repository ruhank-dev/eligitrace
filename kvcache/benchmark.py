"""D2 + D3: greedy-token equality (cached vs no-cache) and prefill/decode/throughput/memory benchmark.

Inputs are built from real VALIDATION prompts (test split is never touched). Long inputs are made
by concatenating several validation patient notes in one prompt. EOS is disabled during timing so
every run generates exactly the requested number of tokens.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

from dataprep.formatting import read_jsonl
from kvcache.decoder import greedy_cached
from kvcache.reference import greedy_no_cache
from modeling.loader import load_model
from modeling.prompts import render_prompt
from settings.config import get_settings

INPUT_LENS = (128, 512, 1024)
OUTPUT_LENS = (16, 64, 128)  # no-cache reference is O(n^2): 128 tokens after 1024 already ~4 min/run


def build_inputs(tokenizer, device) -> dict[int, torch.Tensor]:
    """Return {target_len: input_ids} using real val prompts, trimmed/extended to the target length."""
    val = read_jsonl(get_settings().data_dir / "processed" / "val.jsonl")
    pool: list[int] = []
    for ex in val:
        pool += tokenizer(render_prompt(tokenizer, ex["messages"][:-1]))["input_ids"]
        if len(pool) > max(INPUT_LENS) + 10:
            break
    return {n: torch.tensor([pool[:n]], device=device) for n in INPUT_LENS}


def measure(fn, dev, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    pre, dec, tps, mem = [], [], [], []
    for _ in range(repeats):
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        r = fn()
        pre.append(r.prefill_s)
        dec.append(r.decode_s)
        tps.append(len(r.tokens) / r.total_s)
        if dev.type == "cuda":
            mem.append((torch.cuda.max_memory_allocated() - base) / 2**20)
    sd = lambda x: statistics.stdev(x) if len(x) > 1 else 0.0  # noqa: E731
    return {"prefill_ms": statistics.mean(pre) * 1e3, "prefill_ms_sd": sd(pre) * 1e3,
            "decode_ms": statistics.mean(dec) * 1e3, "decode_ms_sd": sd(dec) * 1e3,
            "tokens_per_s": statistics.mean(tps), "tokens_per_s_sd": sd(tps),
            "peak_extra_mib": statistics.mean(mem) if mem else float("nan"), "n_tokens": len(r.tokens)}


def first_divergence(model, ids, a: list[int], b: list[int]) -> dict:
    """Locate first mismatch and report the target's top-2 logit margin there (fp16 near-tie?)."""
    i = next(k for k, (x, y) in enumerate(zip(a, b)) if x != y)
    seq = torch.cat([ids, torch.tensor([a[:i]], device=ids.device)], 1) if i else ids
    with torch.inference_mode():
        top = model(input_ids=seq).logits[0, -1].float().topk(2)
    return {"step": i, "cached_tok": a[i], "nocache_tok": b[i],
            "top2_margin": float(top.values[0] - top.values[1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()
    s = get_settings()
    lm = load_model(adapter=a.adapter, merge=True)
    dev, model = lm.device, lm.model
    inputs = build_inputs(lm.tokenizer, dev)
    rows, eq = [], []
    for n_in in INPUT_LENS:
        ids = inputs[n_in]
        for n_out in OUTPUT_LENS:
            c = greedy_cached(model, ids, n_out, set())
            r = greedy_no_cache(model, ids, n_out, set())
            same = c.tokens == r.tokens
            rec = {"input_len": n_in, "output_len": n_out, "equal": same,
                   "n_match_prefix": next((k for k, (x, y) in enumerate(zip(c.tokens, r.tokens)) if x != y),
                                          len(c.tokens))}
            if not same:
                rec["divergence"] = first_divergence(model, ids, c.tokens, r.tokens)
            eq.append(rec)
            print(rec, flush=True)
            for name, fn in (("kv_cached", lambda: greedy_cached(model, ids, n_out, set())),
                             ("no_cache", lambda: greedy_no_cache(model, ids, n_out, set()))):
                m = measure(fn, dev, a.repeats, a.warmup)
                rows.append({"mode": name, "input_len": n_in, "output_len": n_out, **m})
                print(rows[-1], flush=True)
    # cache-growth trace for the explanation figure
    tr = greedy_cached(model, inputs[128], 16, set(), record=True)
    out = s.results_dir / "kvcache" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    with (out / "benchmark.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cfg = model.config
    kv_bytes = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2
    (out / "equality.json").write_text(json.dumps({
        "model": lm.model_id, "adapter": str(a.adapter) if a.adapter else None, "dtype": str(lm.dtype),
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "torch": torch.__version__,
        "all_equal": all(e["equal"] for e in eq), "cases": eq,
        "timing": f"{a.warmup} warm-up + {a.repeats} synchronized repeats, batch 1, EOS disabled",
        "kv_bytes_per_token_theoretical": kv_bytes,
        "cache_growth_trace": {"cache_len": tr.cache_lengths, "layer0_k_shape": [list(x) for x in tr.layer0_k_shapes]},
    }, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
