"""E3: numerical validation grid + honest runtime/memory benchmark.

Compares: reference (explicit fp32 scores), tiled online-softmax (ours), PyTorch SDPA with each
available backend (MATH, EFFICIENT, FLASH - flash needs sm_80+, so it is reported unavailable on
Turing). Timing: warm-up, torch.cuda.synchronize(), repeated runs; memory: peak allocated delta.
"""

import csv
import itertools
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from attention.reference import causal_mask, sdpa_reference
from attention.tiled import tiled_attention
from settings.config import get_settings


def _rand(B, Hq, Hkv, Sq, Sk, D, dtype, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda h, s: torch.randn(B, h, s, D, generator=g).to(dev, dtype)  # noqa: E731
    return mk(Hq, Sq), mk(Hkv, Sk), mk(Hkv, Sk)


def torch_sdpa(q, k, v, causal, backend):
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    with sdpa_kernel(backend):
        if causal and q.shape[2] == k.shape[2]:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        mask = ~causal_mask(q.shape[2], k.shape[2], q.device) if causal else None
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def validation(dev) -> list[dict]:
    rows = []
    grid = itertools.product([(1, 1, 1), (2, 4, 4), (1, 12, 2)], [(1, 1), (1, 513), (7, 7), (64, 64), (129, 129),
                             (37, 300), (512, 512), (1000, 1000)], [32, 64, 128], [True, False],
                             [torch.float32, torch.bfloat16, torch.float16])
    for (B, Hq, Hkv), (Sq, Sk), D, causal, dt in grid:
        if dt != torch.float32 and dev.type == "cpu":
            continue
        q, k, v = _rand(B, Hq, Hkv, Sq, Sk, D, dt, dev)
        ref64 = sdpa_reference(q.double(), k.double(), v.double(), causal=causal,
                               compute_dtype=torch.float64)  # genuine float64 ground truth
        for block in (16, 64, 128):
            til = tiled_attention(q, k, v, causal=causal, block_q=block, block_k=block)
            err = (til.double() - ref64).abs()
            rows.append({"B": B, "Hq": Hq, "Hkv": Hkv, "Sq": Sq, "Sk": Sk, "D": D, "causal": causal,
                         "dtype": str(dt).split(".")[-1], "block": block, "impl": "tiled",
                         "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean())})
        for name, out in (("reference", sdpa_reference(q, k, v, causal=causal)),
                          ("torch_sdpa", torch_sdpa(q, k, v, causal, SDPBackend.MATH))):
            err = (out.double() - ref64).abs()
            rows.append({"B": B, "Hq": Hq, "Hkv": Hkv, "Sq": Sq, "Sk": Sk, "D": D, "causal": causal,
                         "dtype": str(dt).split(".")[-1], "block": "", "impl": name,
                         "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean())})
    return rows


def _time(fn, dev, warmup=2, reps=5):
    for _ in range(warmup):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    ts = []
    for _ in range(reps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if dev.type == "cuda" else float("nan")
    ts.sort()
    return sum(ts) / len(ts), ts[len(ts) // 2], peak


def speed(dev) -> list[dict]:
    rows = []
    H, Hkv, D = 12, 2, 128  # Qwen2.5-1.5B attention geometry
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32  # model precision
    seqs = [256, 512, 1024, 2048, 4096] if dev.type == "cuda" else [256, 512, 1024]
    impls = {
        "reference": lambda q, k, v: sdpa_reference(q, k, v, causal=True),
        "tiled_b64": lambda q, k, v: tiled_attention(q, k, v, causal=True, block_q=64, block_k=64),
        "tiled_b128": lambda q, k, v: tiled_attention(q, k, v, causal=True, block_q=128, block_k=128),
        "torch_math": lambda q, k, v: torch_sdpa(q, k, v, True, SDPBackend.MATH),
        "torch_efficient": lambda q, k, v: torch_sdpa(q, k, v, True, SDPBackend.EFFICIENT_ATTENTION),
        "torch_flash": lambda q, k, v: torch_sdpa(q, k, v, True, SDPBackend.FLASH_ATTENTION),
    }
    for S in seqs:
        q, k, v = _rand(1, H, Hkv, S, S, D, dt, dev)
        for name, fn in impls.items():
            try:
                mean, med, peak = _time(lambda: fn(q, k, v), dev)
                rows.append({"device": dev.type, "dtype": str(dt).split(".")[-1], "seq": S, "impl": name,
                             "mean_ms": mean * 1e3, "median_ms": med * 1e3, "peak_extra_mib": peak,
                             "status": "ok"})
            except (RuntimeError, torch.OutOfMemoryError) as e:
                msg = "OOM" if "out of memory" in str(e).lower() else "unavailable"
                rows.append({"device": dev.type, "dtype": str(dt).split(".")[-1], "seq": S, "impl": name,
                             "mean_ms": "", "median_ms": "", "peak_extra_mib": "", "status": msg})
                torch.cuda.empty_cache() if dev.type == "cuda" else None
        print(f"seq {S} done", flush=True)
    return rows


def _write(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-only", action="store_true",
                    help="rerun only the accuracy grid (float64 truth); keep the committed speed benchmark")
    args = ap.parse_args()
    out = get_settings().results_dir / "attention"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = validation(dev)
    _write(val, out / "validation.csv")
    summary = {}
    for r in val:
        key = f"{r['impl']}|{r['dtype']}"
        s = summary.setdefault(key, {"cases": 0, "max_abs_err": 0.0, "mean_abs_err_avg": 0.0})
        s["cases"] += 1
        s["max_abs_err"] = max(s["max_abs_err"], r["max_abs_err"])
        s["mean_abs_err_avg"] += r["mean_abs_err"]
    for s in summary.values():
        s["mean_abs_err_avg"] /= s["cases"]
    if args.validation_only:
        old = json.loads((out / "summary.json").read_text())
        old["validation_summary"] = summary
        old["validation_truth"] = "float64 (sdpa_reference with compute_dtype=torch.float64)"
        (out / "summary.json").write_text(json.dumps(old, indent=2))
        print(json.dumps(summary, indent=2))
        return
    rows = speed(dev) + (speed(torch.device("cpu")) if dev.type == "cuda" else [])
    _write(rows, out / "benchmark.csv")
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "capability": torch.cuda.get_device_capability(0) if dev.type == "cuda" else None,
           "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
           "validation_summary": summary, "validation_truth": "float64", "timing": "2 warm-up + 5 synchronized repeats"}
    (out / "summary.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
