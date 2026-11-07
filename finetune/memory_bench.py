"""Measure peak GPU memory and step time of candidate training configurations on THIS GPU.

Each config runs in a fresh subprocess (clean CUDA state, OOM isolated). Uses the longest real
training examples so the measured peak is the worst case.
"""

import argparse
import json
import subprocess
import sys
import time

from settings.config import get_settings

CONFIGS = [
    # (strategy, grad_ckpt, micro_batch)
    ("fp32", True, 1),
    ("fp32-offload", True, 1),
    ("fp16-mixed", True, 1),
    ("bf16-mixed", False, 1),
    ("bf16-mixed", True, 1),
    ("bf16-mixed", True, 4),
    ("bf16-mixed", True, 8),
    ("qlora-nf4", True, 1),
    ("qlora-nf4", True, 4),
]
LORA = {"r": 16, "alpha": 32, "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def run_one(model_id: str, strategy: str, ckpt: bool, mb: int, steps: int = 4) -> dict:
    import torch
    from transformers import AutoTokenizer

    from dataprep.formatting import read_jsonl
    from finetune.collator import CompletionCollator, tokenize_example
    from finetune.setup_model import AMP_DTYPE, build_trainable

    s = get_settings()
    # Windows' CUDA sysmem fallback silently spills to system RAM (100x slower) instead of OOM.
    # Cap the caching allocator at physical VRAM so "does not fit" is reported honestly.
    torch.cuda.set_per_process_memory_fraction(0.98)
    tok = AutoTokenizer.from_pretrained(model_id)
    exs = read_jsonl(s.data_dir / "processed" / "train.jsonl")
    toks = sorted((tokenize_example(tok, e["messages"], 1024) for e in exs),
                  key=lambda t: -len(t["input_ids"]))[:mb]
    batch = CompletionCollator(tok.pad_token_id)(toks)
    torch.cuda.reset_peak_memory_stats()
    model = build_trainable(model_id, strategy, LORA, ckpt)
    model.train()
    first = next(model.parameters()).device
    batch = {k: v.to(first) for k, v in batch.items()}
    weights_gb = torch.cuda.memory_allocated() / 2**30
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    amp = AMP_DTYPE.get(strategy)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    times, losses = [], []
    for _ in range(steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None):
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        losses.append(float(loss))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "strategy": strategy, "grad_ckpt": ckpt, "micro_batch": mb,
        "seq_len": int(batch["input_ids"].shape[1]),
        "weights_gb": round(weights_gb, 3),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "step_s": round(sum(times[1:]) / max(len(times) - 1, 1), 3),  # first step = warm-up
        "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
        "finite_loss": all(map(lambda x: x == x and abs(x) != float("inf"), losses)),
        "trainable_params": trainable, "status": "ok",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=3, metavar=("STRATEGY", "CKPT", "MB"))
    ap.add_argument("--model", default=get_settings().target_model_id)
    a = ap.parse_args()
    if a.one:
        st, ck, mb = a.one
        print("RESULT " + json.dumps(run_one(a.model, st, ck == "1", int(mb))))
        return
    import torch

    rows = []
    for st, ck, mb in CONFIGS:
        try:
            p = subprocess.run([sys.executable, "-m", "finetune.memory_bench", "--model", a.model,
                            "--one", st, "1" if ck else "0", str(mb)],
                           capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            rows.append({"strategy": st, "grad_ckpt": ck, "micro_batch": mb, "status": "timeout >900s"})
            print(rows[-1], flush=True)
            continue
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line:
            row = json.loads(line[7:])
        else:
            err = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "unknown error"
            row = {"strategy": st, "grad_ckpt": ck, "micro_batch": mb,
                   "status": "OOM" if "out of memory" in p.stderr.lower() else f"error: {err[:160]}"}
        print(row, flush=True)
        rows.append(row)
    out = get_settings().results_dir / "training" / "memory_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0),
                               "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                               "model": a.model, "lora": LORA, "rows": rows}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
