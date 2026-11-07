"""D2/F3 mismatch investigation: is a speculative-vs-greedy divergence an algorithmic bug or a
floating-point near-tie?

1. GPU bf16: find the first diverging step; report the target's top-2 logit margin there, computed
   (a) one token at a time (as greedy does) and (b) inside the multi-token verify block (as
   speculative does). A tiny margin whose winner flips between (a) and (b) = reduction-order effect.
2. CPU fp32: rerun the same prompt; equality in fp32 confirms the algorithm is exact.
"""

import argparse
import json

import torch
from transformers import DynamicCache

from dataprep.formatting import read_jsonl
from kvcache.decoder import greedy_cached, prefill
from modeling.loader import load_model
from modeling.prompts import render_prompt
from settings.config import get_settings
from speculative.decoder import speculative_greedy


def margins(model, ids, prefix: list[int], step: int, gamma: int) -> dict:
    """Top-2 logits for position `step` computed token-by-token vs inside a (gamma+1)-token block."""
    dev = ids.device
    full = torch.cat([ids, torch.tensor([prefix[:step]], device=dev)], 1)
    with torch.inference_mode():
        c = DynamicCache()
        single = prefill(model, full, c, 0)[0, -1].float()  # last position, ordinary path
        # block path: cache everything except the last `k` tokens, then process them in one pass
        k = min(gamma + 1, full.shape[1] - 1)
        c2 = DynamicCache()
        prefill(model, full[:, :-k], c2, 0)
        block = prefill(model, full[:, -k:], c2, full.shape[1] - k)[0, -1].float()

    def top2(x):
        v, i = x.topk(2)
        return {"tokens": i.tolist(), "logits": [round(float(a), 4) for a in v], "margin": round(float(v[0] - v[1]), 4)}

    return {"single_token_path": top2(single), "block_path": top2(block),
            "max_abs_logit_diff": round(float((single - block).abs().max()), 4)}


def run(device: str, dtype: str, idx: int, gamma: int, max_new: int, adapter) -> dict:
    s = get_settings()
    t = load_model(s.target_model_id, adapter=adapter, device=device, dtype=dtype, merge=True)
    d = load_model(s.draft_model_id, device=device, dtype=dtype)
    ex = read_jsonl(s.data_dir / "processed" / "val.jsonl")[idx]
    ids = t.tokenizer(render_prompt(t.tokenizer, ex["messages"][:-1]), return_tensors="pt").input_ids.to(t.device)
    ref = greedy_cached(t.model, ids, max_new, set()).tokens
    spec = speculative_greedy(t.model, d.model, ids, gamma, max_new, set()).tokens
    k = next((i for i, (a, b) in enumerate(zip(spec, ref)) if a != b), None)
    out = {"device": device, "dtype": str(t.dtype), "example": idx, "gamma": gamma, "max_new": max_new,
           "equal": spec == ref, "first_diff": k}
    if k is not None:
        out.update({"greedy_tok": ref[k], "spec_tok": spec[k], "analysis": margins(t.model, ids, ref, k, gamma)})
    del t, d
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", type=int, default=0)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--cpu-fp32", action="store_true")
    a = ap.parse_args()
    res = [run("cuda", "bfloat16", a.example, a.gamma, a.max_new, a.adapter)]
    print(json.dumps(res[-1], indent=2), flush=True)
    if a.cpu_fp32:
        res.append(run("cpu", "float32", a.example, a.gamma, a.max_new, a.adapter))
        print(json.dumps(res[-1], indent=2), flush=True)
    out = get_settings().results_dir / "speculative" / "mismatch_investigation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(prev + res, indent=2))


if __name__ == "__main__":
    main()
