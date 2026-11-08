"""Numerically justified equality check for decoders that differ only by bf16 rounding.

Two decoders that are exact in real arithmetic still round differently in bf16, because different matrix
shapes use different GEMM reduction orders. At the first diverging step, let delta be the largest
|logit_A - logit_B| over the vocabulary (same prefix, each decoder's own tensor shapes). If B picked a token that A
ranks `gap` logits lower, B's own logits would have to be off by more than delta unless gap <= 2*delta. We
therefore accept a divergence only if (a) delta stays below a sanity cap and (b) gap <= 2*delta.
A real defect (wrong position, stale cache) picks a token far outside this band and is rejected.

  uv run python -m speculative.ties      # writes results/speculative/tie_check.json for the real models
"""

import json

import torch
from transformers import DynamicCache

from kvcache.decoder import prefill

BF16_PATH_NOISE_CAP = 1.5  # accepted |logits(path A) - logits(path B)| (observed on real models: 0.22-0.97)


def logits_after(model, ids, prefix, mode, gamma=4):
    """Next-token logits after prompt + prefix, computed the way a decoder computes them:
    full = one no-cache forward; incremental = cache all but the last token, feed it alone (greedy_cached);
    block = cache all but the last gamma+1 tokens, feed them together (a speculative verification pass)."""
    full = torch.cat([ids, torch.tensor([prefix], device=ids.device, dtype=ids.dtype)], 1) if prefix else ids
    n = full.shape[1]
    with torch.inference_mode():
        if mode == "full" or n == 1:
            return model(input_ids=full).logits[0, -1].float()
        k = 1 if mode == "incremental" else min(gamma + 1, n - 1)
        cache = DynamicCache()
        prefill(model, full[:, :-k], cache, 0)
        return prefill(model, full[:, -k:], cache, n - k)[0, -1].float()


def compare(model, ids, a, b, mode_a, mode_b, cap=BF16_PATH_NOISE_CAP):
    """Return None if token streams are identical, else {first_diff, gap, delta, explained_by_rounding}."""
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if k is None:
        return None
    la = logits_after(model, ids, a[:k], mode_a)
    lb = logits_after(model, ids, a[:k], mode_b)
    delta = float((la - lb).abs().max())
    gap = float(la.max() - la[b[k]])
    return {"first_diff": k, "gap": gap, "delta": delta, "explained_by_rounding": delta <= cap and gap <= 2 * delta}


def main() -> None:
    from dataprep.formatting import read_jsonl
    from kvcache.decoder import greedy_cached
    from kvcache.reference import greedy_no_cache
    from modeling.loader import load_model
    from modeling.prompts import render_prompt
    from settings.config import get_settings
    from speculative.decoder import speculative_greedy

    s = get_settings()
    t, d = load_model(s.target_model_id), load_model(s.draft_model_id)
    out = []
    for i, ex in enumerate(read_jsonl(s.data_dir / "processed" / "val.jsonl")[:3]):
        ids = t.tokenizer(render_prompt(t.tokenizer, ex["messages"][:-1]), return_tensors="pt").input_ids.cuda()
        ref = greedy_cached(t.model, ids, 48).tokens
        out.append({"prompt": i, "cached_vs_nocache": compare(t.model, ids, ref, greedy_no_cache(t.model, ids, 48).tokens,
                                                                 "incremental", "full"),
                    "cached_vs_speculative_gamma4": compare(t.model, ids, ref, speculative_greedy(
                        t.model, d.model, ids, 4, 48).tokens, "incremental", "block")})
        print(out[-1], flush=True)
    path = s.results_dir / "speculative" / "tie_check.json"
    path.write_text(json.dumps({"dtype": "bfloat16", "noise_cap": BF16_PATH_NOISE_CAP, "rule": "gap <= 2*delta",
                                "results": out}, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
