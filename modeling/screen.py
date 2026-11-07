"""Screen ONE criterion against a numbered patient note (shared by evaluation and the app)."""

import time

import torch

from kvcache.decoder import greedy_cached
from modeling.explain import explain
from modeling.loader import eos_ids
from modeling.parser import parse_output
from modeling.prompts import build_messages, render_prompt
from speculative.decoder import speculative_greedy


@torch.inference_mode()
def screen_criterion(lm, sentences: dict[int, str], criterion_type: str, criterion_text: str,
                     decoding: str = "kv_cached", gamma: int = 4, draft=None,
                     explain_mode: str = "llm", max_new_tokens: int = 64) -> dict:
    msgs = build_messages(sentences, criterion_type, criterion_text)
    ids = lm.tokenizer(render_prompt(lm.tokenizer, msgs), return_tensors="pt").input_ids.to(lm.device)
    eos = eos_ids(lm.tokenizer, lm.model)
    t = time.perf_counter()
    stats = {}
    if decoding == "speculative" and draft is not None:
        r = speculative_greedy(lm.model, draft.model, ids, gamma, max_new_tokens, eos)
        stats = {"gamma": gamma, "acceptance_rate": r.acceptance_rate, "rounds": r.rounds}
    else:
        r = greedy_cached(lm.model, ids, max_new_tokens, eos)
        decoding = "kv_cached"
    gen_s = time.perf_counter() - t
    raw = lm.tokenizer.decode(r.tokens, skip_special_tokens=True)
    p = parse_output(raw, criterion_type, set(sentences))
    ev = {i: sentences[i] for i in p.evidence}
    t2 = time.perf_counter()
    ex = explain(lm, p.label, criterion_type, criterion_text, ev, explain_mode) if p.ok else {
        "text": "The model output could not be parsed into an allowed label; treat as unscreened.",
        "source": "none", "check": p.error}
    return {
        "criterion_type": criterion_type, "criterion_text": criterion_text,
        "label": p.label, "parse_ok": p.ok, "parse_error": p.error,
        "evidence_ids": p.evidence, "evidence_text": [{"id": i, "text": t} for i, t in sorted(ev.items())],
        "unsupported_ids": p.dropped_ids,
        "missing_info": p.label == "not enough information",
        "explanation": ex["text"], "explanation_source": ex["source"], "explanation_check": ex["check"],
        "raw_output": raw, "decoding": decoding, "decoding_stats": stats,
        "prompt_tokens": int(ids.shape[1]), "output_tokens": len(r.tokens),
        "generation_ms": round(gen_s * 1e3, 1), "explanation_ms": round((time.perf_counter() - t2) * 1e3, 1),
    }
