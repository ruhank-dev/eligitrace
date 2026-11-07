"""Reference greedy decoding WITHOUT cache reuse: every step recomputes the full sequence."""

import time

import torch

from kvcache.decoder import DecodeResult, _sync


@torch.inference_mode()
def greedy_no_cache(
    model, input_ids: torch.Tensor, max_new_tokens: int, eos_token_ids: set[int] | None = None
) -> DecodeResult:
    dev = input_ids.device
    eos = eos_token_ids or set()
    seq = input_ids

    def step(s):
        out = model(input_ids=s, attention_mask=torch.ones_like(s), use_cache=False)
        return int(out.logits[0, -1].argmax())

    _sync(dev)
    t0 = time.perf_counter()
    tok = step(seq)  # "prefill" = first full forward
    _sync(dev)
    res = DecodeResult(tokens=[tok], prefill_s=time.perf_counter() - t0)
    t1 = time.perf_counter()
    while len(res.tokens) < max_new_tokens and tok not in eos:
        seq = torch.cat([seq, torch.tensor([[tok]], device=dev)], dim=1)
        tok = step(seq)
        res.tokens.append(tok)
    _sync(dev)
    res.decode_s = time.perf_counter() - t1
    return res
