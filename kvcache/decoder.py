"""Cache-aware greedy autoregressive decoding written explicitly on top of model forward passes.

Flow (batch size 1):
  prefill : run the whole prompt once, with positions 0..T-1 and a causal mask (built inside the
            model from attention_mask + cache_position); every layer appends K,V for T tokens.
  decode  : feed ONLY the newest token with position T+t; the layer attends over the cached
            T+t keys plus its own key. No causal mask is needed for a single query at the last
            position (it may see everything before it), only the padding mask, which we extend by 1.
RoPE is applied to K before it is cached, so cached keys keep their absolute positions and the
new query must get the correct absolute position (position_ids / cache_position) - this is what
the equality test against the no-cache reference verifies.
"""

import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class DecodeResult:
    tokens: list[int]
    prefill_s: float = 0.0
    decode_s: float = 0.0
    step_s: list[float] = field(default_factory=list)
    cache_lengths: list[int] = field(default_factory=list)
    layer0_k_shapes: list[tuple] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s

    @property
    def decode_tok_per_s(self) -> float:
        n = max(len(self.tokens) - 1, 0)  # first token comes from prefill logits
        return n / self.decode_s if self.decode_s > 0 else float("nan")


def cache_layer0_shape(cache: DynamicCache) -> tuple:
    if hasattr(cache, "layers"):
        return tuple(cache.layers[0].keys.shape)
    return tuple(cache.key_cache[0].shape)


@torch.inference_mode()
def prefill(model, input_ids: torch.Tensor, cache: DynamicCache, start: int = 0):
    """Run `input_ids` ([1, n]) through the model at absolute positions start..start+n-1,
    appending to `cache`. Returns logits [1, n, vocab]."""
    n = input_ids.shape[1]
    pos = torch.arange(start, start + n, device=input_ids.device)
    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones(1, start + n, dtype=torch.long, device=input_ids.device),
        position_ids=pos.unsqueeze(0),
        cache_position=pos,
        past_key_values=cache,
        use_cache=True,
    )
    return out.logits


@torch.inference_mode()
def greedy_cached(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: set[int] | None = None,
    record: bool = False,
) -> DecodeResult:
    dev = input_ids.device
    eos = eos_token_ids or set()
    T = input_ids.shape[1]
    cache = DynamicCache()

    _sync(dev)
    t0 = time.perf_counter()
    logits = prefill(model, input_ids, cache, start=0)
    next_tok = int(logits[0, -1].argmax())
    _sync(dev)
    res = DecodeResult(tokens=[next_tok], prefill_s=time.perf_counter() - t0)
    assert cache.get_seq_length() == T, "prefill must cache exactly T positions"

    t1 = time.perf_counter()
    while len(res.tokens) < max_new_tokens and next_tok not in eos:
        ts = time.perf_counter() if record else 0.0
        step_in = torch.tensor([[next_tok]], device=dev)
        logits = prefill(model, step_in, cache, start=T + len(res.tokens) - 1)
        next_tok = int(logits[0, -1].argmax())
        res.tokens.append(next_tok)
        if record:
            _sync(dev)
            res.step_s.append(time.perf_counter() - ts)
            res.cache_lengths.append(cache.get_seq_length())
            res.layer0_k_shapes.append(cache_layer0_shape(cache))
    _sync(dev)
    res.decode_s = time.perf_counter() - t1
    # cache holds prompt + all generated tokens except the last one (never fed back)
    assert cache.get_seq_length() == T + len(res.tokens) - 1
    return res
