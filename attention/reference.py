"""Reference scaled dot-product attention (explicit, materialises the full score matrix).

Shapes: q [B, Hq, Sq, D], k/v [B, Hkv, Sk, D] with Hq % Hkv == 0 (grouped-query attention).
Causal convention (matches PyTorch / HF with a KV cache): query i sits at absolute position
(Sk - Sq + i) and may attend to keys j <= Sk - Sq + i.
"""

import math

import torch


def expand_kv(k: torch.Tensor, v: torch.Tensor, hq: int):
    rep = hq // k.shape[1]
    if rep == 1:
        return k, v
    return k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)


def causal_mask(sq: int, sk: int, device) -> torch.Tensor:
    """Boolean [Sq, Sk], True where attention is NOT allowed."""
    i = torch.arange(sq, device=device).unsqueeze(1)
    j = torch.arange(sk, device=device).unsqueeze(0)
    return j > i + (sk - sq)


def sdpa_reference(q, k, v, causal: bool = False, attn_mask: torch.Tensor | None = None,
                   scale: float | None = None, compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """attn_mask: optional boolean, True = masked out, broadcastable to [B, H, Sq, Sk].

    All arithmetic runs in `compute_dtype` (default fp32). Pass torch.float64 for a genuine float64 ground
    truth; the result is cast back to q.dtype (so float64 inputs give a float64 output)."""
    k, v = expand_kv(k, v, q.shape[1])
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = (q.to(compute_dtype) @ k.to(compute_dtype).transpose(-2, -1)) * scale  # [B,H,Sq,Sk] in compute_dtype
    masked = torch.zeros_like(s, dtype=torch.bool)
    if causal:
        masked = masked | causal_mask(q.shape[2], k.shape[2], q.device)
    if attn_mask is not None:
        masked = masked | attn_mask
    s = s.masked_fill(masked, float("-inf"))
    # numerically stable softmax: subtract the row max before exp; fully-masked rows -> zeros
    m = s.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    p = torch.exp(s - m)
    denom = p.sum(dim=-1, keepdim=True)
    p = p / torch.where(denom == 0, torch.ones_like(denom), denom)
    return (p @ v.to(compute_dtype)).to(q.dtype)


def naive_softmax_attention(q, k, v, scale: float | None = None) -> torch.Tensor:
    """Deliberately UNSTABLE textbook version (exp without max subtraction) - used only to show
    why the stable form matters (overflows to inf/NaN for large logits)."""
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    e = torch.exp((q @ k.transpose(-2, -1)) * scale)
    return (e / e.sum(-1, keepdim=True)) @ v
