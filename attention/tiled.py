"""Tiled attention with online softmax (FlashAttention-style forward pass, pure PyTorch).

For each block of queries we stream over blocks of keys/values and keep, per query row:
  m   running maximum of the scores seen so far
  l   running softmax normaliser  sum_j exp(s_j - m)
  acc running un-normalised output sum_j exp(s_j - m) v_j
When a new key block raises the maximum to m_new, the old statistics are rescaled by
alpha = exp(m - m_new), so the final acc / l equals softmax(S) V exactly, without ever
materialising the full [Sq, Sk] score matrix (only [block_q, block_k] tiles).
No fused backend is called; every step is an explicit tensor operation.
"""

import math

import torch

from attention.reference import expand_kv


def tiled_attention(q, k, v, causal: bool = False, block_q: int = 64, block_k: int = 64,
                    scale: float | None = None, return_lse: bool = False):
    B, H, Sq, D = q.shape
    k, v = expand_kv(k, v, H)
    Sk = k.shape[2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    offset = Sk - Sq  # absolute position of query i is offset + i
    out = torch.empty(B, H, Sq, v.shape[-1], dtype=q.dtype, device=q.device)
    lse = torch.empty(B, H, Sq, dtype=torch.float32, device=q.device)

    for qs in range(0, Sq, block_q):
        qe = min(qs + block_q, Sq)
        qi = q[:, :, qs:qe].float() * scale
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device)
        acc = torch.zeros((B, H, qe - qs, v.shape[-1]), device=q.device)
        # causal: keys beyond the last query's position in this block are never visible -> skip
        k_end = min(Sk, offset + qe) if causal else Sk
        for ks in range(0, k_end, block_k):
            ke = min(ks + block_k, k_end)
            s = qi @ k[:, :, ks:ke].float().transpose(-2, -1)  # [B,H,bq,bk]
            if causal and ke > offset + qs:  # tile touches the diagonal -> mask inside it
                rows = torch.arange(qs, qe, device=q.device).unsqueeze(1) + offset
                cols = torch.arange(ks, ke, device=q.device).unsqueeze(0)
                s = s.masked_fill(cols > rows, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)  # all-masked rows
            p = torch.exp(s - m_safe)
            alpha = torch.exp(m - m_safe)  # rescale factor for previous statistics (0 on first tile)
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + p @ v[:, :, ks:ke].float()
            m = m_new
        safe_l = torch.where(l == 0, torch.ones_like(l), l)
        out[:, :, qs:qe] = (acc / safe_l).to(q.dtype)
        lse[:, :, qs:qe] = (torch.where(torch.isinf(m), torch.zeros_like(m), m) + torch.log(safe_l)).squeeze(-1)
    return (out, lse) if return_lse else out
