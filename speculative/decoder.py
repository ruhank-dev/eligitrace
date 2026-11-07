"""Greedy speculative decoding (draft -> verify -> accept prefix -> correct), hand-written.

Invariant kept after every round (seq = prompt + generated tokens so far):
  target cache holds exactly seq[:-1]         (last token is "pending", not yet fed)
  draft  cache holds seq[:-1] or seq[:-2]     (one behind only after a fully accepted round)
Round:
  1. draft feeds its pending token(s) and greedily proposes d1..dγ (γ cheap single-token passes)
  2. target runs ONE forward over [seq[-1], d1..dγ] at positions L-1..L-1+γ -> greedy g1..g(γ+1)
  3. n = longest prefix with d_i == g_i ; append d1..dn and g(n+1) (correction, or bonus if n=γ)
  4. roll back caches with DynamicCache.crop so rejected drafts leave no K/V behind.
Because every emitted token equals the target's own argmax given the same prefix, the output is
token-identical to plain greedy target decoding (up to floating-point reduction-order effects).
"""

import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from kvcache.decoder import _sync, prefill


@dataclass
class SpecResult:
    tokens: list[int]
    prefill_s: float = 0.0
    decode_s: float = 0.0
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    accepted_per_round: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else float("nan")

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s


@torch.inference_mode()
def speculative_greedy(target, draft, input_ids: torch.Tensor, gamma: int, max_new_tokens: int,
                       eos_token_ids: set[int] | None = None) -> SpecResult:
    dev = input_ids.device
    eos = eos_token_ids or set()
    seq: list[int] = input_ids[0].tolist()
    T = len(seq)
    tc, dc = DynamicCache(), DynamicCache()

    _sync(dev)
    t0 = time.perf_counter()
    # Prefill both models on the prompt. Target's last logits give the first new token.
    first = int(prefill(target, input_ids, tc, 0)[0, -1].argmax())
    prefill(draft, input_ids[:, :-1], dc, 0) if T > 1 else None  # draft holds seq[:-1]
    _sync(dev)
    res = SpecResult(tokens=[first], prefill_s=time.perf_counter() - t0, target_forwards=1, draft_forwards=1)
    seq.append(first)
    # target cache currently holds the full prompt = seq[:-1]  (invariant satisfied)

    t1 = time.perf_counter()
    while len(res.tokens) < max_new_tokens and seq[-1] not in eos:
        L = len(seq)
        g = min(gamma, max_new_tokens - len(res.tokens))  # never draft past the budget
        # ---- 1. draft: catch up on pending tokens, then propose g tokens greedily
        pending = seq[dc.get_seq_length():]
        logits = prefill(draft, torch.tensor([pending], device=dev), dc, dc.get_seq_length())
        res.draft_forwards += 1
        drafts = [int(logits[0, -1].argmax())]
        for _ in range(g - 1):
            logits = prefill(draft, torch.tensor([[drafts[-1]]], device=dev), dc, dc.get_seq_length())
            res.draft_forwards += 1
            drafts.append(int(logits[0, -1].argmax()))
        # draft cache now holds seq + d1..d(g-1) -> length L + g - 1
        # ---- 2. verify: one target pass over [seq[-1], d1..dg]
        block = torch.tensor([[seq[-1], *drafts]], device=dev)
        tl = prefill(target, block, tc, L - 1)
        res.target_forwards += 1
        greedy = tl[0].argmax(dim=-1).tolist()  # g1..g(g+1): target's choice after each prefix
        # ---- 3. accept longest matching prefix, then add the target's own next token
        n = 0
        while n < g and drafts[n] == greedy[n]:
            n += 1
        new = drafts[:n] + [greedy[n]]
        # stop exactly at EOS / budget even inside an accepted block
        for i, tkn in enumerate(new):
            if tkn in eos or len(res.tokens) + i + 1 >= max_new_tokens:
                new = new[: i + 1]
                break
        res.rounds += 1
        res.proposed += g
        res.accepted += min(n, len(new))
        res.accepted_per_round.append(min(n, len(new)))
        res.tokens.extend(new)
        seq.extend(new)
        # ---- 4. cache synchronisation (drop K/V of rejected / unused tokens)
        tc.crop(len(seq) - 1)
        dc.crop(min(dc.get_seq_length(), len(seq) - 1))
        assert tc.get_seq_length() == len(seq) - 1
        assert len(seq) - 2 <= dc.get_seq_length() <= len(seq) - 1
    _sync(dev)
    res.decode_s = time.perf_counter() - t1
    return res
