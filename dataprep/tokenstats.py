"""Token-length statistics and truncation audit (does any max_seq_len cut drop evidence?)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from dataprep.formatting import to_example


def _stats(xs) -> dict:
    a = np.asarray(xs)
    return {"min": int(a.min()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": int(a.max()), "mean": float(a.mean())}


def compute(df: pd.DataFrame, tokenizer_id: str, max_seq_len: int = 2048) -> dict:
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    rows = []
    for row in df.itertuples():
        ex = to_example(row)
        prompt = tok.apply_chat_template(ex["messages"][:-1], tokenize=True, add_generation_prompt=True)
        full = tok.apply_chat_template(ex["messages"], tokenize=True)
        # token index at which each sentence ends inside the prompt -> evidence-loss check
        truncated = len(full) > max_seq_len
        lost = []
        if truncated:
            for sid in row.expert_sentences:
                prefix = ex["messages"][1]["content"].split(f"\n{sid}. ", 1)[0]
                if len(tok(prefix)["input_ids"]) + 40 > max_seq_len:
                    lost.append(sid)
        rows.append({"annotation_id": int(row.annotation_id), "split": row.split,
                     "prompt_tokens": len(prompt), "target_tokens": len(full) - len(prompt),
                     "total_tokens": len(full), "truncated": truncated, "evidence_lost": lost})
    t = pd.DataFrame(rows)
    out = {"tokenizer": tokenizer_id, "max_seq_len": max_seq_len,
           "n_truncated": int(t.truncated.sum()),
           "n_with_evidence_lost": int((t.evidence_lost.map(len) > 0).sum()),
           "truncated_annotation_ids": t.loc[t.truncated, "annotation_id"].tolist(),
           "overall": {k: _stats(t[k]) for k in ("prompt_tokens", "target_tokens", "total_tokens")},
           "by_split": {s: _stats(g.total_tokens) for s, g in t.groupby("split")}}
    return out, t


def write(df: pd.DataFrame, tokenizer_id: str, out_dir: Path, max_seq_len: int = 2048) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, table = compute(df, tokenizer_id, max_seq_len)
    (out_dir / "token_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    table.drop(columns=["evidence_lost"]).to_csv(out_dir / "token_lengths.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(table.total_tokens, bins=40, color="#3b6ea5")
    ax.axvline(max_seq_len, color="#c0392b", ls="--", label=f"max_seq_len={max_seq_len}")
    ax.set_xlabel("tokens per training example (prompt + target)")
    ax.set_ylabel("examples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "token_hist.png", dpi=160)
    plt.close(fig)
    return stats
