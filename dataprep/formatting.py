"""Turn cleaned rows into chat-format SFT examples (prompt messages + JSON target)."""

import json
from pathlib import Path

import pandas as pd

from modeling.prompts import build_messages, format_target


def to_example(row) -> dict:
    messages = build_messages(row.sentences, row.criterion_type, row.criterion_text)
    messages.append({"role": "assistant", "content": format_target(row.expert_eligibility, row.expert_sentences)})
    return {
        "annotation_id": int(row.annotation_id),
        "patient_id": row.patient_id,
        "criterion_type": row.criterion_type,
        "label": row.expert_eligibility,
        "evidence": list(row.expert_sentences),
        "messages": messages,
    }


def write_jsonl(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("train", "val", "test"):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in df[df["split"] == name].itertuples():
                f.write(json.dumps(to_example(row)) + "\n")
        paths[name] = path
    return paths


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
