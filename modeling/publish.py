"""Publish the selected LoRA adapter to the Hugging Face Hub with a model card (needs `hf auth login`)."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from settings.config import get_settings

CARD = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags: [lora, clinical-trials, eligibility, evidence, qwen2.5]
datasets: [ncbi/TrialGPT-Criterion-Annotations]
---

# EligiTrace LoRA adapter for Qwen2.5-1.5B-Instruct

Criterion-level clinical-trial eligibility screening with sentence-level evidence. Given a numbered patient
note, a criterion type (inclusion/exclusion) and the criterion text, the model returns
`{{"label": ..., "evidence": [sentence ids]}}` with type-specific labels (included / not included / excluded /
not excluded / not enough information / not applicable).

**Educational research prototype - not a clinical enrollment decision tool.**

- Code, prompts and full evaluation: https://github.com/ruhank-dev/eligitrace
- Training data: TRAIN split only (37 patients, 603 annotations) of `ncbi/TrialGPT-Criterion-Annotations`,
  patient-grouped split with `numpy.random.default_rng(42)`; GPT-4 columns never used.
- Training: LoRA r=16, alpha=32, dropout 0.05 on q,k,v,o,gate,up,down; bf16 base + fp32 adapters; AdamW 2e-4,
  cosine; 3 epochs, effective batch 16; selected epoch: {epoch} (best validation macro-F1).
- Validation (8 held-out patients, greedy decoding): accuracy {acc:.3f}, macro-F1 {mf1:.3f}, evidence micro-F1 {ev:.3f}.
- Adapter SHA-256 (adapter_model.safetensors): `{sha}`

Use the exact prompt from `modeling/prompts.py` in the repository.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="checkpoints/lora_r16_bf16/best")
    a = ap.parse_args()
    s = get_settings()
    path = s.repo_root / a.adapter
    vm = json.loads((s.results_dir / "training/lora_r16_bf16/val_metrics_by_epoch.json").read_text())["best"]
    import hashlib

    sha = hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()
    (path / "README.md").write_text(CARD.format(epoch=vm["epoch"], acc=vm["accuracy"], mf1=vm["macro_f1"],
                                                ev=vm["evidence_micro_f1"], sha=sha), encoding="utf-8")
    api = HfApi()
    api.create_repo(s.adapter_repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(Path(path)), repo_id=s.adapter_repo_id, repo_type="model",
                      allow_patterns=["adapter_config.json", "adapter_model.safetensors", "README.md",
                                      "tokenizer*", "special_tokens_map.json", "vocab.json", "merges.txt",
                                      "added_tokens.json", "chat_template.jinja"],
                      commit_message="EligiTrace LoRA adapter")
    print(f"published https://huggingface.co/{s.adapter_repo_id}  (sha256 {sha})")


if __name__ == "__main__":
    main()
