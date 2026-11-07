"""Tokenisation with completion-only labels: loss is computed on the assistant JSON only."""

import torch


def tokenize_example(tokenizer, messages: list[dict], max_len: int) -> dict:
    prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prefix mismatch; cannot mask prompt safely")
    # Qwen template ends the assistant turn with "<|im_end|>\n"; keep <|im_end|> as a target (stop token),
    # drop the trailing newline from the loss.
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if tokenizer.decode(full_ids[-1:]).strip() == "":
        labels[-1] = -100
    if len(full_ids) > max_len:
        raise ValueError(f"example longer than max_len={max_len} (audit says this never happens)")
    return {"input_ids": full_ids, "labels": labels}


class CompletionCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }
