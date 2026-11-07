"""The single prompt contract shared by fine-tuning, evaluation and the application."""

import json

from modeling.labels import LABELS

SYSTEM_PROMPT = (
    "You are a clinical-trial eligibility screener. Judge ONE criterion against a patient note "
    "whose sentences are numbered. Use only facts stated in the note. If the note does not contain "
    "the facts needed, answer \"not enough information\" - missing evidence is never proof. "
    "Answer \"not applicable\" only when the criterion cannot apply to this patient. "
    "Cite the IDs of the sentences that support your label (empty list if none). "
    "Reply with JSON only."
)


def format_note(sentences: dict[int, str]) -> str:
    return "\n".join(f"{i}. {t}" for i, t in sorted(sentences.items()))


def build_messages(sentences: dict[int, str], criterion_type: str, criterion_text: str) -> list[dict]:
    allowed = " | ".join(LABELS[criterion_type])
    user = (
        f"Patient note:\n{format_note(sentences)}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text.strip()}\n"
        f"Allowed labels: {allowed}\n"
        'Return: {"label": "<allowed label>", "evidence": [<sentence ids>]}'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def format_target(label: str, evidence: list[int]) -> str:
    return json.dumps({"label": label, "evidence": sorted(set(evidence))}, separators=(", ", ": "))


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


if __name__ == "__main__":
    import argparse

    from dataprep.loader import load_clean

    ap = argparse.ArgumentParser(description="Print the exact prompt for one annotation")
    ap.add_argument("--annotation-id", type=int, default=0)
    a = ap.parse_args()
    df = load_clean()
    row = df[df.annotation_id == a.annotation_id].iloc[0]
    for m in build_messages(row.sentences, row.criterion_type, row.criterion_text):
        print(f"--- {m['role']} ---\n{m['content']}")
    print("--- target ---\n" + format_target(row.expert_eligibility, row.expert_sentences))
