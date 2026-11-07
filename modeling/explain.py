"""Grounded explanations for a criterion decision.

Strategy (documented in the report, decision D5):
  1. `llm`      - the BASE model (LoRA adapter disabled) writes a 1-2 sentence rationale from a prompt
                  that contains ONLY the criterion, the decided label and the cited sentences.
                  Accepted only if it passes `grounding_check` (no numbers absent from the cited
                  evidence, no contradiction of the decided label); otherwise -> template.
  2. `template` - deterministic text that quotes the cited sentences verbatim.
GPT-4 explanations from the dataset are never used (not as targets, inputs or references).
"""

import re
from contextlib import nullcontext

import torch

from kvcache.decoder import greedy_cached
from modeling.loader import eos_ids

_NUM = re.compile(r"\d+(?:\.\d+)?")

_LABEL_PHRASE = {
    "included": "meets this inclusion criterion",
    "not included": "does not meet this inclusion criterion",
    "excluded": "meets this exclusion criterion and would be excluded",
    "not excluded": "does not meet this exclusion criterion",
    "not enough information": "cannot be judged from the note",
    "not applicable": "is not applicable to this patient",
}


def template_explanation(label: str, criterion: str, evidence: dict[int, str]) -> str:
    head = f"The patient {_LABEL_PHRASE.get(label, label)}."
    if label == "not enough information":
        tail = (" The note does not state the facts this criterion requires; missing information is not"
                " treated as proof either way.")
        if evidence:
            tail += " Related context: " + " ".join(f"[{i}] “{t}”" for i, t in sorted(evidence.items()))
        return head + tail
    if not evidence:
        return head + " No specific sentence was cited."
    return head + " Supporting evidence: " + " ".join(f"[{i}] “{t}”" for i, t in sorted(evidence.items()))


def grounding_check(text: str, label: str, criterion: str, evidence: dict[int, str]) -> tuple[bool, str]:
    if not text or len(text) < 15:
        return False, "empty"
    allowed = set(_NUM.findall(" ".join(evidence.values()) + " " + criterion))
    stray = [n for n in _NUM.findall(text) if n not in allowed]
    if stray:
        return False, f"numbers not in evidence: {stray[:3]}"
    low = text.lower()
    opposite = {"included": "not included", "excluded": "not excluded"}.get(label)
    if opposite and opposite in low:
        return False, "contradicts label"
    if label == "not enough information" and any(p in low for p in _VERDICT):
        return False, "definite verdict for an undecided label"
    if label != "not enough information" and any(p in low for p in _UNDECIDED):
        return False, "undecided wording for a definite label"
    return True, "ok"


# phrases that assert a definite outcome (or its opposite) -- incompatible with "not enough information"
_VERDICT = ("does not meet", "doesn't meet", "meets the", "meets this", "falls outside", "falls within",
            "is included", "is excluded", "not included", "not excluded", "is eligible", "is ineligible")
_UNDECIDED = ("not enough information", "information is missing", "cannot be determined", "cannot be judged")


def _prompt(label: str, criterion_type: str, criterion: str, evidence: dict[int, str]) -> list[dict]:
    ev = "\n".join(f"[{i}] {t}" for i, t in sorted(evidence.items())) or "(no sentence cited)"
    return [
        {"role": "system", "content": "You explain clinical-trial screening decisions. Use ONLY the "
         "evidence lines given. Do not add facts, numbers or diagnoses that are not written there. "
         "Write 1-2 short plain sentences. Refer to evidence as [id]."},
        {"role": "user", "content": f"{criterion_type.capitalize()} criterion: {criterion}\n"
         f"Decision: {label}\nEvidence:\n{ev}\n\nExplain why the decision follows from the evidence"
         + (" and what information is missing." if label == "not enough information" else ".")},
    ]


@torch.inference_mode()
def explain(lm, label: str, criterion_type: str, criterion: str, evidence: dict[int, str],
            mode: str = "llm", max_new_tokens: int = 80) -> dict:
    fallback = template_explanation(label, criterion, evidence)
    if mode != "llm":
        return {"text": fallback, "source": "template", "check": "n/a"}
    tok = lm.tokenizer
    ids = tok.apply_chat_template(_prompt(label, criterion_type, criterion, evidence),
                                  add_generation_prompt=True, return_tensors="pt").to(lm.device)
    ctx = lm.model.disable_adapter() if hasattr(lm.model, "disable_adapter") else nullcontext()
    with ctx:
        res = greedy_cached(lm.model, ids, max_new_tokens, eos_ids(tok, lm.model))
    text = tok.decode(res.tokens, skip_special_tokens=True).strip()
    ok, why = grounding_check(text, label, criterion, evidence)
    if ok:
        return {"text": text, "source": "llm-base-grounded", "check": why}
    return {"text": fallback, "source": "template", "check": f"llm rejected: {why}"}
