"""Request/response shapes for screenings (constraints double as validation + OpenAPI docs)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

CriterionType = Literal["inclusion", "exclusion"]
Label = Literal["included", "not included", "excluded", "not excluded", "not enough information",
                "not applicable", "invalid"]


class CriterionIn(BaseModel):
    type: CriterionType = Field(description="inclusion or exclusion")
    text: str = Field(min_length=3, max_length=2000, description="Criterion text exactly as written in the protocol")

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 3:
            raise ValueError("criterion text is too short")
        return v


class ScreeningOptions(BaseModel):
    decoding: Literal["kv_cached", "speculative"] = "kv_cached"
    gamma: int = Field(4, ge=1, le=8, description="draft tokens per round (speculative only)")
    explanation: Literal["llm", "template"] = "llm"


class ScreeningIn(BaseModel):
    patient_note: str = Field(min_length=20, max_length=12000,
                              description="Free-text clinical note; numbered '0. ...' lines are kept as-is")
    criteria: list[CriterionIn] = Field(min_length=1, max_length=40)
    trial_title: str | None = Field(None, max_length=300)
    trial_id: str | None = Field(None, pattern=r"^NCT\d{8}$")
    patient_label: str | None = Field(None, max_length=120)
    options: ScreeningOptions = ScreeningOptions()


class EvidenceOut(BaseModel):
    id: int
    text: str


class CriterionResultOut(BaseModel):
    position: int
    criterion_type: CriterionType
    criterion_text: str
    status: Literal["pending", "done", "error"]
    label: Label | None = None
    evidence: list[EvidenceOut] = []
    explanation: str | None = None
    explanation_source: str | None = None
    missing_info: bool = False
    parse_ok: bool | None = None
    unsupported_ids: list[int] = []
    decoding: str | None = None
    decoding_stats: dict = {}
    generation_ms: float | None = None
    explanation_ms: float | None = None
    error: str | None = None


class OverallOut(BaseModel):
    outcome: Literal["Likely eligible", "Likely ineligible", "Needs more information", "Incomplete"]
    rule: str
    trace: list[str]
    counts: dict[str, int]


class ScreeningOut(BaseModel):
    id: str
    created_at: datetime
    status: Literal["queued", "running", "done", "error"]
    error: str | None = None
    progress: dict
    trial_title: str | None
    trial_id: str | None
    patient_label: str | None
    sentences: list[EvidenceOut]
    options: ScreeningOptions
    results: list[CriterionResultOut]
    overall: OverallOut | None = None
    provenance: dict | None = None
    disclaimer: str


class ScreeningSummary(BaseModel):
    id: str
    created_at: datetime
    status: str
    trial_title: str | None
    patient_label: str | None
    n_criteria: int
    outcome: str | None
