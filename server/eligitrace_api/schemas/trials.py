from pydantic import BaseModel, Field

from server.eligitrace_api.schemas.screening import CriterionIn


class TrialCriteriaOut(BaseModel):
    trial_id: str
    title: str | None
    criteria: list[CriterionIn]
    source: str = Field(description="where the criteria came from")
    raw_text: str | None = None


class ParseCriteriaIn(BaseModel):
    text: str = Field(min_length=3, max_length=20000, description="Pasted eligibility text with Inclusion/Exclusion headings")


class ExampleOut(BaseModel):
    key: str
    patient_id: str
    trial_id: str
    trial_title: str | None
    note: str
    criteria: list[CriterionIn]
