"""Trial criteria helpers: ClinicalTrials.gov fetch, pasted-text parsing, validation examples."""

from fastapi import APIRouter, Path

from server.eligitrace_api.schemas.trials import ExampleOut, ParseCriteriaIn, TrialCriteriaOut
from server.eligitrace_api.services.criteria_parser import parse_criteria
from server.eligitrace_api.services.examples import validation_examples
from server.eligitrace_api.services.trial_fetch import fetch_trial

router = APIRouter(prefix="/api", tags=["trials"])


@router.get("/trials/{nct_id}", response_model=TrialCriteriaOut, summary="Fetch criteria from ClinicalTrials.gov")
def get_trial(nct_id: str = Path(pattern=r"^NCT\d{8}$")) -> dict:
    return fetch_trial(nct_id)


@router.post("/criteria/parse", response_model=TrialCriteriaOut, summary="Split pasted eligibility text")
def parse(body: ParseCriteriaIn) -> dict:
    return {"trial_id": "pasted", "title": None, "criteria": parse_criteria(body.text)[:40], "source": "pasted text"}


@router.get("/examples", response_model=list[ExampleOut], summary="Validation-split demo cases (never test)")
def examples() -> list[dict]:
    return validation_examples()
