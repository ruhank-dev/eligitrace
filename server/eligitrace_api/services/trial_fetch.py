"""Fetch eligibility criteria for an NCT ID from the public ClinicalTrials.gov v2 API (app support only)."""

import httpx

from server.eligitrace_api.config import get_settings
from server.eligitrace_api.services.criteria_parser import parse_criteria


class TrialFetchError(RuntimeError):
    pass


def fetch_trial(nct_id: str) -> dict:
    url = f"{get_settings().clinicaltrials_base_url}/{nct_id}"  # fixed allowlisted host, validated ID
    try:
        r = httpx.get(url, params={"fields": "protocolSection.identificationModule,"
                                             "protocolSection.eligibilityModule"},
                      timeout=10.0, follow_redirects=False)
    except httpx.HTTPError as e:
        raise TrialFetchError(f"ClinicalTrials.gov unreachable ({type(e).__name__}). Paste the criteria instead.") from e
    if r.status_code == 404:
        raise TrialFetchError(f"{nct_id} was not found on ClinicalTrials.gov.")
    if r.status_code != 200:
        raise TrialFetchError(f"ClinicalTrials.gov returned HTTP {r.status_code}.")
    ps = r.json().get("protocolSection", {})
    text = ps.get("eligibilityModule", {}).get("eligibilityCriteria", "") or ""
    crit = parse_criteria(text)
    if not crit:
        raise TrialFetchError(f"{nct_id} has no parseable eligibility criteria.")
    return {"trial_id": nct_id, "title": ps.get("identificationModule", {}).get("briefTitle"),
            "criteria": crit[:40], "source": "ClinicalTrials.gov API v2", "raw_text": text[:20000]}
