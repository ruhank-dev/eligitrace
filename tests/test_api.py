"""G1/G2/G4: API validation, error handling, aggregation, persistence and report export (fake model)."""

import os
import time

os.environ["ELIGITRACE_LOAD_MODEL_ON_STARTUP"] = "false"
os.environ["ELIGITRACE_DB_URL"] = "sqlite:///" + os.path.join(os.path.dirname(__file__), "..", "app_data", "test.sqlite3")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from server.eligitrace_api.main import app  # noqa: E402
from server.eligitrace_api.services import screening_service  # noqa: E402
from server.eligitrace_api.services.aggregation import aggregate  # noqa: E402
from server.eligitrace_api.services.criteria_parser import parse_criteria  # noqa: E402
from server.eligitrace_api.services.model_registry import registry  # noqa: E402
from server.eligitrace_api.services.sentence_splitter import segment  # noqa: E402

NOTE = "A 58-year-old woman presents with chest pain. She has hypertension. She denies smoking. Dr. Lee saw her at 9 a.m."
BODY = {"patient_note": NOTE, "criteria": [{"type": "inclusion", "text": "Age over 50"},
                                           {"type": "exclusion", "text": "Current smoker"}]}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class _Tok:
    def __call__(self, text):
        return {"input_ids": text.split()}


class _LM:
    tokenizer, model_id, dtype, device = _Tok(), "fake/model", "torch.float32", "cpu"


@pytest.fixture
def fake_model(monkeypatch):
    def fake_screen(lm, sentences, ctype, text, **kw):
        label = "included" if ctype == "inclusion" else "not excluded"
        return {"label": label, "parse_ok": True, "evidence_text": [{"id": 0, "text": sentences[0]}],
                "explanation": "because [0]", "explanation_source": "template", "missing_info": False,
                "decoding": kw.get("decoding"), "decoding_stats": {}, "generation_ms": 1.0, "explanation_ms": 1.0}
    monkeypatch.setattr(screening_service, "screen_criterion", fake_screen)
    monkeypatch.setattr(registry, "status", "ready")
    monkeypatch.setattr(registry, "target", _LM())
    monkeypatch.setattr(registry, "draft", None)
    monkeypatch.setattr(registry, "adapter_path", None)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] in {"idle", "loading", "ready", "error"}


def test_model_not_ready_is_503(client, monkeypatch):
    monkeypatch.setattr(registry, "status", "loading")
    r = client.post("/api/screenings", json=BODY)
    assert r.status_code == 503 and r.json()["error"] == "model_not_ready"


@pytest.mark.parametrize("patch,field", [
    ({"patient_note": "too short"}, "patient_note"),
    ({"criteria": []}, "criteria"),
    ({"criteria": [{"type": "maybe", "text": "Age > 18"}]}, "type"),
    ({"criteria": [{"type": "inclusion", "text": "  "}]}, "text"),
    ({"trial_id": "NCT123"}, "trial_id"),
    ({"options": {"gamma": 99}}, "gamma"),
])
def test_validation_422(client, patch, field):
    r = client.post("/api/screenings", json={**BODY, **patch})
    assert r.status_code == 422
    assert field in str(r.json()["detail"])


def test_full_screening_flow_and_reports(client, fake_model):
    r = client.post("/api/screenings", json=BODY)
    assert r.status_code == 202
    sid = r.json()["id"]
    for _ in range(50):
        sc = client.get(f"/api/screenings/{sid}").json()
        if sc["status"] == "done":
            break
        time.sleep(0.05)
    assert sc["status"] == "done" and sc["progress"] == {"done": 2, "total": 2}
    assert sc["overall"]["outcome"] == "Likely eligible"
    assert sc["results"][0]["evidence"][0]["id"] == 0
    html = client.get(f"/api/screenings/{sid}/report.html").text
    assert "not a clinical enrollment decision" in html.lower() and "Provenance" in html
    assert client.get(f"/api/screenings/{sid}/report.md").status_code == 200
    assert client.get(f"/api/screenings/{sid}/report.json").json()["provenance"]["target_model"] == "fake/model"
    assert client.get(f"/api/screenings/{sid}/report.pdf").status_code == 400
    assert any(x["id"] == sid for x in client.get("/api/screenings").json())


def test_unknown_screening_404(client):
    assert client.get("/api/screenings/doesnotexist").status_code == 404


def test_bad_nct_path_422(client):
    assert client.get("/api/trials/12345").status_code == 422


def test_segment_and_numbered_notes(client):
    s, numbered = segment(NOTE)
    assert not numbered and len(s) == 4 and s[3].startswith("Dr. Lee")
    s2, numbered2 = segment("0. First sentence.\n1. Second one.")
    assert numbered2 and s2 == {0: "First sentence.", 1: "Second one."}
    r = client.post("/api/notes/segment", json={"text": NOTE})
    assert r.status_code == 200 and len(r.json()["sentences"]) == 4


def test_criteria_parser():
    c = parse_criteria("Inclusion Criteria:\n- Age >= 18\n- Signed consent\n\nExclusion Criteria:\n* Pregnancy\n* Severe renal\n  impairment")
    assert [x["type"] for x in c] == ["inclusion", "inclusion", "exclusion", "exclusion"]
    assert c[3]["text"] == "Severe renal impairment"


def _r(t, lab, i=0):
    return {"position": i, "criterion_type": t, "criterion_text": "x", "label": lab}


@pytest.mark.parametrize("rows,outcome", [
    ([_r("inclusion", "included"), _r("exclusion", "not excluded")], "Likely eligible"),
    ([_r("inclusion", "included"), _r("exclusion", "excluded")], "Likely ineligible"),
    ([_r("inclusion", "not included"), _r("exclusion", "not enough information")], "Likely ineligible"),
    ([_r("inclusion", "not enough information"), _r("exclusion", "not excluded")], "Needs more information"),
    ([_r("inclusion", "not applicable"), _r("exclusion", "not applicable")], "Likely eligible"),
    ([_r("inclusion", "invalid")], "Needs more information"),
])
def test_aggregation_rule(rows, outcome):
    assert aggregate(rows)["outcome"] == outcome
