"""Screening endpoints (thin: validation by schemas, work by the screening service)."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from sqlalchemy.orm import Session

from server.eligitrace_api.database import get_db
from server.eligitrace_api.repositories import screening_repo
from server.eligitrace_api.schemas.notes import SegmentIn, SegmentOut
from server.eligitrace_api.schemas.screening import ScreeningIn, ScreeningOut, ScreeningSummary
from server.eligitrace_api.services import report_renderer, screening_service
from server.eligitrace_api.services.sentence_splitter import segment

router = APIRouter(prefix="/api", tags=["screening"])


@router.post("/notes/segment", response_model=SegmentOut, summary="Preview numbered sentences of a note")
def segment_note(body: SegmentIn) -> SegmentOut:
    sents, numbered = segment(body.text)
    return SegmentOut(sentences=[{"id": i, "text": t} for i, t in sents.items()], already_numbered=numbered,
                      n_tokens_estimate=int(len(body.text) / 3.8))


@router.post("/screenings", response_model=ScreeningOut, status_code=202,
             summary="Start screening a patient note against trial criteria")
def create_screening(body: ScreeningIn, db: Session = Depends(get_db)) -> dict:
    sid = screening_service.start(body)
    return screening_service.to_out(screening_repo.get(db, sid))


@router.get("/screenings", response_model=list[ScreeningSummary], summary="Recent screenings")
def list_screenings(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)) -> list[dict]:
    return [{"id": s.id, "created_at": s.created_at, "status": s.status, "trial_title": s.trial_title,
             "patient_label": s.patient_label, "n_criteria": s.n_criteria,
             "outcome": (s.overall or {}).get("outcome")} for s in screening_repo.list_recent(db, limit)]


def _get(db: Session, sid: str):
    sc = screening_repo.get(db, sid)
    if not sc:
        raise HTTPException(404, f"Screening {sid} not found")
    return sc


@router.get("/screenings/{sid}", response_model=ScreeningOut, summary="Screening status and results")
def get_screening(sid: str, db: Session = Depends(get_db)) -> dict:
    return screening_service.to_out(_get(db, sid))


@router.delete("/screenings/{sid}", status_code=204, summary="Delete a screening")
def delete_screening(sid: str, db: Session = Depends(get_db)) -> Response:
    screening_repo.delete(db, _get(db, sid))
    return Response(status_code=204)


@router.get("/screenings/{sid}/report.{fmt}", summary="Download the screening report (html | md | json)")
def report(sid: str, fmt: str, db: Session = Depends(get_db)):
    sc = _get(db, sid)
    if sc.status != "done":
        raise HTTPException(409, "The screening is not finished yet")
    out = screening_service.to_out(sc)
    out["created_at"] = str(out["created_at"])
    name = f"eligitrace-report-{sid}"
    if fmt == "html":
        return HTMLResponse(report_renderer.to_html(out), headers={"Content-Disposition": f'inline; filename="{name}.html"'})
    if fmt == "md":
        return PlainTextResponse(report_renderer.to_markdown(out), media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="{name}.md"'})
    if fmt == "json":
        return Response(report_renderer.to_json(out), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{name}.json"'})
    raise HTTPException(400, "format must be html, md or json")
