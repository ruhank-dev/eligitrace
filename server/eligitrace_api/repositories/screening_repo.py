"""Repository: all database access for screenings (SQLAlchemy ORM only)."""

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from server.eligitrace_api.orm.models import CriterionResult, Screening


def create(db: Session, *, note: str, sentences: dict[int, str], criteria: list[dict], options: dict,
           trial_title: str | None, trial_id: str | None, patient_label: str | None) -> Screening:
    sc = Screening(patient_note=note, sentences={str(k): v for k, v in sentences.items()}, options=options,
                   trial_title=trial_title, trial_id=trial_id, patient_label=patient_label,
                   n_criteria=len(criteria))
    sc.results = [CriterionResult(position=i, criterion_type=c["type"], criterion_text=c["text"])
                  for i, c in enumerate(criteria)]
    db.add(sc)
    db.commit()
    return sc


def get(db: Session, screening_id: str) -> Screening | None:
    return db.scalar(select(Screening).options(selectinload(Screening.results)).where(Screening.id == screening_id))


def list_recent(db: Session, limit: int = 20) -> list[Screening]:
    return list(db.scalars(select(Screening).order_by(Screening.created_at.desc()).limit(limit)))


def set_status(db: Session, sc: Screening, status: str, error: str | None = None) -> None:
    sc.status, sc.error = status, error
    db.commit()


def save_result(db: Session, row: CriterionResult, result: dict | None, status: str) -> None:
    row.result, row.status = result, status
    db.commit()


def finish(db: Session, sc: Screening, overall: dict, provenance: dict) -> None:
    sc.overall, sc.provenance, sc.status = overall, provenance, "done"
    db.commit()


def delete(db: Session, sc: Screening) -> None:
    db.delete(sc)
    db.commit()
