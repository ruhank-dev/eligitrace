"""Repository: inference-lab benchmark runs (SQLAlchemy ORM only)."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.eligitrace_api.orm.models import LabRun


def add(db: Session, kind: str, params: dict, result: dict) -> LabRun:
    run = LabRun(kind=kind, params=params, result=result)
    db.add(run)
    db.commit()
    return run


def list_recent(db: Session, kind: str | None = None, limit: int = 20) -> list[LabRun]:
    q = select(LabRun).order_by(LabRun.created_at.desc()).limit(limit)
    if kind:
        q = q.where(LabRun.kind == kind)
    return list(db.scalars(q))
