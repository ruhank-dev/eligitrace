"""ORM tables: a screening (patient note + trial) owns many criterion results; lab runs are separate."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.eligitrace_api.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(UTC)


class Screening(Base):
    __tablename__ = "screenings"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|error
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    trial_title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    trial_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    patient_label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    patient_note: Mapped[str] = mapped_column(Text)
    sentences: Mapped[dict] = mapped_column(JSON)
    options: Mapped[dict] = mapped_column(JSON)
    n_criteria: Mapped[int] = mapped_column(Integer)
    overall: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    provenance: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    results: Mapped[list["CriterionResult"]] = relationship(
        back_populates="screening", cascade="all, delete-orphan", order_by="CriterionResult.position")


class CriterionResult(Base):
    __tablename__ = "criterion_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    screening_id: Mapped[str] = mapped_column(ForeignKey("screenings.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    criterion_type: Mapped[str] = mapped_column(String(10))
    criterion_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|done|error
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    screening: Mapped[Screening] = relationship(back_populates="results")


class LabRun(Base):
    __tablename__ = "lab_runs"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    kind: Mapped[str] = mapped_column(String(16))  # decode|attention
    params: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON)
