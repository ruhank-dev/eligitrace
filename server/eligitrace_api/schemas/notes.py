from pydantic import BaseModel, Field

from server.eligitrace_api.schemas.screening import EvidenceOut


class SegmentIn(BaseModel):
    text: str = Field(min_length=1, max_length=12000)


class SegmentOut(BaseModel):
    sentences: list[EvidenceOut]
    already_numbered: bool
    n_tokens_estimate: int
