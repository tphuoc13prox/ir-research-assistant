from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievalResult:
    chunk_id: str
    paper_id: str
    score: float
    text: str
    page: int | None = None
    section: str | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fusion_score: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)
