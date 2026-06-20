from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievalResult:
    chunk_id: str
    paper_id: str
    score: float
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
