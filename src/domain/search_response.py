from dataclasses import dataclass, field

from src.domain.retrieval_result import RetrievalResult


@dataclass(slots=True)
class SearchResponse:
    query: str
    results: list[RetrievalResult] = field(default_factory=list)
    answer: str | None = None
