from dataclasses import dataclass, field


@dataclass(slots=True)
class RankingFeatures:
    rrf_score: float = 0.0
    title_score: float = 0.0
    keyword_score: float = 0.0
    abstract_score: float = 0.0
    body_score: float = 0.0
    phrase_score: float = 0.0
    intent_score: float = 0.0
    final_score: float = 0.0


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
    ranking_features: RankingFeatures | None = None
    explanation: dict = field(default_factory=dict)
