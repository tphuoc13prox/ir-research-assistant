from dataclasses import dataclass, field


@dataclass(slots=True)
class Query:
    text: str
    top_k: int = 10
    filters: dict[str, str] = field(default_factory=dict)
    paper_id: str | None = None
    section: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
