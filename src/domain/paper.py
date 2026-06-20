from dataclasses import dataclass, field


@dataclass(slots=True)
class Paper:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    source_url: str | None = None
    published_at: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
