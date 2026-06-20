from dataclasses import dataclass, field


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    paper_id: str
    text: str
    section: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)
