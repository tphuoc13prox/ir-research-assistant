from dataclasses import dataclass


@dataclass(slots=True)
class EmbeddingModelConfig:
    name: str
    dimension: int
    normalize: bool = True
