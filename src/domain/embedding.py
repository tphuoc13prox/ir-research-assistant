from dataclasses import dataclass


@dataclass(slots=True)
class Embedding:
    item_id: str
    vector: list[float]
    model_name: str
