import json
from pathlib import Path


class DatasetBuilder:
    def __init__(self, output_path: Path = Path("data/evaluation/dataset.json")) -> None:
        self.output_path = output_path

    def build(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = {
            "queries": [
                {
                    "query_id": "q1",
                    "text": "dense retrieval",
                    "relevant_chunk_ids": ["c1", "c3"],
                },
                {
                    "query_id": "q2",
                    "text": "neural ranking",
                    "relevant_chunk_ids": ["c2"],
                },
                {
                    "query_id": "q3",
                    "text": "hybrid fusion reciprocal rank",
                    "relevant_chunk_ids": ["c1", "c2"],
                }
            ],
            "corpus": [
                {
                    "chunk_id": "c1",
                    "paper_id": "p1",
                    "text": "Dense retrieval models map queries and passages to low-dimensional vectors using deep neural networks.",
                    "metadata": {"year": "2024", "section": "Introduction"},
                },
                {
                    "chunk_id": "c2",
                    "paper_id": "p2",
                    "text": "Neural ranking algorithms use transformer layers to score relevance between query terms and text bodies.",
                    "metadata": {"year": "2023", "section": "Methodology"},
                },
                {
                    "chunk_id": "c3",
                    "paper_id": "p1",
                    "text": "We evaluate dense encoders on MS MARCO showing improved recall compared to lexical term matching.",
                    "metadata": {"year": "2024", "section": "Evaluation"},
                }
            ]
        }
        self.output_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
