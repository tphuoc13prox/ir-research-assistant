import json
import math
from pathlib import Path

from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import IdentityReRanker
from src.indexing.faiss_index import FaissIndex
from src.evaluation.retrieval_metrics import recall_at_k
from src.evaluation.ranking_metrics import mean_reciprocal_rank
from src.evaluation.dataset_builder import DatasetBuilder


def ndcg_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    dcg = 0.0
    for i, item_id in enumerate(retrieved_k, start=1):
        if item_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
            
    idcg = 0.0
    for i in range(1, min(len(relevant_ids), k) + 1):
        idcg += 1.0 / math.log2(i + 1)
        
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


class FakeEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "dense" in text.lower():
                vectors.append([1.0, 0.0])
            elif "neural" in text.lower():
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.7, 0.7])
        return vectors


class BenchmarkRunner:
    def __init__(self, dataset_path: Path = Path("data/evaluation/dataset.json")) -> None:
        self.dataset_path = dataset_path

    def run(self) -> dict[str, dict[str, float]]:
        if not self.dataset_path.exists():
            DatasetBuilder(self.dataset_path).build()

        dataset = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        queries = dataset["queries"]
        corpus = dataset["corpus"]

        index = FaissIndex(dimension=2)
        embedder = FakeEmbedder()
        
        chunks = {}
        for c in corpus:
            chunks[c["chunk_id"]] = RetrievalResult(
                chunk_id=c["chunk_id"],
                paper_id=c["paper_id"],
                score=0.0,
                text=c["text"],
                metadata=c["metadata"],
            )

        texts = [c["text"] for c in corpus]
        vectors = embedder.encode(texts)
        index.add(vectors, [c["chunk_id"] for c in corpus])

        dense_retriever = DenseRetriever(embedder, index, chunks)
        sparse_retriever = BM25Retriever(chunks)
        hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
        reranker = IdentityReRanker()

        configurations = {
            "Dense": dense_retriever,
            "Sparse": sparse_retriever,
            "Hybrid": hybrid_retriever,
        }

        results = {}
        for name, retriever in configurations.items():
            tot_recall_5 = 0.0
            tot_recall_10 = 0.0
            tot_mrr = 0.0
            tot_ndcg_5 = 0.0
            tot_ndcg_10 = 0.0

            for q in queries:
                query_obj = Query(text=q["text"], top_k=10)
                retrieved = retriever.retrieve(query_obj)
                if name == "Hybrid":
                    retrieved = reranker.rerank(query_obj, retrieved)
                    
                retrieved_ids = [r.chunk_id for r in retrieved]
                relevant_ids = set(q["relevant_chunk_ids"])

                tot_recall_5 += recall_at_k(relevant_ids, retrieved_ids, 5)
                tot_recall_10 += recall_at_k(relevant_ids, retrieved_ids, 10)
                tot_mrr += mean_reciprocal_rank(relevant_ids, retrieved_ids)
                tot_ndcg_5 += ndcg_at_k(relevant_ids, retrieved_ids, 5)
                tot_ndcg_10 += ndcg_at_k(relevant_ids, retrieved_ids, 10)

            num_queries = len(queries)
            results[name] = {
                "Recall@5": round(tot_recall_5 / num_queries, 4),
                "Recall@10": round(tot_recall_10 / num_queries, 4),
                "MRR": round(tot_mrr / num_queries, 4),
                "nDCG@5": round(tot_ndcg_5 / num_queries, 4),
                "nDCG@10": round(tot_ndcg_10 / num_queries, 4),
            }

        return results
