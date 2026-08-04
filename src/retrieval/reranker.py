from abc import ABC, abstractmethod
from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult


class BaseReRanker(ABC):
    @abstractmethod
    def rerank(self, query: Query, results: list[RetrievalResult]) -> list[RetrievalResult]:
        pass


class IdentityReRanker(BaseReRanker):
    def rerank(self, query: Query, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return results


class CrossEncoderReRanker(BaseReRanker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def rerank(self, query: Query, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return []
        
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                print(f"Warning: Could not load CrossEncoder model {self.model_name} ({exc}). Falling back to Identity ranking.")
                return results

        pairs = [(query.text, r.text) for r in results]
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            print(f"Warning: Re-ranking prediction failed ({exc}).")
            return results

        reranked = []
        for r, score in zip(results, scores):
            r.score = float(score)
            reranked.append(r)
            
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked


class Reranker(IdentityReRanker):
    """Legacy Reranker class maintaining 100% backward compatibility."""
    pass
