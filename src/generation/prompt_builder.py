from src.domain.retrieval_result import RetrievalResult


class PromptBuilder:
    def build(self, question: str, contexts: list[RetrievalResult]) -> str:
        context_text = "\n\n".join(result.text for result in contexts)
        return f"Question: {question}\n\nContext:\n{context_text}"
