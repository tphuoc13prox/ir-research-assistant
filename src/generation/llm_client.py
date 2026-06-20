from __future__ import annotations


class LLMClient:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class ExtractiveLLMClient(LLMClient):
    """Small local fallback that answers from retrieved context without an API key."""

    def complete(self, prompt: str) -> str:
        context_marker = "Context:\n"
        if context_marker not in prompt:
            return "I could not find enough retrieved context to answer."

        context = prompt.split(context_marker, maxsplit=1)[1].strip()
        if not context:
            return "I could not find relevant paper passages for that question yet."

        passages = [part.strip() for part in context.split("\n\n") if part.strip()]
        selected = passages[:3]
        answer = " ".join(selected)
        if len(answer) > 1200:
            answer = answer[:1200].rsplit(" ", maxsplit=1)[0] + "..."
        return answer
