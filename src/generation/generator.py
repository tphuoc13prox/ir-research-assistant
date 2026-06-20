from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder


class Generator:
    def __init__(self, llm_client: LLMClient, prompt_builder: PromptBuilder) -> None:
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder

    def generate(self, question: str, contexts) -> str:
        prompt = self.prompt_builder.build(question, contexts)
        return self.llm_client.complete(prompt)
