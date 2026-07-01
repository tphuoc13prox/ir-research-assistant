from __future__ import annotations
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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


class LocalFinetunedLLMClient(LLMClient):
    def __init__(self, model_dir: Path, base_model_name: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        # Load LoRA adapter if it exists
        if (model_dir / "adapter_config.json").exists():
            self.model = PeftModel.from_pretrained(base_model, model_dir)
        else:
            self.model = base_model
            
        self.model.eval()

    def complete(self, prompt: str) -> str:
        # Prompt builder builds: "Question: {question}\n\nContext:\n{context_text}"
        # We wrap this in a ChatML template compatible with Qwen Instruct models
        formatted_prompt = (
            f"<|im_start|>system\nYou are a helpful research assistant. Answer the question using the provided context.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.tokenizer.encode("<|im_end|>") or self.tokenizer.eos_token_id,
            )
            
        # Decode only the generated tokens (exclude input prompt)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response
