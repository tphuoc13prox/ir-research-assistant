import json
import os
import torch
from pathlib import Path
from typing import Callable, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType

def load_training_data(chunks_path: Path) -> list[dict[str, str]]:
    examples = []
    if not chunks_path.exists():
        return examples
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
                text = chunk.get("text", "")
                title = chunk.get("metadata", {}).get("title", "this paper")
                
                if not text:
                    continue
                
                # Format 1: Summarize context
                examples.append({
                    "instruction": f"Using this context from the paper '{title}':\n\n{text}",
                    "input": "Summarize the key information in this passage.",
                    "output": f"This passage from the paper '{title}' states: {text[:250]}..."
                })
                
                # Format 2: Explain findings
                examples.append({
                    "instruction": f"Here is a passage from '{title}':\n\n{text}",
                    "input": "What is the focus of this finding?",
                    "output": f"The focus of this finding is: {text[:300]}..."
                })
            except Exception:
                continue
    # Keep it to top 20 examples for rapid local fine-tuning
    return examples[:20]

def run_finetuning(
    base_model_name: str,
    chunks_path: Path,
    output_dir: Path,
    progress_callback: Callable[[str], None] | None = None
) -> None:
    if progress_callback:
        progress_callback("Initializing fine-tuning configuration...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    if progress_callback:
        progress_callback(f"Loading base model '{base_model_name}' on {device.upper()}...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if progress_callback:
        progress_callback("Preparing training dataset...")

    raw_data = load_training_data(chunks_path)
    if not raw_data:
        raise ValueError("No training data found in chunks.jsonl")

    # Format into SFT datasets
    tokenized_data = []
    for item in raw_data:
        prompt = (
            f"<|im_start|>system\nYou are a helpful research assistant. Answer the user's question using the provided context.<|im_end|>\n"
            f"<|im_start|>user\n{item['instruction']}\n\nQuestion: {item['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_text = prompt + item['output'] + "<|im_end|>"
        encodings = tokenizer(full_text, truncation=True, max_length=512)
        labels = encodings["input_ids"].copy()
        
        tokenized_data.append({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        })

    if progress_callback:
        progress_callback("Applying LoRA configuration...")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=10,  # Fast training run
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        fp16=(device == "cuda"),
        use_cpu=(device == "cpu"),
        report_to="none",
    )

    class ProgressLoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss = logs["loss"]
                step = state.global_step
                max_steps = state.max_steps
                if progress_callback:
                    progress_callback(f"Fine-tuning step {step}/{max_steps} - Loss: {loss:.4f}")

    if progress_callback:
        progress_callback("Starting local fine-tuning (LoRA adapter)...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        callbacks=[ProgressLoggingCallback()],
    )

    trainer.train()

    if progress_callback:
        progress_callback("Fine-tuning completed. Saving LoRA weights...")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if progress_callback:
        progress_callback("Model saved successfully.")

if __name__ == "__main__":
    # Test script execution
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    run_finetuning(
        base_model_name=args.base_model,
        chunks_path=Path(args.chunks),
        output_dir=Path(args.output),
        progress_callback=print
    )
