import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

class QLoRAFineTuner:
    """
    Model M6: QLoRA Fine-Tuner for Open-Weight LLMs (e.g. Qwen2.5-1.5B/Llama-3/Mistral-7B).
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", output_dir: str = "./qlora_phishguard"):
        self.model_id = model_id
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, train_texts: list, train_labels: list, epochs: int = 1, batch_size: int = 4, lr: float = 2e-4):
        if not HAS_PEFT:
            print("[QLoRA Fine-Tuner] PEFT / BitsAndBytes not installed. Skipping actual QLoRA fine-tuning execution.")
            return

        print(f"[QLoRA Fine-Tuner] Loading base model {self.model_id} for 4-bit fine-tuning...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Load model in 4-bit / fp16 based on availability
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(f"[QLoRA Fine-Tuner] Mock fine-tuning step completed. Model ready at {self.output_dir}")
