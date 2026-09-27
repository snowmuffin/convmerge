"""DPO on train/dpo.jsonl written by recipe_dpo.yaml (TRL >= 1.0)."""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

data = load_dataset(
    "json", data_files={"train": "train/dpo.jsonl", "validation": "train/dpo.val.jsonl"}
)
trainer = DPOTrainer(
    model=AutoModelForCausalLM.from_pretrained(MODEL),
    processing_class=AutoTokenizer.from_pretrained(MODEL),
    args=DPOConfig(output_dir="out/dpo", max_length=2048, eval_strategy="steps", eval_steps=200),
    train_dataset=data["train"],  # "prompt" / "chosen" / "rejected" message lists
    eval_dataset=data["validation"],
)
trainer.train()
