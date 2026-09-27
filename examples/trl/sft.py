"""Fine-tune on train/sft.jsonl written by recipe_sft.yaml (TRL >= 1.0)."""

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

data = load_dataset(
    "json", data_files={"train": "train/sft.jsonl", "validation": "train/sft.val.jsonl"}
)
trainer = SFTTrainer(
    model=MODEL,
    args=SFTConfig(output_dir="out/sft", max_length=2048, eval_strategy="steps", eval_steps=200),
    train_dataset=data["train"],  # "messages" (+ "tools") columns: TRL applies the chat template
    eval_dataset=data["validation"],
)
trainer.train()
