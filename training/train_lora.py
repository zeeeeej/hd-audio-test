from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


class ManifestDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class ASRCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        conversations = []
        for row in rows:
            messages = []
            if row.get("prompt"):
                messages.append(
                    {"role": "system", "content": [{"type": "text", "text": row["prompt"]}]}
                )
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": [{"type": "audio", "path": row["audio"]}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": row["text"]}],
                    },
                ]
            )
            conversations.append(messages)
        return self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            processor_kwargs={"output_labels": True},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3-ASR-0.6B")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B-hf")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--grad-acc", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear",
        ),
    )
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        fp16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ManifestDataset(args.train),
        eval_dataset=ManifestDataset(args.eval),
        data_collator=ASRCollator(processor),
    )
    trainer.train()
    trainer.save_model(args.output / "final-adapter")
    processor.save_pretrained(args.output / "final-adapter")


if __name__ == "__main__":
    main()
