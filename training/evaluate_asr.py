from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, char_left in enumerate(left, 1):
        current = [index]
        for other, char_right in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[other] + 1, previous[other - 1] + (char_left != char_right))
            )
        previous = current
    return previous[-1]


def clean_target(value: str) -> str:
    return value.split("<asr_text>", 1)[-1].replace(" ", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-ASR CER and product-name recall")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B-hf")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="sdpa", low_cpu_mem_usage=True
    ).to("cuda:0")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    total_edits = total_chars = products = products_hit = 0
    samples = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kwargs = {"audio": row["audio"], "language": "Chinese"}
        if row.get("prompt"):
            kwargs["prompt"] = row["prompt"]
        inputs = processor.apply_transcription_request(**kwargs).to("cuda:0", torch.float16)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1] :]
        predicted = processor.decode(generated, return_format="transcription_only")[0].strip()
        expected = clean_target(row["text"])
        normalized_prediction = predicted.replace(" ", "")
        total_edits += levenshtein(expected, normalized_prediction)
        total_chars += max(1, len(expected))
        product = str(row.get("product", "")).strip()
        if product:
            products += 1
            products_hit += int(product in predicted)
        samples.append({"audio": row["audio"], "expected": expected, "predicted": predicted})
    report = {
        "samples": len(samples),
        "cer": total_edits / total_chars if total_chars else None,
        "product_recall": products_hit / products if products else None,
        "product_samples": products,
        "details": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
