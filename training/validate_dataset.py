from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import soundfile as sf

TARGET_PATTERN = re.compile(r"^language [A-Za-z]+<asr_text>.+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Qwen3-ASR fine-tuning manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    return parser.parse_args()


def validate(path: Path, max_seconds: float) -> dict[str, object]:
    errors: list[str] = []
    durations: list[float] = []
    speakers: Counter[str] = Counter()
    products: Counter[str] = Counter()
    rows = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        audio = Path(str(row.get("audio", "")))
        text = row.get("text")
        if not audio.is_absolute():
            audio = path.parent / audio
        if not audio.is_file():
            errors.append(f"line {line_number}: audio does not exist: {audio}")
        else:
            try:
                info = sf.info(audio)
                duration = info.frames / info.samplerate
                durations.append(duration)
                if info.channels != 1 or info.samplerate != 16000:
                    errors.append(f"line {line_number}: audio must be 16 kHz mono")
                if duration <= 0 or duration > max_seconds:
                    errors.append(f"line {line_number}: duration {duration:.2f}s is out of range")
            except RuntimeError as exc:
                errors.append(f"line {line_number}: unreadable audio: {exc}")
        if not isinstance(text, str) or not TARGET_PATTERN.match(text):
            errors.append(
                f"line {line_number}: text must look like language Chinese<asr_text>转写内容"
            )
        if "prompt" in row and not isinstance(row["prompt"], str):
            errors.append(f"line {line_number}: prompt must be a string")
        if speaker := row.get("speaker_id"):
            speakers[str(speaker)] += 1
        if product := row.get("product"):
            products[str(product)] += 1
    if rows == 0:
        errors.append("manifest contains no samples")
    return {
        "valid": not errors,
        "samples": rows,
        "duration_seconds": round(sum(durations), 3),
        "speakers": len(speakers),
        "products": len(products),
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    report = validate(args.manifest, args.max_seconds)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
