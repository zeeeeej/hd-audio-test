from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Qwen3-ASR JSONL audio to 16 kHz mono WAV")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_root = args.input.resolve().parent
    prepared: list[dict[str, object]] = []

    for number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        source = Path(str(row["audio"]))
        if not source.is_absolute():
            source = source_root / source
        if not source.is_file():
            raise FileNotFoundError(f"line {number}: missing audio: {source}")
        target = args.audio_dir / f"{len(prepared):06d}.wav"
        completed = subprocess.run(
            [
                args.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(target),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"line {number}: ffmpeg failed: {completed.stderr.strip()}")
        info = sf.info(target)
        if info.frames == 0 or info.samplerate != 16000 or info.channels != 1:
            raise ValueError(f"line {number}: normalized audio is invalid")
        row["audio"] = str(target.resolve())
        prepared.append(row)

    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared), encoding="utf-8"
    )
    print(f"prepared {len(prepared)} samples -> {args.output}")


if __name__ == "__main__":
    main()
