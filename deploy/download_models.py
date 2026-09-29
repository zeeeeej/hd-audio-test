from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one voice model to a fixed local directory")
    parser.add_argument("model")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source", choices=("modelscope", "huggingface"), default="modelscope")
    args = parser.parse_args()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    if args.source == "modelscope":
        from modelscope import snapshot_download

        snapshot_download(args.model, local_dir=str(args.destination))
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=args.model, local_dir=str(args.destination))


if __name__ == "__main__":
    main()
