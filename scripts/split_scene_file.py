#!/usr/bin/env python3
"""Split a Qwen-Drive scene JSONL by scene token without scene leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def belongs_to_val(scene_token: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{scene_token}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < fraction


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be between 0 and 1")
    groups: dict[str, list[str]] = {}
    with args.input.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            token = str(record.get("meta_info", {}).get("scene_token", ""))
            if not token:
                raise ValueError("scene record is missing meta_info.scene_token")
            groups.setdefault(token, []).append(line)

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with args.train_output.open("w") as train, args.val_output.open("w") as val:
        for token, lines in groups.items():
            target = val if belongs_to_val(token, args.seed, args.val_fraction) else train
            split = "val" if target is val else "train"
            for line in lines:
                target.write(line)
            counts[split] += len(lines)
            counts[f"{split}_scenes"] += 1

    report = {
        "input": str(args.input.resolve()),
        "train_output": str(args.train_output.resolve()),
        "val_output": str(args.val_output.resolve()),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "unique_scene_tokens": len(groups),
        **counts,
    }
    report_path = args.train_output.with_suffix(".split.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
