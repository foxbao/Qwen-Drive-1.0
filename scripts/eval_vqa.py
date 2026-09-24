#!/usr/bin/env python
# Copyright 2026 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Evaluate Qwen-Drive on held-out DriveLM and A-OKVQA questions.

Each model is evaluated independently and writes a JSONL prediction file plus a JSON
summary.  Existing prediction rows are reused, so an interrupted run can be resumed.
The DriveLM score is reported as normalized exact match and token F1; A-OKVQA also
reports option-letter accuracy because its canonical metric is multiple choice.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_drive import QwenDriveForPlanning
from qwen_drive.multitask_data import AOKVQAData, DriveLMData, VQASample

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
OPTION_AT_START_RE = re.compile(r"^\s*(?:option\s*)?([A-Z])(?:[\).:\s]|$)", re.IGNORECASE)
OPTION_NAMED_RE = re.compile(r"\b(?:option|answer)\s*(?:is|:)?\s*([A-Z])\b", re.IGNORECASE)
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize(prediction).split()
    ref = normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def option_letter(text: str) -> str | None:
    text = text.strip()
    match = OPTION_AT_START_RE.search(text)
    if match:
        return match.group(1).upper()
    match = OPTION_NAMED_RE.search(text)
    return match.group(1).upper() if match else None


def sample_option_letter(sample: VQASample) -> str | None:
    return option_letter(sample.answer)


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if "evaluation" not in payload or "models" not in payload:
        raise ValueError("VQA config must contain [evaluation] and [[models]]")
    return payload


def iter_samples(cfg: dict, dataset: str) -> list[VQASample]:
    data = cfg["data"]
    if dataset == "drivelm":
        adapter = DriveLMData(
            resolve_path(data["drivelm_annotations"]),
            split="validation",
            validation_fraction=float(data.get("drivelm_validation_fraction", 0.05)),
        )
    elif dataset == "aokvqa":
        adapter = AOKVQAData(
            resolve_path(data["aok_val_annotations"]),
            resolve_path(data["aok_val_images"]),
            split="validation",
        )
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return [adapter.sample_by_index(index) for index in range(len(adapter))]


def prediction_paths(output_dir: Path, dataset: str) -> tuple[Path, Path]:
    return output_dir / f"{dataset}_predictions.jsonl", output_dir / f"{dataset}_metrics.json"


def load_existing(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.is_file():
        return rows
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["sample_id"]] = row
    return rows


def score_rows(rows: Iterable[dict], dataset: str) -> dict:
    rows = list(rows)
    if not rows:
        return {"dataset": dataset, "count": 0}
    exact = [float(row["normalized_prediction"] == row["normalized_answer"]) for row in rows]
    f1 = [float(row["token_f1"]) for row in rows]
    result = {
        "dataset": dataset,
        "count": len(rows),
        "normalized_exact_match": sum(exact) / len(exact),
        "token_f1": sum(f1) / len(f1),
    }
    if dataset == "aokvqa":
        valid = [row for row in rows if row.get("prediction_option") is not None]
        result["option_accuracy"] = sum(
            row["prediction_option"] == row["answer_option"] for row in valid
        ) / len(valid) if valid else 0.0
        result["option_parse_rate"] = len(valid) / len(rows)
    return result


def evaluate_model(cfg: dict, model_cfg: dict, datasets: list[str], args: argparse.Namespace) -> None:
    name = str(model_cfg["name"])
    output_dir = resolve_path(args.output or model_cfg.get("output", f"outputs/vqa-{name}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype_name = str(cfg["evaluation"].get("dtype", "bfloat16"))
    device = str(cfg["evaluation"].get("device", "cuda:0"))
    if dtype_name not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype_name!r}")
    print(f"loading model {name}: {resolve_path(model_cfg['model'])}", flush=True)
    model = QwenDriveForPlanning.from_pretrained(
        resolve_path(model_cfg["model"]),
        lora_adapter=resolve_path(model_cfg["lora_adapter"]) if model_cfg.get("lora_adapter") else None,
        dtype=DTYPES[dtype_name],
        attn_implementation=str(cfg["evaluation"].get("attn_implementation", "sdpa")),
    ).to(device).eval()
    evaluation = cfg["evaluation"]
    max_tokens = int(evaluation.get("max_new_tokens", 64))
    image_budget = evaluation.get("vqa_image_pixels")
    image_budget = int(image_budget) if image_budget is not None else None
    for dataset in datasets:
        samples = iter_samples(cfg, dataset)
        if args.limit is not None:
            samples = samples[: args.limit]
        predictions_path, metrics_path = prediction_paths(output_dir, dataset)
        existing = load_existing(predictions_path) if args.resume else {}
        pending = [sample for sample in samples if sample.sample_id not in existing]
        print(f"{name} {dataset}: {len(samples)} samples, {len(pending)} pending", flush=True)
        with predictions_path.open("a") as handle:
            for sample in tqdm(pending, desc=f"{name} {dataset}", unit="sample"):
                with torch.inference_mode():
                    output = model.generate_text(
                        sample.images,
                        sample.question,
                        max_new_tokens=max_tokens,
                        temperature=float(evaluation.get("temperature", 0.01)),
                        top_k=int(evaluation.get("top_k", 1)),
                        top_p=float(evaluation.get("top_p", 0.001)),
                        repetition_penalty=float(evaluation.get("repetition_penalty", 1.0)),
                        presence_penalty=float(evaluation.get("presence_penalty", 0.0)),
                        image_pixel_budget=image_budget,
                        seed=int(evaluation.get("seed", 3407)),
                    )
                prediction = output.text.strip()
                row = {
                    "sample_id": sample.sample_id,
                    "dataset": dataset,
                    "prediction": prediction,
                    "answer": sample.answer,
                    "normalized_prediction": normalize(prediction),
                    "normalized_answer": normalize(sample.answer),
                    "token_f1": token_f1(prediction, sample.answer),
                }
                if dataset == "aokvqa":
                    row["prediction_option"] = option_letter(prediction)
                    row["answer_option"] = sample_option_letter(sample)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                existing[sample.sample_id] = row
        metrics = score_rows(existing.values(), dataset)
        metrics["model"] = name
        metrics["checkpoint"] = str(model_cfg.get("lora_adapter") or model_cfg["model"])
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, required=True)
    boot, _ = bootstrap.parse_known_args()
    cfg = load_config(boot.config.expanduser().resolve())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default=None, help="model name from [[models]], default all")
    parser.add_argument("--dataset", choices=("drivelm", "aokvqa"), action="append")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    datasets = args.dataset or list(cfg["evaluation"].get("datasets", ["aokvqa", "drivelm"]))
    models = cfg["models"]
    if args.model:
        models = [model for model in models if model.get("name") == args.model]
        if not models:
            parser.error(f"unknown model {args.model!r}")
    args.resume = not args.no_resume
    for model_cfg in models:
        evaluate_model(cfg, model_cfg, datasets, args)


if __name__ == "__main__":
    main()
