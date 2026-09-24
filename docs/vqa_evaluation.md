# VQA Held-out Evaluation

`scripts/eval_vqa.py` evaluates the base VLM and an optional LoRA adapter on the held-out
DriveLM and A-OKVQA splits. Its settings and model list live in
[`configs/vqa_eval.toml`](../configs/vqa_eval.toml), so ordinary experiments do not need a
long command line.

## Run

Run every configured model/dataset pair:

```bash
bash scripts/run_vqa_eval.sh
```

Use a small subset before a long evaluation:

```bash
MODEL=vlm-lora-pilot-200 DATASET=aokvqa LIMIT=20 \
  bash scripts/run_vqa_eval.sh
```

`MODEL` is a `name` under `[[models]]`; `DATASET` is `aokvqa` or `drivelm`. The launcher
also accepts `CONFIG_FILE=...`, `OUTPUT_DIR=...`, `CONDA_ENV=...`, and `NO_RESUME=1`.
Normally leave resume enabled: a completed prediction is appended as one JSON object per
line, and a rerun skips its `sample_id`. This is important for the full DriveLM evaluation,
which is much longer than the A-OKVQA validation evaluation.

## Outputs

For each configured model, the configured output directory contains:

```text
aokvqa_predictions.jsonl
aokvqa_metrics.json
drivelm_predictions.jsonl
drivelm_metrics.json
```

Each prediction row preserves the sample ID, generated answer, reference answer, normalized
strings and token F1. A-OKVQA rows also record the parsed prediction/reference option letters.

## Metrics

- **A-OKVQA `option_accuracy`** is the fraction of validation questions where the first
  generated option letter matches the labeled correct choice. `option_parse_rate` reports
  whether the model followed the required multiple-choice answer format.
- **Normalized exact match** lowercases, strips punctuation and collapses whitespace before
  comparing generated/reference answers.
- **Token F1** is a lexical overlap diagnostic. It is useful for DriveLM's short free-form
  answers but is not a substitute for semantic or human evaluation.

DriveLM does not have a single official multiple-choice accuracy under this local adapter;
therefore its exact-match and token-F1 values should be interpreted as repeatable proxy
metrics, and individual predictions should be inspected before drawing strong reasoning
quality conclusions.

## Current comparison

The default configuration compares `Qwen-Drive-1.0-4B` with the 200-step shared
NAVSIM/DriveLM/A-OKVQA LoRA pilot. Pair those VQA results with the complete NAVSIM validation
comparison in `outputs/navsim-vlm-lora-full-review/` before deciding whether to extend training.
