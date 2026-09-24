#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-qwen-drive}"
MODEL_DIR="${MODEL_DIR:-$ROOT/Qwen-Drive-1.0-4B}"
PLANNER_DIR="${PLANNER_DIR:-$MODEL_DIR/planner-sft}"
SCENES="${SCENES:-$ROOT/data/demo/planning_scenes.jsonl}"
IMAGE_ARCHIVE="${IMAGE_ARCHIVE:-$ROOT/data/demo/frames.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/planner-demo}"
LIMIT="${LIMIT:-1}"
EPOCHS="${EPOCHS:-1}"

for path in "$MODEL_DIR" "$PLANNER_DIR" "$SCENES" "$IMAGE_ARCHIVE"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 1
  fi
done

echo "Running planner demo training in conda environment: $CONDA_ENV"
echo "Model: $MODEL_DIR"
echo "Initial planner: $PLANNER_DIR"
echo "Output: $OUTPUT_DIR"

exec conda run --no-capture-output -n "$CONDA_ENV" \
  python "$ROOT/scripts/train_planner.py" \
  --model "$MODEL_DIR" \
  --planner "$PLANNER_DIR" \
  --scenes "$SCENES" \
  --image-archive "$IMAGE_ARCHIVE" \
  --limit "$LIMIT" \
  --epochs "$EPOCHS" \
  --output "$OUTPUT_DIR" \
  --dtype bfloat16 \
  --attn-implementation sdpa
