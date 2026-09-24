#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-qwen-drive}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/configs/vqa_eval.toml}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_vqa_eval.sh

Overrides:
  CONFIG_FILE=...   VQA TOML config
  MODEL=...         one model name from the config
  DATASET=...       aokvqa or drivelm
  LIMIT=...         evaluate only the first N samples (smoke test)
  OUTPUT_DIR=...    override the selected model output directory
  NO_RESUME=1       ignore existing JSONL rows
EOF
  exit 0
fi

args=(--config "$CONFIG_FILE")
[[ -n "${MODEL:-}" ]] && args+=(--model "$MODEL")
[[ -n "${DATASET:-}" ]] && args+=(--dataset "$DATASET")
[[ -n "${LIMIT:-}" ]] && args+=(--limit "$LIMIT")
[[ -n "${OUTPUT_DIR:-}" ]] && args+=(--output "$OUTPUT_DIR")
[[ -n "${NO_RESUME:-}" ]] && args+=(--no-resume)

echo "Running VQA evaluation"
echo "Environment: $CONDA_ENV"
echo "Config: $CONFIG_FILE"
exec conda run --no-capture-output -n "$CONDA_ENV" python "$ROOT/scripts/eval_vqa.py" "${args[@]}"
