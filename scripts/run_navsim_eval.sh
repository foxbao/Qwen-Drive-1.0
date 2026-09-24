#!/usr/bin/env bash
set -euo pipefail

# Evaluate a predictions JSONL file using a TOML config. Environment variables
# below are convenient one-off overrides and are translated to CLI arguments.
#
# Examples:
#   bash scripts/run_navsim_eval.sh
#   PREDICTIONS=outputs/navsim-val-direct/predictions.jsonl \
#     OUTPUT_DIR=outputs/navsim-val-direct bash scripts/run_navsim_eval.sh
#   CONFIG_FILE=configs/my_navsim_eval.toml bash scripts/run_navsim_eval.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_navsim_eval.sh

Configuration:
  CONFIG_FILE=...       TOML file (default: configs/navsim_eval.toml)

One-off overrides:
  PREDICTIONS=...       predictions JSONL file
  OUTPUT_DIR=...        directory for navsim_metrics.json
  METRIC_CACHE=...      optional official NAVSIM metric cache
  CONDA_ENV=qwen-drive  conda environment
EOF
  exit 0
fi

CONDA_ENV="${CONDA_ENV:-qwen-drive}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/configs/navsim_eval.toml}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Evaluation config does not exist: $CONFIG_FILE" >&2
  exit 1
fi

args=(--config "$CONFIG_FILE")

append_if_set() {
  local variable="$1"
  local option="$2"
  if [[ -n "${!variable+x}" ]]; then
    args+=("$option" "${!variable}")
  fi
}

append_if_set PREDICTIONS --predictions
append_if_set METRIC_CACHE --metric-cache
append_if_set OUTPUT_DIR --output

echo "Running NAVSIM evaluation"
echo "Environment: $CONDA_ENV"
echo "Config: $CONFIG_FILE"

exec conda run --no-capture-output -n "$CONDA_ENV" \
  python "$ROOT/scripts/eval_navsim.py" "${args[@]}"
