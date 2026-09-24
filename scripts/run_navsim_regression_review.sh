#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-qwen-drive}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/configs/navsim_regression_review.toml}"

args=(--config "$CONFIG_FILE")
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "${TOP_K:-}" ]]; then
  args+=(--top-k "$TOP_K")
fi

exec conda run --no-capture-output -n "$CONDA_ENV" \
  python "$ROOT/scripts/review_navsim_regressions.py" "${args[@]}"
