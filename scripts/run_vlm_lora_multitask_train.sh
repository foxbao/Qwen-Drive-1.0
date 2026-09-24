#!/usr/bin/env bash
set -euo pipefail

# Config-first launcher for the shared VLM LoRA / NAVSIM + VQA trainer.
#
# Examples:
#   bash scripts/run_vlm_lora_multitask_train.sh
#   MAX_STEPS=3 NAVSIM_TRAIN_LIMIT=4 NAVSIM_VAL_LIMIT=2 \
#     VAL_SAMPLES_PER_TASK=1 TASK_SEQUENCE=navsim,drivelm,aokvqa \
#     NO_SAVE_CHECKPOINT=1 bash scripts/run_vlm_lora_multitask_train.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_vlm_lora_multitask_train.sh

Configuration:
  CONFIG_FILE=...             TOML file (default: configs/vlm_lora_multitask.toml)

One-off overrides:
  MAX_STEPS=3                 optimizer updates; use 3 for the first smoke test
  NAVSIM_TRAIN_LIMIT=100      cap NAVSIM training scenes for a pilot
  NAVSIM_VAL_LIMIT=100        cap NAVSIM validation scenes
  VAL_SAMPLES_PER_TASK=1      validation examples per task
  TASK_SEQUENCE=navsim,drivelm,aokvqa  deterministic debug task order
  OUTPUT_DIR=...              checkpoint/log directory
  NO_SAVE_CHECKPOINT=1        don't write the large planner checkpoint (smoke tests)
  CONDA_ENV=qwen-drive        conda environment

First run (real 3-task forward/backward smoke test, no checkpoint saved):
  MAX_STEPS=3 NAVSIM_TRAIN_LIMIT=4 NAVSIM_VAL_LIMIT=2 \
    VAL_SAMPLES_PER_TASK=1 TASK_SEQUENCE=navsim,drivelm,aokvqa \
    NO_SAVE_CHECKPOINT=1 OUTPUT_DIR=outputs/vlm-lora-smoke \
    bash scripts/run_vlm_lora_multitask_train.sh
EOF
  exit 0
fi

CONDA_ENV="${CONDA_ENV:-qwen-drive}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/configs/vlm_lora_multitask.toml}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Training config does not exist: $CONFIG_FILE" >&2
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

append_if_set MAX_STEPS --max-steps
append_if_set OUTPUT_DIR --output
append_if_set NAVSIM_TRAIN_LIMIT --navsim-train-limit
append_if_set NAVSIM_VAL_LIMIT --navsim-val-limit
append_if_set VAL_SAMPLES_PER_TASK --val-samples-per-task
append_if_set TASK_SEQUENCE --task-sequence
if [[ "${NO_SAVE_CHECKPOINT:-0}" == "1" ]]; then
  args+=(--no-save-checkpoint)
fi

echo "Running shared VLM LoRA multitask training"
echo "Environment: $CONDA_ENV"
echo "Config: $CONFIG_FILE"
echo "Command-line overrides: ${#args[*]} argument(s)"

exec conda run --no-capture-output -n "$CONDA_ENV" \
  python "$ROOT/scripts/train_vlm_lora_multitask.py" "${args[@]}"
