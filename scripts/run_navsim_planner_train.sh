#!/usr/bin/env bash
set -euo pipefail

# Manual launcher for the NAVSIM Planning Expert trainer. Most settings live in
# configs/navsim_planner_train.toml; environment variables below are convenient
# one-off command-line overrides and are translated to trainer arguments.
#
# Examples:
#   bash scripts/run_navsim_planner_train.sh
#   LIMIT=100 VAL_LIMIT=100 bash scripts/run_navsim_planner_train.sh
#   CONFIG_FILE=configs/my_experiment.toml bash scripts/run_navsim_planner_train.sh
#   RESUME=outputs/planner-navsim-epoch1 bash scripts/run_navsim_planner_train.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_navsim_planner_train.sh

Configuration:
  CONFIG_FILE=...       TOML file (default: configs/navsim_planner_train.toml)

One-off overrides:
  LIMIT=100             train on 100 scenes
  VAL_LIMIT=100         validate on 100 scenes
  EPOCHS=1              number of epochs
  GRAD_ACCUM_STEPS=4    effective batch size = 1 * 4
  CONDITIONING_MODE=reasoning   train with the frozen VLM's generated rationale cache
  MAX_REASONING_TOKENS=96      cap the generated rationale length
  OUTPUT_DIR=...        planner checkpoint directory
  LORA_ADAPTER=...      frozen PEFT VLM adapter for Stage 2 training
  RESUME=...             resume from a trainer checkpoint directory
  DEVICE=cuda:0         CUDA device
  DTYPE=bfloat16        model dtype
  ATTN_IMPLEMENTATION=sdpa  attention backend

Examples:
  LIMIT=100 VAL_LIMIT=100 bash scripts/run_navsim_planner_train.sh
  bash scripts/run_navsim_planner_train.sh
  RESUME=outputs/planner-navsim-epoch1 bash scripts/run_navsim_planner_train.sh
EOF
  exit 0
fi

CONDA_ENV="${CONDA_ENV:-qwen-drive}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/configs/navsim_planner_train.toml}"

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

# These overrides are deliberately optional: unset variables leave the TOML
# values untouched. Set LIMIT/VAL_LIMIT only for a pilot or debug run.
append_if_set MODEL_DIR --model
append_if_set PLANNER_DIR --planner
append_if_set LORA_ADAPTER --lora-adapter
append_if_set TRAIN_SCENES --scenes
append_if_set VAL_SCENES --val-scenes
append_if_set IMAGE_ROOT --image-root
append_if_set IMAGE_ARCHIVE --image-archive
append_if_set VAL_IMAGE_ROOT --val-image-root
append_if_set VAL_IMAGE_ARCHIVE --val-image-archive
append_if_set OUTPUT_DIR --output
append_if_set EPOCHS --epochs
append_if_set LIMIT --limit
append_if_set VAL_LIMIT --val-limit
append_if_set GRAD_ACCUM_STEPS --gradient-accumulation-steps
append_if_set LEARNING_RATE --learning-rate
append_if_set WEIGHT_DECAY --weight-decay
append_if_set WARMUP_STEPS --warmup-steps
append_if_set MAX_GRAD_NORM --max-grad-norm
append_if_set SEED --seed
append_if_set DEVICE --device
append_if_set DTYPE --dtype
append_if_set ATTN_IMPLEMENTATION --attn-implementation
append_if_set CONDITIONING_MODE --conditioning-mode
append_if_set MAX_REASONING_TOKENS --max-reasoning-tokens
append_if_set RESUME --resume

echo "Running NAVSIM Planning Expert training"
echo "Environment: $CONDA_ENV"
echo "Config: $CONFIG_FILE"
echo "Command-line overrides: ${#args[*]} argument(s)"

exec conda run --no-capture-output -n "$CONDA_ENV" \
  python "$ROOT/scripts/train_planner.py" "${args[@]}"
