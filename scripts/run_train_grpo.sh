#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 && "$1" != -* ]]; then
  set -- --model-name "$@"
fi

set -a
source .env
set +a

NUM_PROCESSES=${NUM_PROCESSES:-1}
NUM_PROMPTS=${NUM_PROMPTS:-2048}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-10240}
SOFT_OVERLONG_START=${SOFT_OVERLONG_START:-8192}
REPORT_TO=${REPORT_TO:-wandb}
RUN_NAME_ARG=()
if [[ -n "${RUN_NAME:-}" ]]; then
  RUN_NAME_ARG=(--run-name "${RUN_NAME}_mcl${MAX_COMPLETION_LENGTH}_sop${SOFT_OVERLONG_START}")
fi

accelerate launch --num_processes "${NUM_PROCESSES}" -m src.train_grpo \
  "${RUN_NAME_ARG[@]}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --soft-overlong-start "${SOFT_OVERLONG_START}" \
  --report-to "${REPORT_TO}" \
  ${LEARNING_RATE:+--learning-rate "${LEARNING_RATE}"} \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
  "$@"
