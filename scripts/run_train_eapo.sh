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
EAPO_KAPPA=${EAPO_KAPPA:-1.3862943611198906}
REPORT_TO=${REPORT_TO:-wandb}
DECAY_STEPS_ARG=()
if [[ -n "${EAPO_KAPPA_DECAY_STEPS:-}" ]]; then
  DECAY_STEPS_ARG=(--eapo-kappa-decay-steps "${EAPO_KAPPA_DECAY_STEPS}")
fi
RUN_NAME_ARG=()
if [[ -n "${RUN_NAME:-}" ]]; then
  RUN_NAME_ARG=(--run-name "${RUN_NAME}_mcl${MAX_COMPLETION_LENGTH}_sop${SOFT_OVERLONG_START}")
fi

accelerate launch --num_processes "${NUM_PROCESSES}" -m src.train_eapo \
  "${RUN_NAME_ARG[@]}" \
  "${DECAY_STEPS_ARG[@]}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --soft-overlong-start "${SOFT_OVERLONG_START}" \
  --eapo-kappa "${EAPO_KAPPA}" \
  --report-to "${REPORT_TO}" \
  ${LEARNING_RATE:+--learning-rate "${LEARNING_RATE}"} \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
  "$@"
