#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/eval.sh MODEL_OR_PATH [eval_model.py args]

Examples:
  scripts/eval.sh Qwen/Qwen3-4B-Base --datasets AIME,HMMT
  scripts/eval.sh outputs/eapo_Qwen3-4B-Base_lora32_n2048_seed42_lr1e-05_mcl10240_sop8192_kappa1.38629

Default outputs:
  results/{inferred_name}/metrics.json
  results/{inferred_name}/raw_responses.json
  inferred_name is the last component of the model ID or path. For final or
  checkpoint-* directories, it is the parent directory name. Characters other
  than letters, digits, dots, underscores, and hyphens become underscores.

Environment overrides:
  EVAL_TARGET_KIND       auto (default), lora, or model; use lora for remote adapters
  EVAL_OUTPUT_DIR        Exact output directory; default: results/{inferred_name}
EOF
  exit 2
fi

target=$1
shift

if [[ -z "${target}" ]]; then
  echo "MODEL_OR_PATH must not be empty" >&2
  exit 2
fi

target_kind=${EVAL_TARGET_KIND:-auto}
if [[ "${target_kind}" == auto ]]; then
  if [[ -f "${target%/}/adapter_config.json" ]]; then
    target_kind=lora
  else
    target_kind=model
  fi
fi
case "${target_kind}" in
  lora) target_arg=--lora-path ;;
  model) target_arg=--model-path ;;
  *)
    echo "EVAL_TARGET_KIND must be auto, lora, or model." >&2
    exit 2
    ;;
esac

target_path=${target%/}
if [[ -d "${target_path}" ]]; then
  target_path=$(realpath -- "${target_path}")
fi
output_name=$(basename -- "${target_path}")
if [[ "${output_name}" == checkpoint-* || "${output_name}" == final ]]; then
  parent_name=$(basename -- "$(dirname -- "${target_path}")")
  if [[ "${parent_name}" != "." ]]; then
    output_name=${parent_name}
  fi
fi
output_name=$(printf '%s' "${output_name}" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//')
output_dir=${EVAL_OUTPUT_DIR:-results/${output_name:-model}}

echo "Evaluation target: ${target}"
echo "Target kind: ${target_kind}"
echo "Output directory: ${output_dir}"

python -m src.eval.eval_model "${target_arg}" "${target}" \
  --output-dir "${output_dir}" "$@"
