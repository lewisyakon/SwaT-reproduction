#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly PROJECT_ROOT="/home/lewis/Subproblem4UnifiedReproduction"
readonly SWAT_ROOT="/mnt/hd2/lewis/SwaTReproduction"
readonly PYTHON="${SWAT_ROOT}/env3.10/bin/python"
readonly EVALUATOR="${PROJECT_ROOT}/scripts/evaluate_swat_unified_no_am.py"
readonly OUTPUT_ROOT="${SWAT_ROOT}/outputs/subproblem4_original_recipe/evaluations/swat_am_two_budget_v1"
readonly TQ_ROOT="${SWAT_ROOT}/outputs/subproblem4_original_recipe/am_query_transfer_original_recipe_v1"
readonly STATUS_FILE="${OUTPUT_ROOT}/queue.status"
readonly TRAIN_GPU=0
readonly EVAL_GPU=0
readonly MEDIUM_RATIO="0.25"
readonly EXTREME_LOW_RATIO="0.03125"
readonly QWEN4B_BUDGET="${SWAT_ROOT}/multiagent/model/head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json"

export MULTIKV_DATASETS_ROOT="${SWAT_ROOT}/datasets"
export MULTIKV_MODELS_ROOT="${SWAT_ROOT}/models"
export MULTIKV_SCRATCH_ROOT="${SWAT_ROOT}/scratch"
export MULTIKV_PYTHON="${PYTHON}"
export MULTIKV_VLLM_PYTHON="${PYTHON}"
export MULTIKV_TEXMF_ROOT="${SWAT_ROOT}/texmf"
export HF_HOME="${SWAT_ROOT}/datasets/hf_home"
export HF_DATASETS_CACHE="${SWAT_ROOT}/datasets/hf_datasets"
export TMPDIR="${SWAT_ROOT}/tmp"
export PYTHONPATH="${SWAT_ROOT}:${PROJECT_ROOT}/scripts"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

install -d -m 0750 "${OUTPUT_ROOT}/logs" "${TQ_ROOT}" "${TMPDIR}"

write_status() {
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >"${STATUS_FILE}"
}

trap 'write_status "stopped signal_received"; exit 143' INT TERM

query_transfer_ready() {
  local root="$1"
  [[ -f "${root}/latest.json" ]] || return 1
  "${PYTHON}" - "${root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
latest = json.loads((root / "latest.json").read_text())
step = int(latest.get("step", -1))
checkpoint = str(latest.get("checkpoint", ""))
ready = (
    step >= 2000
    and checkpoint.startswith("exit-checkpoint-")
    and (root / checkpoint / "query_transfer.pt").is_file()
)
raise SystemExit(0 if ready else 1)
PY
}

train_query_transfer() {
  local pair="$1"
  local sender="$2"
  local receiver="$3"
  local root="${TQ_ROOT}/${pair}"
  local feature_dir="${root}/head_pairings"
  local features="${feature_dir}/features.pt"
  local pairing="${feature_dir}/pairing_tied_w7.json"
  install -d -m 0750 "${feature_dir}"

  if [[ ! -f "${features}" ]]; then
    write_status "query_features pair=${pair} gpu=${TRAIN_GPU}"
    "${PYTHON}" -u "${SWAT_ROOT}/multiagent/train/head_pairings/collect_query_features.py" \
      --sender-model "${sender}" \
      --receiver-model "${receiver}" \
      --datasets qasper+nox:train quality:train \
      --limit-per-dataset 4000 \
      --num-contexts 128 \
      --pairs-per-context 128 \
      --min-aligned 64 \
      --max-length 8192 \
      --seed 0 \
      --device cuda:0 \
      --receiver-device cuda:0 \
      --output "${features}" \
      >>"${OUTPUT_ROOT}/logs/${pair}_query_transfer.log" 2>&1
  fi

  if [[ ! -f "${pairing}" ]]; then
    write_status "query_head_pairing pair=${pair} gpu=${TRAIN_GPU}"
    "${PYTHON}" -u "${SWAT_ROOT}/multiagent/train/head_pairings/score_pairings.py" \
      --features "${features}" \
      --allowed-width 7 \
      --receiver-chunk 1 \
      --device cuda:0 \
      >>"${OUTPUT_ROOT}/logs/${pair}_query_transfer.log" 2>&1
  fi

  if ! query_transfer_ready "${root}"; then
    local resume_args=()
    if [[ -f "${root}/latest.json" ]]; then
      local resume_checkpoint
      resume_checkpoint="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "${root}/latest.json")"
      if [[ -f "${root}/${resume_checkpoint}/query_transfer.pt" ]]; then
        resume_args=(--resume-from "${root}")
      fi
    fi
    write_status "query_transfer_training pair=${pair} gpu=${TRAIN_GPU} max_steps=2000"
    "${PYTHON}" -u "${SWAT_ROOT}/multiagent/train/train_query_transfer.py" \
      --sender-model "${sender}" \
      --receiver-model "${receiver}" \
      --architecture mlp \
      --expansion 8 \
      --pairing-file "${pairing}" \
      --datasets qasper+nox:train quality:train \
      --max-length 4000 \
      --query-samples 1024 \
      --batch-size 1 \
      --epochs 6 \
      --max-steps 2000 \
      --learning-rate 1e-4 \
      --lr-scheduler cosine \
      --warmup-steps 100 \
      --lr-total-steps 6048 \
      --weight-decay 0.001 \
      --max-grad-norm 5.0 \
      --device cuda:0 \
      --dtype bfloat16 \
      --report-every 25 \
      --save-every 500 \
      --seed 43 \
      "${resume_args[@]}" \
      --output-dir "${root}" \
      >>"${OUTPUT_ROOT}/logs/${pair}_query_transfer.log" 2>&1
  fi
}

evaluate_budget() {
  local pair="$1"
  local label="$2"
  local ratio="$3"
  local budget_mode="$4"
  local log="${OUTPUT_ROOT}/logs/${pair}_${label}.log"
  local budget_args=(--am-head-budget-mode "${budget_mode}")
  if [[ "${budget_mode}" == "precomputed" ]]; then
    budget_args+=(--am-head-budget-path "${QWEN4B_BUDGET}")
  fi

  write_status "evaluating pair=${pair} budget=${label} ratio=${ratio} conditions=gsm8k:P4,hotpotqa:P3 gpu=${EVAL_GPU}"
  "${PYTHON}" -u "${EVALUATOR}" \
    --pair "${pair}" \
    --output-root "${OUTPUT_ROOT}/${label}_ratio_${ratio}" \
    --only-condition gsm8k:P4 \
    --only-condition hotpotqa:P3 \
    --am-query-transfer "${TQ_ROOT}/${pair}" \
    --am-target-ratio "${ratio}" \
    --am-budget-label "${label}" \
    --am-physical-gpu "${EVAL_GPU}" \
    "${budget_args[@]}" \
    >>"${log}" 2>&1
}

run_pair() {
  local pair="$1"
  local sender="$2"
  local receiver="$3"
  local budget_mode="$4"
  train_query_transfer "${pair}" "${sender}" "${receiver}"
  evaluate_budget "${pair}" medium "${MEDIUM_RATIO}" "${budget_mode}"
  evaluate_budget "${pair}" extreme_low "${EXTREME_LOW_RATIO}" "${budget_mode}"
}

write_status "started serial=true train_gpu=${TRAIN_GPU} eval_gpu=${EVAL_GPU} budgets=medium:${MEDIUM_RATIO},extreme_low:${EXTREME_LOW_RATIO}"

run_pair M1 \
  "${SWAT_ROOT}/models/qwen/Qwen3-0.6B" \
  "${SWAT_ROOT}/models/qwen/Qwen3-1.7B" \
  uniform

run_pair M2 \
  "${SWAT_ROOT}/models/qwen/Qwen3-0.6B" \
  "${SWAT_ROOT}/models/qwen/Qwen3-4B" \
  precomputed

run_pair M3 \
  "${SWAT_ROOT}/models/qwen/Qwen3-4B" \
  "${SWAT_ROOT}/models/qwen/Qwen3-0.6B" \
  uniform

write_status "completed_all pairs=M1,M2,M3 budgets=medium,extreme_low conditions=gsm8k:P4,hotpotqa:P3"
