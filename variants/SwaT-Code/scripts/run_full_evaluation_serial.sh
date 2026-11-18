#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly ROOT="/mnt/hd2/lewis/SwaT-Code"
readonly SWAT="/mnt/hd2/lewis/SwaTReproduction"
readonly PROJECT="/home/lewis/Subproblem4UnifiedReproduction"
readonly PYTHON="${SWAT}/env3.10/bin/python"
readonly EVALUATOR="${PROJECT}/scripts/evaluate_swat_unified_no_am.py"
readonly OUTPUT_ROOT="${ROOT}/evaluations/full_shared_tasks_code_adaptation_v1"
readonly STATUS="${ROOT}/status/full_evaluation.status"

export MULTIKV_DATASETS_ROOT="${SWAT}/datasets"
export MULTIKV_MODELS_ROOT="${SWAT}/models"
export MULTIKV_SCRATCH_ROOT="${ROOT}/scratch"
export MULTIKV_PYTHON="${PYTHON}"
export MULTIKV_VLLM_PYTHON="${PYTHON}"
export MULTIKV_TEXMF_ROOT="${SWAT}/texmf"
export HF_HOME="${ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${ROOT}/cache/huggingface/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="${ROOT}/tmp"
export PYTHONPATH="${SWAT}:${PROJECT}/scripts"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

install -d -m 0750 "${OUTPUT_ROOT}/logs" "${ROOT}/status" "${ROOT}/tmp"

write_status() {
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >"${STATUS}"
}

checkpoint_for() {
  case "$1" in
    M1) printf '%s\n' "${ROOT}/outputs/m1_qwen3_0.6b_to_qwen3_1.7b/stage2_code_adaptation/exit-checkpoint-4000/renderer.pt" ;;
    M2) printf '%s\n' "${ROOT}/outputs/m2_qwen3_0.6b_to_qwen3_4b/stage2_code_adaptation/exit-checkpoint-4000/renderer.pt" ;;
    M3) printf '%s\n' "${ROOT}/outputs/m3_qwen3_4b_to_qwen3_0.6b/stage2_code_adaptation/exit-checkpoint-4000/renderer.pt" ;;
  esac
}

while true; do
  missing=()
  for pair in M1 M2 M3; do
    checkpoint="$(checkpoint_for "${pair}")"
    [[ -f "${checkpoint}" ]] || missing+=("${pair}")
  done
  if (( ${#missing[@]} == 0 )) && ! systemctl --user is-active --quiet swat-code-m1-m2-m3.service; then
    break
  fi
  if (( ${#missing[@]} == 0 )); then
    write_status "waiting_for_training_service_exit checkpoints=M1,M2,M3"
  else
    write_status "waiting_for_training missing=${missing[*]}"
  fi
  sleep 30
done

write_status "starting full_suite conditions=gsm8k:P1,P4 hotpotqa:P1,P2,P3,P4 humaneval:P1,P4"
for pair in M1 M2 M3; do
  checkpoint="$(checkpoint_for "${pair}")"
  write_status "evaluating pair=${pair} gpu=0"
  "${PYTHON}" -u "${EVALUATOR}" \
    --pair "${pair}" \
    --checkpoint "${checkpoint}" \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda:0 \
    >>"${OUTPUT_ROOT}/logs/${pair}.log" 2>&1
done

write_status "completed pairs=M1,M2,M3 conditions_per_pair=8"
