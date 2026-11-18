#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly PROJECT_ROOT="/home/lewis/Subproblem4UnifiedReproduction"
readonly SWAT_ROOT="/mnt/hd2/lewis/SwaTReproduction"
readonly PYTHON="${SWAT_ROOT}/env3.10/bin/python"
readonly EVALUATOR="${PROJECT_ROOT}/scripts/evaluate_swat_unified_no_am.py"
readonly GUARDED_RUNNER="${PROJECT_ROOT}/scripts/run_swat_eval_guarded.sh"
readonly OUTPUT_ROOT="${SWAT_ROOT}/outputs/subproblem4_original_recipe/evaluations/swat_full_cache_no_am_v1"
readonly STATUS_FILE="${OUTPUT_ROOT}/queue.status"
readonly GPU=0
readonly MIN_FREE_MIB=18000
readonly MAX_TOTAL_USED_MIB=19000

export MULTIKV_DATASETS_ROOT="${SWAT_ROOT}/datasets"
export MULTIKV_MODELS_ROOT="${SWAT_ROOT}/models"
export MULTIKV_SCRATCH_ROOT="${SWAT_ROOT}/scratch"
export MULTIKV_PYTHON="${PYTHON}"
export MULTIKV_VLLM_PYTHON="${PYTHON}"
export MULTIKV_TEXMF_ROOT="${SWAT_ROOT}/texmf"
export HF_HOME="${SWAT_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${SWAT_ROOT}/datasets/hf_datasets"
export TMPDIR="${SWAT_ROOT}/tmp"
export PYTHONPATH="${SWAT_ROOT}:${PROJECT_ROOT}/scripts"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

install -d -m 0750 "${OUTPUT_ROOT}/logs" "${TMPDIR}"

write_status() {
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >"${STATUS_FILE}"
}

m1_pid=""
m2_pid=""
m3_pid=""

stop_children() {
  for pid in "${m1_pid}" "${m2_pid}" "${m3_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}

trap 'write_status "stopped signal_received"; stop_children; exit 143' INT TERM

wait_for_gpu() {
  while true; do
    free_mib="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib >= MIN_FREE_MIB )); then
      return 0
    fi
    write_status "waiting_for_gpu gpu=${GPU} free_mib=${free_mib} required_mib=${MIN_FREE_MIB}"
    sleep 30
  done
}

wait_for_gpu
write_status "starting_parallel pairs=M1,M2 gpu=${GPU} max_total_used_mib=${MAX_TOTAL_USED_MIB}"

"${PYTHON}" -u "${EVALUATOR}" \
  --pair M1 \
  --output-root "${OUTPUT_ROOT}" \
  >>"${OUTPUT_ROOT}/logs/M1.log" 2>&1 &
m1_pid=$!

SWAT_EVAL_GPU="${GPU}" \
SWAT_EVAL_MAX_TOTAL_USED_MIB="${MAX_TOTAL_USED_MIB}" \
  "${GUARDED_RUNNER}" M2 &
m2_pid=$!

write_status "running_parallel M1_pid=${m1_pid} M2_guard_pid=${m2_pid} gpu=${GPU} max_total_used_mib=${MAX_TOTAL_USED_MIB}"

set +e
wait "${m1_pid}"
m1_status=$?
m1_pid=""
wait "${m2_pid}"
m2_status=$?
m2_pid=""
set -e

if (( m1_status != 0 )); then
  write_status "failed_parallel M1_exit=${m1_status} M2_exit=${m2_status}"
  exit 1
fi

if (( m2_status != 0 )); then
  write_status "resuming_M2_serial after_parallel_exit=${m2_status}"
  wait_for_gpu
  "${PYTHON}" -u "${EVALUATOR}" \
    --pair M2 \
    --output-root "${OUTPUT_ROOT}" \
    >>"${OUTPUT_ROOT}/logs/M2.log" 2>&1
fi

write_status "starting_serial pair=M3 gpu=${GPU}"
wait_for_gpu
"${PYTHON}" -u "${EVALUATOR}" \
  --pair M3 \
  --output-root "${OUTPUT_ROOT}" \
  >>"${OUTPUT_ROOT}/logs/M3.log" 2>&1 &
m3_pid=$!
wait "${m3_pid}"
m3_pid=""

write_status "completed_all pairs=M1,M2,M3 conditions_per_pair=8 compression=none"
