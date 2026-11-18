#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly PROJECT_ROOT="/home/lewis/Subproblem4UnifiedReproduction"
readonly SWAT_ROOT="/mnt/hd2/lewis/SwaTReproduction"
readonly PYTHON="${SWAT_ROOT}/env3.10/bin/python"
readonly EVALUATOR="${PROJECT_ROOT}/scripts/evaluate_swat_unified_no_am.py"
readonly OUTPUT_ROOT="${SWAT_ROOT}/outputs/subproblem4_original_recipe/evaluations/swat_full_cache_no_am_v1"
readonly PAIR="${1:?usage: run_swat_eval_guarded.sh M1|M2|M3}"
readonly GPU="${SWAT_EVAL_GPU:-0}"
readonly MAX_TOTAL_USED_MIB="${SWAT_EVAL_MAX_TOTAL_USED_MIB:-19000}"
readonly STATUS_FILE="${OUTPUT_ROOT}/parallel_${PAIR}.status"
readonly LOG_FILE="${OUTPUT_ROOT}/logs/${PAIR}.log"

job_pid=""

write_status() {
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >"${STATUS_FILE}"
}

terminate_job() {
  if [[ -n "${job_pid}" ]] && kill -0 "${job_pid}" 2>/dev/null; then
    kill -TERM -- "-${job_pid}" 2>/dev/null || kill -TERM "${job_pid}" 2>/dev/null || true
    wait "${job_pid}" 2>/dev/null || true
  fi
}

trap 'terminate_job; write_status "stopped signal_received"; exit 143' INT TERM

write_status "starting pair=${PAIR} gpu=${GPU} max_total_used_mib=${MAX_TOTAL_USED_MIB}"
setsid "${PYTHON}" -u "${EVALUATOR}" \
  --pair "${PAIR}" \
  --output-root "${OUTPUT_ROOT}" \
  >>"${LOG_FILE}" 2>&1 &
job_pid=$!

over_limit_samples=0
while kill -0 "${job_pid}" 2>/dev/null; do
  used_mib="$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if (( used_mib > MAX_TOTAL_USED_MIB )); then
    over_limit_samples=$((over_limit_samples + 1))
  else
    over_limit_samples=0
  fi
  write_status "running pair=${PAIR} pid=${job_pid} gpu=${GPU} total_used_mib=${used_mib} limit_mib=${MAX_TOTAL_USED_MIB} over_limit_samples=${over_limit_samples}"
  if (( over_limit_samples >= 3 )); then
    terminate_job
    write_status "stopped_memory_guard pair=${PAIR} total_used_mib=${used_mib} limit_mib=${MAX_TOTAL_USED_MIB}"
    exit 78
  fi
  sleep 2
done

set +e
wait "${job_pid}"
exit_status=$?
set -e
job_pid=""
if (( exit_status == 0 )); then
  write_status "completed pair=${PAIR}"
else
  write_status "failed pair=${PAIR} exit=${exit_status}"
fi
exit "${exit_status}"
