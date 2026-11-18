#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly ROOT="/mnt/hd2/lewis/SwaT-Code"
readonly SWAT="/mnt/hd2/lewis/SwaTReproduction"
readonly PYTHON="${SWAT}/env3.10/bin/python"

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
export PYTHONPATH="${SWAT}:/home/lewis/Subproblem4UnifiedReproduction/scripts"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

install -d -m 0750 "${ROOT}/logs" "${ROOT}/status" "${ROOT}/scratch" "${ROOT}/tmp"

for pair in M1 M2 M3; do
  printf '%s starting %s on physical GPU 0\n' "$(date --iso-8601=seconds)" "${pair}"
  "${PYTHON}" -u "${ROOT}/scripts/run_pair.py" "${pair}" \
    2>&1 | tee -a "${ROOT}/logs/${pair}_code_adaptation.log"
done

printf '%s completed M1 M2 M3 training and HumanEval P4 evaluation\n' "$(date --iso-8601=seconds)"
