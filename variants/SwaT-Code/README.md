# SwaT-Code

This directory is a controlled code-domain adaptation of the existing SwaT reproduction.
It is intentionally separate from `/mnt/hd2/lewis/SwaTReproduction` outputs.

- Starts from each completed original SwaT Phase-2 checkpoint and continues the same
  renderer for 4,000 code-only CE steps with the original `3e-6` learning rate.
- Uses MBPP `full/train` and 7,500 deterministic Python rows from CodeAlpaca-20k.
- Does not train on HumanEval. Exact prompt/solution matches and examples defining a
  HumanEval entry-point name are excluded.
- Trains M1, M2, and M3 serially on physical GPU 0 (RTX 3090), then runs the full
  shared-task suite serially: GSM8K P1/P4, HotpotQA P1/P2/P3/P4, and HumanEval P1/P4.
- No CUDA hard memory limit or token/KV reduction is applied. M1 is observed at about
  10.4 GiB, but static BF16 model weights plus FP32 renderer/gradient/Adam state already
  imply about 13.2 GiB for M2 and 12.7 GiB for M3 before activations. Therefore a strict
  50% policy on a 24 GiB RTX 3090 is feasible for M1 only.
- The code-data token audit found context <= 389, decode <= 1,749 and total <= 1,779;
  the configured 512/1,792 bounds therefore do not truncate a code sample.

The continuation design was selected because a full original Phase-2 rerun cannot fit
on a 24 GiB RTX 3090: even before Adam state allocation, its 4,096-token dense backward
exhausts the card. About 4,000 code-only updates also closely matches the expected number
of code CE draws in a 24,000-step full-mixture rerun.

Key locations:

- `data/data_manifest.json`: source, selection and leakage-screening record.
- `manifests/receiver_length_audit.json`: full-mixture token-length audit.
- `configs/M*_recipe.json`: resolved pair recipes.
- `outputs/<pair>/stage2_code_adaptation`: code-adapted checkpoints.
- `evaluations/full_shared_tasks_code_adaptation_v1`: complete results kept separate
  from original SwaT and the earlier HumanEval-only probe.
- `logs/M*.log` and `status/M*.json`: progress and resumable state.
