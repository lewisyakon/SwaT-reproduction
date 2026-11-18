#!/usr/bin/env python3
"""Continue one trained SwaT renderer on code data, then run HumanEval P4."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/mnt/hd2/lewis/SwaT-Code")
SWAT = Path("/mnt/hd2/lewis/SwaTReproduction")
PROJECT = Path("/home/lewis/Subproblem4UnifiedReproduction")
PYTHON = SWAT / "env3.10/bin/python"
TRAINER = SWAT / "multiagent/train/train_cache_renderer.py"
EVALUATOR = PROJECT / "scripts/evaluate_swat_unified_no_am.py"
CODE_DATA = [
    ROOT / "data/mbpp_train_python_traces.jsonl",
    ROOT / "data/codealpaca_python_7500_traces.jsonl",
]
PAIRS = {
    "M1": "m1_qwen3_0.6b_to_qwen3_1.7b",
    "M2": "m2_qwen3_0.6b_to_qwen3_4b",
    "M3": "m3_qwen3_4b_to_qwen3_0.6b",
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(output: Path) -> Path | None:
    candidates = [
        path
        for path in output.glob("*checkpoint-*")
        if checkpoint_step(path) >= 0 and (path / "renderer.pt").is_file()
    ]
    return max(candidates, key=checkpoint_step) if candidates else None


def flags(values: dict) -> list[str]:
    result: list[str] = []
    for name, value in values.items():
        if value is None or value is False:
            continue
        result.append("--" + name.replace("_", "-"))
        if value is True:
            continue
        if isinstance(value, list):
            result.extend(str(item) for item in value)
        else:
            result.append(str(value))
    return result


def status(pair: str, state: str, **extra) -> None:
    atomic_json(
        ROOT / "status" / f"{pair}.json",
        {"pair": pair, "state": state, "updated_unix": time.time(), **extra},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair", choices=sorted(PAIRS))
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--run-eval", action="store_true", help="Run the legacy immediate HumanEval-P4 check")
    args = parser.parse_args()

    pair_name = PAIRS[args.pair]
    original_pair = SWAT / "outputs/subproblem4_original_recipe" / pair_name
    original_args_path = original_pair / "stage2/args.json"
    original_args = json.loads(original_args_path.read_text(encoding="utf-8"))
    output = ROOT / "outputs" / pair_name / "stage2_code_adaptation"
    adaptation_steps = 4000
    final_checkpoint = output / f"exit-checkpoint-{adaptation_steps}"

    train_args = dict(original_args)
    train_args["datasets"] = [str(path) for path in CODE_DATA]
    train_args["output_dir"] = str(output)
    train_args["init_weights_from"] = str(original_pair / "stage2/exit-checkpoint-24000")
    train_args["resume_from"] = None
    train_args["max_length"] = 512
    train_args["max_trace_length"] = 1792
    train_args["epochs"] = 1
    train_args["max_steps"] = adaptation_steps
    train_args["save_every"] = 1000
    train_args["dense_batch_fraction"] = 0.0
    train_args["dense_datasets"] = None
    train_args["downstream_eval_every"] = 0
    train_args["downstream_eval_datasets"] = None
    # Full code-data audit: context <=389, decode <=1749, total <=1779.
    train_args["preflight_max_receiver_tokens"] = 1779

    resume = latest_checkpoint(output)
    if resume is not None and checkpoint_step(resume) < int(train_args["max_steps"]):
        train_args["resume_from"] = str(resume)
        train_args["init_weights_from"] = None

    recipe = {
        "schema": "swat-code-pair-v2",
        "pair": args.pair,
        "pair_name": pair_name,
        "design": "Domain-adaptation continuation of the completed original Phase-2 renderer on code traces only",
        "initial_checkpoint": str(original_pair / "stage2/exit-checkpoint-24000"),
        "original_stage2_args": str(original_args_path),
        "code_datasets": [str(path) for path in CODE_DATA],
        "adaptation_steps": adaptation_steps,
        "exposure_rationale": "About 4,000 code-only steps approximates the 3,500-3,600 code CE draws expected from a 24,000-step full-mixture rerun with dense fraction 0.3",
        "resource_only_override": {
            "max_length": 512,
            "max_trace_length": 1792,
            "preflight_max_receiver_tokens": 1779,
            "basis": str(ROOT / "manifests/receiver_length_audit.json"),
            "note": "The audited code maxima are context=389, decode=1749 and total=1779, so these bounds do not truncate code training samples",
        },
        "training_args": train_args,
        "evaluation": {
            "dataset": "humaneval",
            "protocol": "P4",
            "checkpoint": str(final_checkpoint / "renderer.pt"),
            "output_root": str(ROOT / "evaluations/humaneval_p4_code_adaptation_v1"),
        },
    }
    atomic_json(ROOT / "configs" / f"{args.pair}_recipe.json", recipe)

    if not (final_checkpoint / "renderer.pt").is_file():
        status(
            args.pair,
            "training",
            output=str(output),
            resumed_from=str(resume) if resume else None,
        )
        subprocess.run(
            [str(PYTHON), "-u", str(TRAINER), *flags(train_args)],
            cwd=SWAT,
            env=os.environ.copy(),
            check=True,
        )
    status(args.pair, "training_complete", checkpoint=str(final_checkpoint / "renderer.pt"))

    if args.run_eval and not args.skip_eval:
        status(args.pair, "evaluating_humaneval_p4", checkpoint=str(final_checkpoint / "renderer.pt"))
        subprocess.run(
            [
                str(PYTHON),
                "-u",
                str(EVALUATOR),
                "--pair",
                args.pair,
                "--checkpoint",
                str(final_checkpoint / "renderer.pt"),
                "--output-root",
                str(ROOT / "evaluations/humaneval_p4_code_adaptation_v1"),
                "--only-condition",
                "humaneval:P4",
                "--device",
                "cuda:0",
            ],
            cwd=PROJECT,
            env=os.environ.copy(),
            check=True,
        )
        status(
            args.pair,
            "complete",
            checkpoint=str(final_checkpoint / "renderer.pt"),
            evaluation=str(ROOT / "evaluations/humaneval_p4_code_adaptation_v1" / pair_name / "humaneval/P4/summary.json"),
        )
    else:
        status(
            args.pair,
            "training_complete_pending_full_evaluation",
            checkpoint=str(final_checkpoint / "renderer.pt"),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        pair = next((item for item in sys.argv[1:] if item in PAIRS), "unknown")
        status(pair, "failed", error=f"{type(error).__name__}: {error}")
        raise
