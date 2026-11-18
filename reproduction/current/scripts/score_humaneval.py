#!/usr/bin/env python3
"""Score generated HumanEval completions in an isolated subprocess per sample."""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import time
from pathlib import Path
from typing import Any


PYTHON_BIN = "/mnt/hd/lewis/conda-envs/c2c/bin/python"
PROMPT_PREAMBLE = (
    "Complete the following Python function. "
    "Return only executable Python code, without Markdown fences.\n\n"
)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def extract_code(prediction: str) -> str:
    """Remove only common outer Markdown formatting, without repairing code."""
    fenced = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", prediction, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip("\n")
    return prediction.strip("\n")


def source_prompt(record: dict[str, Any]) -> str:
    content = record["messages"][0]["content"]
    if content.startswith(PROMPT_PREAMBLE):
        return content[len(PROMPT_PREAMBLE):]
    parts = content.split("\n\n", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot recover HumanEval prompt for {record['record_id']}")
    return parts[1]


def build_program(prediction: str, record: dict[str, Any]) -> tuple[str, str]:
    code = extract_code(prediction)
    entry_point = re.escape(record["entry_point"])
    has_full_definition = re.search(
        rf"(?m)^\s*(?:async\s+)?def\s+{entry_point}\s*\(", code
    ) is not None
    if has_full_definition:
        candidate = code
        mode = "full_function"
    else:
        prompt = source_prompt(record)
        separator = "" if prompt.endswith("\n") or code.startswith("\n") else "\n"
        candidate = prompt + separator + code
        mode = "completion"
    program = (
        candidate
        + "\n"
        + record["test"]
        + f"\ncheck({record['entry_point']})\n"
    )
    return program, mode


def execution_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    # RLIMIT_NPROC is UID-wide; leave headroom for existing shared-host threads.
    resource.setrlimit(resource.RLIMIT_NPROC, (1024, 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def execute(program: str, timeout_seconds: float) -> dict[str, Any]:
    command = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/mnt/hd/lewis/conda-envs/c2c", "/mnt/hd/lewis/conda-envs/c2c",
        "--tmpfs", "/tmp",
        "--proc", "/proc",
        "--dev", "/dev",
        "--dir", "/home",
        "--chdir", "/tmp",
        PYTHON_BIN,
        "-I",
        "-c",
        "import sys; source=sys.stdin.read(); exec(compile(source, '<humaneval>', 'exec'), {})",
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            input=program,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            preexec_fn=execution_limits,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "passed": False,
            "status": "timed_out",
            "returncode": None,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "stdout": (error.stdout or "")[-2000:] if isinstance(error.stdout, str) else "",
            "stderr": (error.stderr or "")[-4000:] if isinstance(error.stderr, str) else "",
        }
    return {
        "passed": result.returncode == 0,
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=6.0)
    args = parser.parse_args()

    records = {
        row["record_id"]: row
        for line in args.input.open(encoding="utf-8")
        if line.strip()
        for row in [json.loads(line)]
    }
    predictions = [
        json.loads(line)
        for line in args.predictions.open(encoding="utf-8")
        if line.strip()
    ]
    if len(predictions) != len(records):
        raise ValueError(f"Prediction/input mismatch: {len(predictions)} != {len(records)}")

    scores = []
    for prediction in predictions:
        record_id = prediction["record_id"]
        if record_id not in records:
            raise KeyError(f"Unknown record_id: {record_id}")
        program, candidate_mode = build_program(prediction["prediction"], records[record_id])
        result = execute(program, args.timeout_seconds)
        score = {
            "record_id": record_id,
            "task_id": records[record_id]["task_id"],
            "candidate_mode": candidate_mode,
            **result,
        }
        scores.append(score)
        print(json.dumps(score, ensure_ascii=False))

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for score in scores:
            stream.write(json.dumps(score, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, args.output)

    passed = sum(int(score["passed"]) for score in scores)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    summary.setdefault("quality_means", {})["pass_at_1"] = passed / len(scores)
    summary["humaneval_execution"] = {
        "passed": passed,
        "total": len(scores),
        "timeout_seconds_per_sample": args.timeout_seconds,
        "isolated_with_bubblewrap": True,
        "network_unshared": True,
        "score_file": str(args.output),
    }
    atomic_write_json(args.summary, summary)
    print(json.dumps({"passed": passed, "total": len(scores), "pass_at_1": passed / len(scores)}))


if __name__ == "__main__":
    main()
