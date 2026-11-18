#!/usr/bin/env python3
"""Build leakage-screened Python trace files for the SwaT-Code experiment."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_fence(text: str) -> str:
    text = text.strip()
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.I | re.S)
    if blocks:
        return max(blocks, key=len).strip()
    return text


def python_program(text: str) -> str | None:
    code = strip_fence(text)
    if not code or len(code) > 24000:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    meaningful = (
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.Call,
        ast.ClassDef,
        ast.For,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.If,
        ast.Import,
        ast.ImportFrom,
        ast.ListComp,
        ast.DictComp,
        ast.Return,
        ast.Try,
        ast.While,
        ast.With,
    )
    if not any(isinstance(node, meaningful) for node in ast.walk(tree)):
        return None
    return code


def read_humaneval(path: Path) -> tuple[set[str], set[str], set[str]]:
    prompts: set[str] = set()
    solutions: set[str] = set()
    entry_points: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            prompts.update(normalized(message["content"]) for message in row["messages"])
            solutions.add(normalized(row["canonical_solution"]))
            entry_points.add(str(row["entry_point"]))
    return prompts, solutions, entry_points


def leaks_humaneval(
    question: str,
    code: str,
    eval_prompts: set[str],
    eval_solutions: set[str],
    entry_points: set[str],
) -> bool:
    question_norm, code_norm = normalized(question), normalized(code)
    if question_norm in eval_prompts or code_norm in eval_solutions:
        return True
    # A matching HumanEval function name is a conservative contamination signal.
    defined = set(re.findall(r"\bdef\s+([A-Za-z_]\w*)\s*\(", code))
    return bool(defined & entry_points)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--humaneval", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--codealpaca-limit", type=int, default=7500)
    parser.add_argument("--max-target-tokens", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    eval_prompts, eval_solutions, entry_points = read_humaneval(args.humaneval)
    rejected = {"invalid_python": 0, "too_long": 0, "humaneval_overlap": 0}

    def eligible(question: str, raw_code: str) -> str | None:
        code = python_program(raw_code)
        if code is None:
            rejected["invalid_python"] += 1
            return None
        if len(tokenizer(code, add_special_tokens=False)["input_ids"]) > args.max_target_tokens:
            rejected["too_long"] += 1
            return None
        if leaks_humaneval(question, code, eval_prompts, eval_solutions, entry_points):
            rejected["humaneval_overlap"] += 1
            return None
        return code

    mbpp_source = load_dataset("google-research-datasets/mbpp", "full", split="train")
    mbpp_rows: list[dict] = []
    for source in mbpp_source:
        tests = "\n".join(str(test) for test in source["test_list"])
        question = (
            "Write a complete Python solution for the following task. Return only executable Python code, "
            "without Markdown fences.\n\n"
            f"Task: {source['text']}\n\nPublic tests:\n{tests}"
        )
        code = eligible(question, str(source["code"]))
        if code is None:
            continue
        mbpp_rows.append(
            {
                "id": f"mbpp/train/{source['task_id']}",
                "dataset": "mbpp_code_train",
                "split": "train",
                "question": question,
                "trace": code,
                "messages": [{"role": "user", "content": question}],
                "source": "google-research-datasets/mbpp:full:train",
                "source_task_id": str(source["task_id"]),
            }
        )

    alpaca_source = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    indices = list(range(len(alpaca_source)))
    random.Random(args.seed).shuffle(indices)
    alpaca_rows: list[dict] = []
    for index in indices:
        source = alpaca_source[index]
        input_text = str(source["input"]).strip()
        question = str(source["instruction"]).strip()
        if input_text:
            question += f"\n\nInput/context:\n{input_text}"
        question += "\n\nReturn only executable Python code, without Markdown fences."
        code = eligible(question, str(source["output"]))
        if code is None:
            continue
        alpaca_rows.append(
            {
                "id": f"codealpaca/train/{index}",
                "dataset": "codealpaca_python_train",
                "split": "train",
                "question": question,
                "trace": code,
                "messages": [{"role": "user", "content": question}],
                "source": "sahil2801/CodeAlpaca-20k:train",
                "source_row": index,
            }
        )
        if len(alpaca_rows) >= args.codealpaca_limit:
            break

    if len(alpaca_rows) < args.codealpaca_limit:
        raise RuntimeError(
            f"only {len(alpaca_rows)} eligible CodeAlpaca rows; requested {args.codealpaca_limit}"
        )

    mbpp_path = args.output_root / "mbpp_train_python_traces.jsonl"
    alpaca_path = args.output_root / "codealpaca_python_7500_traces.jsonl"
    write_jsonl(mbpp_path, mbpp_rows)
    write_jsonl(alpaca_path, alpaca_rows)
    manifest = {
        "schema": "swat-code-data-v1",
        "purpose": "Phase-2 code-domain adaptation without using HumanEval test targets",
        "seed": args.seed,
        "max_target_tokens": args.max_target_tokens,
        "human_eval_exclusion": {
            "path": str(args.humaneval),
            "sha256": sha256(args.humaneval),
            "rules": [
                "exclude exact normalized prompt matches",
                "exclude exact normalized canonical-solution matches",
                "exclude code defining any HumanEval entry-point name",
            ],
        },
        "datasets": {
            "mbpp": {
                "source": "google-research-datasets/mbpp",
                "config": "full",
                "split": "train only",
                "rows": len(mbpp_rows),
                "path": str(mbpp_path),
                "sha256": sha256(mbpp_path),
            },
            "codealpaca": {
                "source": "sahil2801/CodeAlpaca-20k",
                "split": "train only",
                "selection": f"deterministic shuffle seed {args.seed}, first {len(alpaca_rows)} eligible Python rows",
                "rows": len(alpaca_rows),
                "path": str(alpaca_path),
                "sha256": sha256(alpaca_path),
            },
        },
        "rejected_candidates": rejected,
    }
    manifest_path = args.output_root / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
