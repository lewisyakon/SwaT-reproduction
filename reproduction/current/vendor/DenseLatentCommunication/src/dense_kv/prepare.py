from __future__ import annotations

from pathlib import Path
from typing import Any

from .answers import extract_gsm8k_answer, extract_last_boxed
from .data import write_jsonl


def _choice_question(question: str, choices: dict[str, list[str]]) -> str:
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return f"{question}\n\nChoices:\n{rendered}"


def _load_math_train():
    from datasets import load_dataset

    try:
        dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
        return [
            row
            for row in dataset
            if "algebra" in str(row.get("type", row.get("subject", ""))).lower()
        ]
    except Exception as first_error:
        rows: list[dict[str, Any]] = []
        try:
            for subject in ("algebra", "prealgebra", "intermediate_algebra"):
                rows.extend(
                    load_dataset(
                        "EleutherAI/hendrycks_math", subject, split="train"
                    )
                )
            return rows
        except Exception:
            raise RuntimeError("Unable to load either supported MATH training source") from first_error


def prepare_datasets(
    output_dir: str | Path,
    *,
    max_train_per_dataset: int | None = None,
    max_eval_per_dataset: int | None = None,
) -> tuple[Path, Path]:
    from datasets import load_dataset

    output_dir = Path(output_dir)
    train: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []

    gsm = load_dataset("openai/gsm8k", "main")
    gsm_train = list(gsm["train"])
    gsm_eval = list(gsm["test"])
    if max_train_per_dataset is not None:
        gsm_train = gsm_train[:max_train_per_dataset]
    if max_eval_per_dataset is not None:
        gsm_eval = gsm_eval[:max_eval_per_dataset]
    for index, row in enumerate(gsm_train):
        train.append(
            {
                "id": f"gsm8k-train-{index}",
                "dataset": "gsm8k",
                "question": row["question"],
                "gold_answer": extract_gsm8k_answer(row["answer"]),
                "reference_solution": row["answer"],
            }
        )
    for index, row in enumerate(gsm_eval):
        evaluation.append(
            {
                "id": f"gsm8k-test-{index}",
                "dataset": "gsm8k",
                "question": row["question"],
                "gold_answer": extract_gsm8k_answer(row["answer"]),
                "reference_solution": row["answer"],
            }
        )

    math_train = list(_load_math_train())
    if max_train_per_dataset is not None:
        math_train = math_train[:max_train_per_dataset]
    for index, row in enumerate(math_train):
        solution = str(row.get("solution", row.get("answer", "")))
        train.append(
            {
                "id": f"math-algebra-train-{index}",
                "dataset": "math",
                "question": row["problem"],
                "gold_answer": extract_last_boxed(solution),
                "reference_solution": solution,
                "subject": row.get("type", row.get("subject", "algebra")),
            }
        )

    math500 = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    if max_eval_per_dataset is not None:
        math500 = math500[:max_eval_per_dataset]
    for index, row in enumerate(math500):
        evaluation.append(
            {
                "id": str(row.get("unique_id", f"math500-test-{index}")),
                "dataset": "math500",
                "question": row["problem"],
                "gold_answer": str(
                    row.get("answer", extract_last_boxed(str(row.get("solution", ""))))
                ),
                "reference_solution": row.get("solution"),
                "subject": row.get("subject"),
            }
        )

    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    arc_train = list(arc["train"])
    arc_eval = list(arc["validation"])
    if max_train_per_dataset is not None:
        arc_train = arc_train[:max_train_per_dataset]
    if max_eval_per_dataset is not None:
        arc_eval = arc_eval[:max_eval_per_dataset]
    for split, rows, destination in (
        ("train", arc_train, train),
        ("validation", arc_eval, evaluation),
    ):
        for index, row in enumerate(rows):
            destination.append(
                {
                    "id": str(row.get("id", f"arc-{split}-{index}")),
                    "dataset": "arc-c",
                    "question": _choice_question(row["question"], row["choices"]),
                    "gold_answer": row["answerKey"],
                }
            )

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(eval_path, evaluation)
    return train_path, eval_path

