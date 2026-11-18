from __future__ import annotations

import re
from typing import Any


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _normalize_number(value: str) -> str:
    value = value.replace(",", "").strip().rstrip(".")
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return f"{number:.12g}"


def extract_gsm8k_answer(text: str) -> str:
    marker = text.rsplit("####", 1)[-1] if "####" in text else text
    numbers = NUMBER_RE.findall(marker)
    if not numbers:
        numbers = NUMBER_RE.findall(text)
    return _normalize_number(numbers[-1]) if numbers else ""


def extract_last_boxed(text: str) -> str:
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return text.strip()
    start = starts[-1]
    opening = text.find("{", start)
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index].strip()
    return text[opening + 1 :].strip()


def _normalize_math(text: str) -> str:
    text = extract_last_boxed(text)
    text = text.replace("$", "").replace("\\,", "").replace(" ", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return text.rstrip(".")


def math_equal(prediction: str, gold: str) -> bool:
    try:
        from math_verify import parse, verify

        return bool(verify(parse(gold), parse(prediction)))
    except Exception:
        return _normalize_math(prediction) == _normalize_math(gold)


def extract_arc_answer(text: str) -> str:
    explicit = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\(?([A-D1-4])\)?",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit[-1].upper()
    standalone = re.findall(
        r"(?<![A-Za-z0-9])([A-D1-4])(?![A-Za-z0-9])", text.upper()
    )
    return standalone[-1] if standalone else ""


def score_answer(record: dict[str, Any], prediction: str) -> bool:
    dataset = str(record.get("dataset", "")).lower()
    gold = str(record.get("gold_answer", record.get("answer", "")))
    if dataset == "gsm8k":
        return extract_gsm8k_answer(prediction) == extract_gsm8k_answer(gold)
    if dataset in {"math", "math500", "math-500"}:
        return math_equal(prediction, gold)
    if dataset in {"arc", "arc-c", "arc_challenge"}:
        return extract_arc_answer(prediction) == extract_arc_answer(gold)
    return prediction.strip() == gold.strip()
