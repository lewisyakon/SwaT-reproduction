#!/usr/bin/env python3
"""Evaluate the frozen T2T prompt bundle on the three unified test sets.

The prompt bundle itself is read directly from the user-provided ZIP archive so
that the evaluated text can be tied to the archive checksum.  One invocation
loads a sender/receiver pair once and evaluates all eight applicable protocol
conditions serially.  Result JSONL files are append-only and resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import string
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("SUBPROBLEM4_PROJECT_ROOT", REPO_ROOT / "reproduction" / "current"))
ASSET_ROOT = Path(os.environ.get("SUBPROBLEM4_ASSET_ROOT", REPO_ROOT))
DENSE_ROOT = PROJECT_ROOT / "vendor" / "DenseLatentCommunication"
if str(DENSE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(DENSE_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dense_kv.config import load_config  # noqa: E402
from dense_kv.runtime import load_model, load_tokenizer, parse_dtype, set_seed  # noqa: E402
from score_humaneval import build_program, execute  # noqa: E402


PAIR_CONFIGS = {
    "M1": PROJECT_ROOT / "configs/stage1/converters/dense_gsm8k_math_algebra_arc_c/m1_sender_qwen3_0.6b__receiver_qwen3_1.7b.json",
    "M2": PROJECT_ROOT / "configs/stage1/converters/dense_gsm8k_math_algebra_arc_c/m2_sender_qwen3_0.6b__receiver_qwen3_4b.json",
    "M3": PROJECT_ROOT / "configs/stage1/converters/dense_gsm8k_math_algebra_arc_c/m3_sender_qwen3_4b__receiver_qwen3_0.6b.json",
}
RAW_ROOT = ASSET_ROOT / "datasets"
DEFAULT_PROMPT_ZIP = REPO_ROOT / "reproduction" / "prompts" / "提示词模板.zip"
DEFAULT_OUTPUT_ROOT = ASSET_ROOT / "experiments/stage1/evaluations/t2t_generated_only_v1"
CONDITIONS = (
    ("gsm8k", "P1", "generated"),
    ("gsm8k", "P4", "sender_restated_q"),
    ("hotpotqa", "P1", "generated"),
    ("hotpotqa", "P2", "generated"),
    ("hotpotqa", "P3", "generated"),
    ("hotpotqa", "P4", "sender_restated_q"),
    ("humaneval", "P1", "generated"),
    ("humaneval", "P4", "sender_restated_q"),
)
RESULT_SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_bundle(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.endswith("prompt_assembly_generated_only_v1.json")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one prompt assembly JSON in {path}, found {matches}")
        return json.loads(archive.read(matches[0]).decode("utf-8"))


def parquet_rows(paths: Iterable[Path]):
    for path in sorted(paths):
        yield from pq.read_table(path).to_pylist()


def gsm8k_records() -> Iterable[dict[str, Any]]:
    paths = (RAW_ROOT / "GSM8K" / "main").glob("test-*.parquet")
    for index, row in enumerate(parquet_rows(paths)):
        yield {
            "record_id": f"gsm8k:test:{index:05d}",
            "task_id": None,
            "q": str(row["question"]),
            "answer": str(row["answer"]),
            "all_evidence": [],
            "source_evidence": [],
            "receiver_evidence": [],
            "p2_eligible": False,
        }


def humaneval_records() -> Iterable[dict[str, Any]]:
    paths = (RAW_ROOT / "HumanEval" / "openai_humaneval").glob("test-*.parquet")
    for row in parquet_rows(paths):
        question = str(row["prompt"])
        yield {
            "record_id": f"humaneval:test:{row['task_id']}",
            "task_id": str(row["task_id"]),
            "q": question,
            "answer": None,
            "canonical_solution": str(row["canonical_solution"]),
            "test": str(row["test"]),
            "entry_point": str(row["entry_point"]),
            "messages": [{
                "role": "user",
                "content": (
                    "Complete the following Python function. "
                    "Return only executable Python code, without Markdown fences.\n\n"
                    + question
                ),
            }],
            "all_evidence": [],
            "source_evidence": [],
            "receiver_evidence": [],
            "p2_eligible": False,
        }


def hotpotqa_records() -> Iterable[dict[str, Any]]:
    paths = (RAW_ROOT / "HotpotQA" / "distractor").glob("validation-*.parquet")
    for row in parquet_rows(paths):
        if row["type"] != "bridge":
            continue
        contexts = [
            {
                "title": str(title),
                "text": " ".join(str(sentence).strip() for sentence in sentences),
            }
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
        ]
        support_titles = list(dict.fromkeys(str(title) for title in row["supporting_facts"]["title"]))
        support = [item for item in contexts if item["title"] in support_titles]
        distractors = [item for item in contexts if item["title"] not in support_titles]
        p2_eligible = len(support) == 2 and len(distractors) >= 8
        source_titles = {
            item["title"] for item in (support[:1] + distractors[:4])
        } if p2_eligible else set()
        receiver_titles = {
            item["title"] for item in (support[1:2] + distractors[4:8])
        } if p2_eligible else set()
        yield {
            "record_id": f"hotpotqa_bridge:validation:{row['id']}",
            "task_id": None,
            "q": str(row["question"]),
            "answer": str(row["answer"]),
            "all_evidence": contexts,
            # Preserve the original context order inside both fixed P2 views.
            "source_evidence": [item for item in contexts if item["title"] in source_titles],
            "receiver_evidence": [item for item in contexts if item["title"] in receiver_titles],
            "p2_eligible": p2_eligible,
            "support_titles": support_titles,
            "context_count": len(contexts),
        }


def records_for(dataset: str) -> Iterable[dict[str, Any]]:
    if dataset == "gsm8k":
        return gsm8k_records()
    if dataset == "hotpotqa":
        return hotpotqa_records()
    if dataset == "humaneval":
        return humaneval_records()
    raise ValueError(dataset)


def format_evidence(items: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[{item['title']}]\n{item['text']}" for item in items)


def view(question: str, evidence: list[dict[str, str]]) -> str:
    result = f"<TASK>\n{question}\n</TASK>"
    if evidence:
        result += f"\n<EVIDENCE>\n{format_evidence(evidence)}\n</EVIDENCE>"
    return result


def protocol_views(record: dict[str, Any], protocol: str, p0: str) -> tuple[str, str]:
    if protocol == "P1":
        complete = view(record["q"], record["all_evidence"])
        return complete, complete
    if protocol == "P2":
        if not record["p2_eligible"]:
            raise ValueError(f"P2-ineligible record: {record['record_id']}")
        return (
            view(record["q"], record["source_evidence"]),
            view(record["q"], record["receiver_evidence"]),
        )
    if protocol == "P3":
        return view(record["q"], record["all_evidence"]), view(record["q"], [])
    if protocol == "P4":
        return view(record["q"], record["all_evidence"]), p0
    raise ValueError(protocol)


def apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(result, torch.Tensor):
        result = result.tolist()
    if result and isinstance(result[0], list):
        result = result[0]
    return list(result)


@torch.no_grad()
def generate_to_context_limit(
    model,
    prefix_ids: list[int],
    *,
    context_limit: int,
    eos_token_ids: set[int],
) -> tuple[list[int], dict[str, Any]]:
    """Greedy generation bounded only by EOS, context capacity, or CUDA OOM."""
    if not prefix_ids:
        raise ValueError("Generation prefix cannot be empty")
    available = context_limit - len(prefix_ids)
    if available <= 0:
        return [], {
            "finish_reason": "prompt_exceeds_context",
            "prefill_tokens": len(prefix_ids),
            "decode_tokens": 0,
            "forward_calls": [],
            "token_selection_and_d2h_wall_ms": 0.0,
        }

    device = model.device
    current_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(current_ids)
    cache = None
    generated: list[int] = []
    events: list[tuple[torch.cuda.Event, torch.cuda.Event, int]] = []
    finish_reason = "context_limit"
    token_selection_and_d2h_wall_ms = 0.0

    for _ in range(available):
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        try:
            started.record()
            outputs = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            finished.record()
            events.append((started, finished, int(current_ids.shape[-1])))
            cache = outputs.past_key_values
            selection_started = time.perf_counter()
            next_token = outputs.logits[:, -1, :].float().argmax(dim=-1)
            token = int(next_token.item())
            token_selection_and_d2h_wall_ms += (time.perf_counter() - selection_started) * 1000
            generated.append(token)
            if token in eos_token_ids:
                finish_reason = "eos"
                break
            current_ids = next_token[:, None]
            attention_mask = torch.cat(
                (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)),
                dim=-1,
            )
            del outputs
        except torch.OutOfMemoryError:
            finish_reason = "cuda_oom"
            break

    torch.cuda.synchronize(device)
    calls = [
        {"milliseconds": float(start.elapsed_time(end)), "input_tokens": length}
        for start, end, length in events
    ]
    del cache
    if finish_reason == "cuda_oom":
        torch.cuda.empty_cache()
    return generated, {
        "finish_reason": finish_reason,
        "prefill_tokens": len(prefix_ids),
        "decode_tokens": len(generated),
        "forward_calls": calls,
        "token_selection_and_d2h_wall_ms": token_selection_and_d2h_wall_ms,
    }


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def hotpot_scores(prediction: str, reference: str) -> tuple[float, float]:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    exact = float(prediction_tokens == reference_tokens)
    if not prediction_tokens or not reference_tokens:
        return exact, float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return exact, 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return exact, 2 * precision * recall / (precision + recall)


def extract_gsm_answer(text: str) -> str:
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if not matches:
        matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return matches[-1].replace(",", "").strip().rstrip(".") if matches else ""


def score_prediction(
    dataset: str,
    prediction: str,
    record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if dataset == "gsm8k":
        parsed = extract_gsm_answer(prediction)
        reference = extract_gsm_answer(record["answer"])
        return {"parsed_prediction": parsed, "exact_match": float(parsed == reference)}, None
    if dataset == "hotpotqa":
        exact, f1 = hotpot_scores(prediction, record["answer"])
        return {
            "parsed_prediction": normalize_answer(prediction),
            "exact_match": exact,
            "f1": f1,
        }, None
    if dataset == "humaneval":
        program, candidate_mode = build_program(prediction, record)
        execution = execute(program, 6.0)
        execution["candidate_mode"] = candidate_mode
        return {"pass_at_1": float(execution["passed"])}, execution
    raise ValueError(dataset)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def quality_statistics(metric_values: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, values in metric_values.items():
        summary = distribution(values)
        mean = float(summary["mean"])
        count = len(values)
        if name in {"exact_match", "pass_at_1"} and all(value in {0.0, 1.0} for value in values):
            z = 1.959963984540054
            denominator = 1.0 + z * z / count
            center = (mean + z * z / (2 * count)) / denominator
            half_width = z * math.sqrt(mean * (1 - mean) / count + z * z / (4 * count * count)) / denominator
            low, high = max(0.0, center - half_width), min(1.0, center + half_width)
            method = "wilson_95"
        elif count > 1:
            half_width = 1.959963984540054 * statistics.stdev(values) / math.sqrt(count)
            low, high = max(0.0, mean - half_width), min(1.0, mean + half_width)
            method = "normal_approximation_95"
        else:
            low = high = mean
            method = "single_observation"
        result[name] = {
            **summary,
            "ci95_low": low,
            "ci95_high": high,
            "ci95_method": method,
        }
    return result


def numeric_fields(row: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    }


def add_values(destination: dict[str, list[float]], row: dict[str, Any]) -> None:
    for name, value in numeric_fields(row).items():
        destination[name].append(value)


def prior_state(predictions_path: Path):
    completed: set[str] = set()
    metric_values: dict[str, list[float]] = defaultdict(list)
    timing_values: dict[str, list[float]] = defaultdict(list)
    communication_values: dict[str, list[float]] = defaultdict(list)
    token_values: dict[str, list[float]] = defaultdict(list)
    finish_reasons = {"sender": Counter(), "receiver": Counter()}
    if not predictions_path.exists():
        return completed, metric_values, timing_values, communication_values, token_values, finish_reasons, 0
    with predictions_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["record_id"] in completed:
                raise ValueError(f"Duplicate prior result: {row['record_id']}")
            if row.get("result_schema_version") != RESULT_SCHEMA_VERSION:
                raise ValueError(
                    f"Result schema mismatch in {predictions_path}: "
                    f"{row.get('result_schema_version')} != {RESULT_SCHEMA_VERSION}"
                )
            completed.add(row["record_id"])
            add_values(metric_values, row["metrics"])
            add_values(timing_values, row["timing_ms"])
            add_values(communication_values, row["communication_bytes"])
            add_values(token_values, row["token_counts"])
            finish_reasons["sender"][row["termination"]["sender_finish_reason"]] += 1
            finish_reasons["receiver"][row["termination"]["receiver_finish_reason"]] += 1
    return (
        completed,
        metric_values,
        timing_values,
        communication_values,
        token_values,
        finish_reasons,
        len(completed),
    )


def output_pair_name(pair: str, cfg) -> str:
    sender = cfg.sender.model_id.split("/")[-1].lower().replace("-", "_")
    receiver = cfg.receiver.model_id.split("/")[-1].lower().replace("-", "_")
    return f"{pair.lower()}_sender_{sender}__receiver_{receiver}"


def expected_and_excluded(dataset: str, protocol: str) -> tuple[int, int]:
    total = 0
    excluded = 0
    for record in records_for(dataset):
        if protocol == "P2" and not record["p2_eligible"]:
            excluded += 1
            continue
        total += 1
    return total, excluded


def validate_bundle(bundle: dict[str, Any], pair: str) -> None:
    expected = {
        f"{pair}__{dataset}__{protocol}__{path}"
        for dataset, protocol, path in CONDITIONS
    }
    active = {
        condition["condition_id"]
        for condition in bundle["active_conditions"]
        if condition["pair"] == pair
    }
    if active != expected:
        raise ValueError(f"Prompt bundle conditions differ: active={sorted(active)}, expected={sorted(expected)}")
    if bundle["chat_template"] != {"add_generation_prompt": True, "enable_thinking": False}:
        raise ValueError(f"Unexpected chat template policy: {bundle['chat_template']}")


def runtime_environment() -> dict[str, Any]:
    import pyarrow
    import transformers

    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unknown").split(",")[0]
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().split(", ")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pyarrow": pyarrow.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "attention_backend": "torch_sdpa",
        "generation_backend": "custom_greedy_dynamic_cache",
        "timing_backends": {
            "model_forward": "torch.cuda.Event",
            "cpu_and_wall": "time.perf_counter",
        },
        "batch_size": 1,
        "logical_cuda_device": str(torch.cuda.current_device()),
        "physical_cuda_device": int(query[0]),
        "gpu_uuid": query[1],
        "gpu_name": query[2],
        "nvidia_driver": query[3],
        "gpu_total_memory_mib": int(query[4]),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def model_deployment(model, spec, role: str) -> dict[str, Any]:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return {
        "role": role,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "source_path": spec.path,
        "parameter_count_actual": parameter_count,
        "parameter_count_declared": spec.parameter_count,
        "weight_bytes_in_loaded_dtype": weight_bytes,
        "frozen": True,
        "deployment": "cuda:0",
    }


def build_summary(
    *,
    condition_id: str,
    pair: str,
    cfg,
    dataset: str,
    protocol: str,
    path: str,
    total: int,
    expected_rows: int,
    source_dataset_rows: int,
    excluded_rows: int,
    processed_this_run: int,
    metric_values: dict[str, list[float]],
    timing_values: dict[str, list[float]],
    communication_values: dict[str, list[float]],
    token_values: dict[str, list[float]],
    finish_reasons: dict[str, Counter],
    prompt_bundle: dict[str, Any],
    prompt_zip: Path,
    prompt_zip_sha256: str,
    predictions_path: Path,
    environment: dict[str, Any],
    deployments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "method": "T2T",
        "experiment_stage": "stage1_unified_test_and_base_overhead",
        "condition_id": condition_id,
        "pair": pair,
        "direction": cfg.direction,
        "dataset": dataset,
        "protocol": protocol,
        "path": path,
        "completed_rows": total,
        "expected_rows": expected_rows,
        "source_dataset_rows": source_dataset_rows,
        "protocol_ineligible_rows_excluded": excluded_rows,
        "processed_rows_this_run": processed_this_run,
        "quality_statistics": quality_statistics(metric_values),
        "relative_to_same_condition_t2t_baseline": {
            "status": "self_is_baseline",
            "delta": 0.0,
            "ci95": [0.0, 0.0],
        },
        "timing_statistics_ms": {
            name: distribution(values) for name, values in timing_values.items()
        },
        "communication_statistics_bytes": {
            name: distribution(values) for name, values in communication_values.items()
        },
        "token_statistics": {
            name: distribution(values) for name, values in token_values.items()
        },
        "termination_counts": {
            role: dict(counts) for role, counts in finish_reasons.items()
        },
        "experiment_configuration": {
            "weights": "frozen pretrained base models; T2T has no trained conversion weights",
            "training_source": "not_applicable_to_text_baseline",
            "K": None,
            "k": None,
            "d": None,
            "transmission_position": "after_sender_text_decode",
            "communication_path": path,
            "test_split": "GSM8K test / HotpotQA bridge validation / HumanEval test",
        },
        "module_parameters": {
            "pooling": 0,
            "projection": 0,
            "conversion": 0,
            "fusion": 0,
            "deduplicated_trainable_total": 0,
            "deployment": "no learned communication module; sender and receiver are frozen base models",
            "base_models_excluded_from_module_total": deployments,
        },
        "communication_representation": {
            "actual_payload": "canonical UTF-8 JSON wire envelope containing metadata and generated_text",
            "receiver_model_input": "canonical UTF-8 JSON containing generated_text only",
            "tensor_shape": None,
            "tensor_dtype": None,
            "text_encoding": "UTF-8",
            "metadata_enters_model_prompt": False,
            "network_transport_time_included": False,
        },
        "operation_deployment": {
            "upstream": {
                "prompt_assembly_and_tokenization": "CPU",
                "prefill": "physical GPU 0",
                "text_message_decode": "physical GPU 0",
                "token_selection_and_device_to_host": "GPU 0 plus synchronous D2H scalar reads",
                "detokenization_and_json_packaging": "CPU",
            },
            "downstream": {
                "json_unpack_and_prompt_assembly": "CPU",
                "local_and_message_prefill": "physical GPU 0",
                "final_answer_decode": "physical GPU 0",
                "detokenization": "CPU",
            },
            "not_applicable_to_t2t": [
                "KV filtering",
                "KV pooling",
                "projection",
                "cache conversion",
                "cache restoration",
                "cache broadcast",
                "cache fusion",
            ],
        },
        "prompt": {
            "archive": str(prompt_zip),
            "zip_sha256": prompt_zip_sha256,
            "schema": prompt_bundle["schema"],
            "version": prompt_bundle["version"],
            "source_frozen_task_bundle_sha256": prompt_bundle["source_frozen_task_bundle_sha256"],
            "archive_status_at_evaluation": prompt_bundle["status"],
            "chat_template": prompt_bundle["chat_template"],
            "communication_serialization": prompt_bundle["communication"]["json_serialization"],
        },
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "enable_thinking": False,
            "artificial_token_limit": None,
            "hard_ceiling": "each model's max_position_embeddings",
        },
        "measurement_environment": environment,
        "model_deployments": deployments,
        "predictions": str(predictions_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=sorted(PAIR_CONFIGS), required=True)
    parser.add_argument("--prompt-zip", type=Path, default=DEFAULT_PROMPT_ZIP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--only-condition")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    prompt_bundle = load_prompt_bundle(args.prompt_zip)
    validate_bundle(prompt_bundle, args.pair)
    selected = []
    for dataset, protocol, path in CONDITIONS:
        condition_id = f"{args.pair}__{dataset}__{protocol}__{path}"
        if args.only_condition is None or args.only_condition == condition_id:
            selected.append((dataset, protocol, path, condition_id))
    if not selected:
        raise ValueError(f"No matching condition: {args.only_condition}")

    inventory = {}
    for dataset, protocol, path, condition_id in selected:
        expected, excluded = expected_and_excluded(dataset, protocol)
        inventory[condition_id] = {"expected_rows": expected, "excluded_rows": excluded}
    if args.validate_only:
        print(json.dumps({
            "pair": args.pair,
            "prompt_schema": prompt_bundle["schema"],
            "prompt_version": prompt_bundle["version"],
            "prompt_zip_sha256": sha256_file(args.prompt_zip),
            "conditions": inventory,
        }, indent=2, ensure_ascii=False))
        return

    cfg = load_config(PAIR_CONFIGS[args.pair])
    set_seed(cfg.training.seed)
    dtype = parse_dtype(cfg.adapter.dtype)
    sender_tokenizer = load_tokenizer(cfg.sender, local_files_only=True)
    receiver_tokenizer = load_tokenizer(cfg.receiver, local_files_only=True)
    sender = load_model(cfg.sender, "cuda:0", dtype=dtype, local_files_only=True)
    receiver = load_model(cfg.receiver, "cuda:0", dtype=dtype, local_files_only=True)
    environment = runtime_environment()
    deployments = {
        "sender": model_deployment(sender, cfg.sender, "upstream"),
        "receiver": model_deployment(receiver, cfg.receiver, "downstream"),
    }
    eos_sender = {sender_tokenizer.eos_token_id} if sender_tokenizer.eos_token_id is not None else set()
    eos_receiver = {receiver_tokenizer.eos_token_id} if receiver_tokenizer.eos_token_id is not None else set()
    pair_output = args.output_root / output_pair_name(args.pair, cfg)
    pair_output.mkdir(parents=True, exist_ok=True)
    prompt_zip_sha256 = sha256_file(args.prompt_zip)

    for dataset, protocol, path, condition_id in selected:
        task_prompt = prompt_bundle["per_task"][dataset]
        condition_output = pair_output / dataset / f"{protocol}_{path}"
        condition_output.mkdir(parents=True, exist_ok=True)
        predictions_path = condition_output / "predictions.jsonl"
        summary_path = condition_output / "summary.json"
        (
            completed,
            metric_values,
            timing_values,
            communication_values,
            token_values,
            finish_reasons,
            total,
        ) = prior_state(predictions_path)
        processed_this_run = 0
        expected_rows, excluded_rows = expected_and_excluded(dataset, protocol)
        effective_expected = min(expected_rows, args.limit) if args.limit is not None else expected_rows

        with predictions_path.open("a", encoding="utf-8") as output:
            eligible_ordinal = 0
            for record in records_for(dataset):
                if protocol == "P2" and not record["p2_eligible"]:
                    continue
                if args.limit is not None and eligible_ordinal >= args.limit:
                    break
                eligible_ordinal += 1
                if record["record_id"] in completed:
                    continue

                torch.cuda.reset_peak_memory_stats()
                inference_started = time.perf_counter()
                sender_assembly_started = time.perf_counter()
                sender_view, receiver_view = protocol_views(record, protocol, task_prompt["p0"])
                sender_user = sender_view + "\n\n" + task_prompt["sender_task"]
                if protocol == "P4":
                    sender_user += "\n" + task_prompt["restatement_instruction"]
                sender_messages = [
                    {"role": "system", "content": task_prompt["sender_system"]},
                    {"role": "user", "content": sender_user},
                ]
                sender_prefix = apply_chat_template(sender_tokenizer, sender_messages)
                sender_visible_input_tokens = len(
                    sender_tokenizer.encode(sender_view, add_special_tokens=False)
                )
                sender_prompt_assembly_cpu_ms = (
                    time.perf_counter() - sender_assembly_started
                ) * 1000

                torch.cuda.synchronize()
                sender_generation_started = time.perf_counter()
                sender_ids, sender_usage = generate_to_context_limit(
                    sender,
                    sender_prefix,
                    context_limit=cfg.sender.max_position_embeddings,
                    eos_token_ids=eos_sender,
                )
                sender_generation_wall_ms = (
                    time.perf_counter() - sender_generation_started
                ) * 1000
                sender_detokenize_started = time.perf_counter()
                sender_text = sender_tokenizer.decode(sender_ids, skip_special_tokens=True).strip()
                sender_detokenize_cpu_ms = (
                    time.perf_counter() - sender_detokenize_started
                ) * 1000

                packaging_started = time.perf_counter()
                communication_json = json.dumps(
                    {"generated_text": sender_text},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                communication_metadata = {
                    "condition_id": condition_id,
                    "sample_id": record["record_id"],
                    "schema_version": 1,
                }
                metadata_json = json.dumps(
                    communication_metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                wire_payload_json = json.dumps(
                    {"metadata": communication_metadata, "generated_text": sender_text},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                sender_packaging_cpu_ms = (time.perf_counter() - packaging_started) * 1000

                unpack_started = time.perf_counter()
                unpacked_generated_text = json.loads(communication_json)["generated_text"]
                if unpacked_generated_text != sender_text:
                    raise RuntimeError("Communication JSON round trip changed generated_text")
                receiver_unpack_cpu_ms = (time.perf_counter() - unpack_started) * 1000

                receiver_assembly_started = time.perf_counter()
                receiver_messages = [
                    {"role": "system", "content": task_prompt["receiver_system"]},
                    {"role": "user", "content": receiver_view},
                    {
                        "role": "user",
                        "content": (
                            f"<COMMUNICATION>\n{communication_json}\n</COMMUNICATION>\n\n"
                            f"{task_prompt['receiver_task']}"
                        ),
                    },
                ]
                receiver_prefix = apply_chat_template(receiver_tokenizer, receiver_messages)
                receiver_local_input_tokens = len(
                    receiver_tokenizer.encode(receiver_view, add_special_tokens=False)
                )
                receiver_communication_tokens = len(
                    receiver_tokenizer.encode(communication_json, add_special_tokens=False)
                )
                receiver_prompt_assembly_cpu_ms = (
                    time.perf_counter() - receiver_assembly_started
                ) * 1000
                receiver_generation_started = time.perf_counter()
                receiver_ids, receiver_usage = generate_to_context_limit(
                    receiver,
                    receiver_prefix,
                    context_limit=cfg.receiver.max_position_embeddings,
                    eos_token_ids=eos_receiver,
                )
                receiver_generation_wall_ms = (
                    time.perf_counter() - receiver_generation_started
                ) * 1000
                torch.cuda.synchronize()
                receiver_detokenize_started = time.perf_counter()
                prediction = receiver_tokenizer.decode(receiver_ids, skip_special_tokens=True).strip()
                receiver_detokenize_cpu_ms = (
                    time.perf_counter() - receiver_detokenize_started
                ) * 1000
                end_to_end_inference_wall_ms = (time.perf_counter() - inference_started) * 1000
                scoring_started = time.perf_counter()
                metrics, execution = score_prediction(dataset, prediction, record)
                quality_scoring_wall_ms = (time.perf_counter() - scoring_started) * 1000

                sender_prefill_ms = sender_usage["forward_calls"][0]["milliseconds"] if sender_usage["forward_calls"] else 0.0
                sender_decode_ms = sum(call["milliseconds"] for call in sender_usage["forward_calls"][1:])
                receiver_prefill_ms = receiver_usage["forward_calls"][0]["milliseconds"] if receiver_usage["forward_calls"] else 0.0
                receiver_decode_ms = sum(call["milliseconds"] for call in receiver_usage["forward_calls"][1:])
                cuda_forward_sum = sender_prefill_ms + sender_decode_ms + receiver_prefill_ms + receiver_decode_ms
                expected_task_block = f"<TASK>\n{record['q']}\n</TASK>"
                exact_restatement = expected_task_block in sender_text if protocol == "P4" else None
                timing = {
                    "sender_prompt_assembly_cpu_ms": sender_prompt_assembly_cpu_ms,
                    "sender_prefill_cuda_ms": sender_prefill_ms,
                    "sender_decode_cuda_ms": sender_decode_ms,
                    "sender_generation_wall_ms": sender_generation_wall_ms,
                    "sender_token_selection_and_d2h_wall_ms": sender_usage["token_selection_and_d2h_wall_ms"],
                    "sender_detokenize_cpu_ms": sender_detokenize_cpu_ms,
                    "sender_packaging_cpu_ms": sender_packaging_cpu_ms,
                    "receiver_unpack_cpu_ms": receiver_unpack_cpu_ms,
                    "receiver_prompt_assembly_cpu_ms": receiver_prompt_assembly_cpu_ms,
                    "receiver_prefill_cuda_ms": receiver_prefill_ms,
                    "receiver_decode_cuda_ms": receiver_decode_ms,
                    "receiver_generation_wall_ms": receiver_generation_wall_ms,
                    "receiver_token_selection_and_d2h_wall_ms": receiver_usage["token_selection_and_d2h_wall_ms"],
                    "receiver_detokenize_cpu_ms": receiver_detokenize_cpu_ms,
                    "end_to_end_inference_wall_ms": end_to_end_inference_wall_ms,
                    "end_to_end_minus_cuda_forward_ms": max(
                        0.0, end_to_end_inference_wall_ms - cuda_forward_sum
                    ),
                    "quality_scoring_wall_ms": quality_scoring_wall_ms,
                }
                communication_bytes = {
                    "sender_generated_text_utf8_bytes": len(sender_text.encode("utf-8")),
                    "receiver_prompt_communication_json_utf8_bytes": len(communication_json.encode("utf-8")),
                    "wire_metadata_json_utf8_bytes": len(metadata_json.encode("utf-8")),
                    "actual_wire_payload_json_utf8_bytes": len(wire_payload_json.encode("utf-8")),
                    "logical_transmission_bytes": len(wire_payload_json.encode("utf-8")),
                    "supplemental_original_task_block_utf8_bytes": (
                        len(expected_task_block.encode("utf-8")) if exact_restatement else 0
                    ),
                    "required_missing_q_utf8_bytes": (
                        len(record["q"].encode("utf-8")) if protocol == "P4" else 0
                    ),
                    "source_original_kv_cache_bytes": 0,
                    "transmitted_tensor_bytes": 0,
                }
                token_counts = {
                    "sender_visible_input_tokens": sender_visible_input_tokens,
                    "sender_prompt_tokens": len(sender_prefix),
                    "sender_output_tokens": len(sender_ids),
                    "receiver_local_input_tokens": receiver_local_input_tokens,
                    "receiver_communication_tokens": receiver_communication_tokens,
                    "receiver_prompt_tokens": len(receiver_prefix),
                    "receiver_output_tokens": len(receiver_ids),
                }
                result = {
                    "result_schema_version": RESULT_SCHEMA_VERSION,
                    "method": "T2T",
                    "record_id": record["record_id"],
                    "task_id": record.get("task_id"),
                    "condition_id": condition_id,
                    "pair": args.pair,
                    "direction": cfg.direction,
                    "dataset": dataset,
                    "protocol": protocol,
                    "path": path,
                    "prediction": prediction,
                    "reference": record.get("answer"),
                    "metrics": metrics,
                    "humaneval_execution": execution,
                    "sender_generated_text": sender_text,
                    "communication_metadata": communication_metadata,
                    "token_counts": token_counts,
                    "input_bytes": {
                        "question_utf8_bytes": len(record["q"].encode("utf-8")),
                        "sender_visible_view_utf8_bytes": len(sender_view.encode("utf-8")),
                        "receiver_local_view_utf8_bytes": len(receiver_view.encode("utf-8")),
                        "sender_full_prompt_content_utf8_bytes": sum(
                            len(message["content"].encode("utf-8")) for message in sender_messages
                        ),
                        "receiver_full_prompt_content_utf8_bytes": sum(
                            len(message["content"].encode("utf-8")) for message in receiver_messages
                        ),
                    },
                    "communication_bytes": communication_bytes,
                    "timing_ms": timing,
                    "termination": {
                        "sender_finish_reason": sender_usage["finish_reason"],
                        "receiver_finish_reason": receiver_usage["finish_reason"],
                        "sender_context_capacity_tokens": cfg.sender.max_position_embeddings,
                        "receiver_context_capacity_tokens": cfg.receiver.max_position_embeddings,
                        "sender_forward_calls": len(sender_usage["forward_calls"]),
                        "receiver_forward_calls": len(receiver_usage["forward_calls"]),
                    },
                    "protocol_compliance": {
                        "p4_exact_task_restatement_present": exact_restatement,
                        "p2_source_support_paragraphs": 1 if protocol == "P2" else None,
                        "p2_source_distractor_paragraphs": 4 if protocol == "P2" else None,
                        "p2_receiver_support_paragraphs": 1 if protocol == "P2" else None,
                        "p2_receiver_distractor_paragraphs": 4 if protocol == "P2" else None,
                    },
                    "experiment_configuration": {
                        "K": None,
                        "k": None,
                        "d": None,
                        "transmission_position": "after_sender_text_decode",
                        "batch_size": 1,
                        "dtype": cfg.adapter.dtype,
                        "enable_thinking": False,
                        "do_sample": False,
                    },
                    "memory": {
                        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                    },
                }
                output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()
                completed.add(record["record_id"])
                total += 1
                processed_this_run += 1
                add_values(metric_values, metrics)
                add_values(timing_values, timing)
                add_values(communication_values, communication_bytes)
                add_values(token_values, token_counts)
                finish_reasons["sender"][sender_usage["finish_reason"]] += 1
                finish_reasons["receiver"][receiver_usage["finish_reason"]] += 1

                if processed_this_run == 1 or total % 10 == 0 or total == effective_expected:
                    atomic_write_json(summary_path, build_summary(
                        condition_id=condition_id,
                        pair=args.pair,
                        cfg=cfg,
                        dataset=dataset,
                        protocol=protocol,
                        path=path,
                        total=total,
                        expected_rows=effective_expected,
                        source_dataset_rows=expected_rows + excluded_rows,
                        excluded_rows=excluded_rows,
                        processed_this_run=processed_this_run,
                        metric_values=metric_values,
                        timing_values=timing_values,
                        communication_values=communication_values,
                        token_values=token_values,
                        finish_reasons=finish_reasons,
                        prompt_bundle=prompt_bundle,
                        prompt_zip=args.prompt_zip,
                        prompt_zip_sha256=prompt_zip_sha256,
                        predictions_path=predictions_path,
                        environment=environment,
                        deployments=deployments,
                    ))
                print(json.dumps({
                    "condition_id": condition_id,
                    "record_id": record["record_id"],
                    "completed": total,
                    "expected": effective_expected,
                    "metrics": metrics,
                    "sender_finish": sender_usage["finish_reason"],
                    "receiver_finish": receiver_usage["finish_reason"],
                }, ensure_ascii=False), flush=True)

        atomic_write_json(summary_path, build_summary(
            condition_id=condition_id,
            pair=args.pair,
            cfg=cfg,
            dataset=dataset,
            protocol=protocol,
            path=path,
            total=total,
            expected_rows=effective_expected,
            source_dataset_rows=expected_rows + excluded_rows,
            excluded_rows=excluded_rows,
            processed_this_run=processed_this_run,
            metric_values=metric_values,
            timing_values=timing_values,
            communication_values=communication_values,
            token_values=token_values,
            finish_reasons=finish_reasons,
            prompt_bundle=prompt_bundle,
            prompt_zip=args.prompt_zip,
            prompt_zip_sha256=prompt_zip_sha256,
            predictions_path=predictions_path,
            environment=environment,
            deployments=deployments,
        ))
        if total != effective_expected:
            raise RuntimeError(f"Incomplete condition {condition_id}: {total} != {effective_expected}")

    atomic_write_json(pair_output / "pair_complete.json", {
        "pair": args.pair,
        "direction": cfg.direction,
        "conditions": [condition_id for _, _, _, condition_id in selected],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prompt_zip_sha256": prompt_zip_sha256,
    })


if __name__ == "__main__":
    main()
