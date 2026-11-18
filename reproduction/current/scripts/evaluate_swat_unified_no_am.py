#!/usr/bin/env python3
"""Evaluate a trained SwaT renderer on the unified Stage-1 protocols, with optional AM."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("SUBPROBLEM4_PROJECT_ROOT", REPO_ROOT / "reproduction" / "current"))
SWAT_ROOT = Path(os.environ.get("SWAT_ROOT", REPO_ROOT))
if str(SWAT_ROOT) not in sys.path:
    sys.path.insert(0, str(SWAT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_t2t_frozen_prompts import (  # noqa: E402
    atomic_write_json,
    distribution,
    expected_and_excluded,
    load_prompt_bundle,
    protocol_views,
    quality_statistics,
    records_for,
    score_prediction,
    sha256_file,
)
from multiagent.dataset.c2c import chat_prompt_ids, generation_prompt_ids  # noqa: E402
from multiagent.dataset.c2c_cross_family import encode_context_pair  # noqa: E402
from multiagent.dataset.c2c_types import C2CExample  # noqa: E402
from multiagent.evaluation.native_am import generate_from_beta_cache  # noqa: E402
from multiagent.evaluation.renderer_am import generate_attention_matching_renderer  # noqa: E402
from multiagent.evaluation.utils import (  # noqa: E402
    eos_ids,
    load_head_budget_shares,
    load_model,
    load_tokenizer,
    seed_attention_matching_context,
)
from multiagent.model.attention_matching import AttentionMatchingConfig  # noqa: E402
from multiagent.model.cache_renderer import (  # noqa: E402
    flatten_sender_cache,
    load_renderer_checkpoint,
    pred_to_cache,
)
from multiagent.model.query_transfer import load_query_transfer  # noqa: E402
from multiagent.model.unaware_comm import full_sender_cache, tuple_to_dynamic_cache  # noqa: E402


PAIR_RUNS = {
    "M1": "m1_qwen3_0.6b_to_qwen3_1.7b",
    "M2": "m2_qwen3_0.6b_to_qwen3_4b",
    "M3": "m3_qwen3_4b_to_qwen3_0.6b",
}
CONDITIONS = (
    ("gsm8k", "P1"),
    ("gsm8k", "P4"),
    ("hotpotqa", "P1"),
    ("hotpotqa", "P2"),
    ("hotpotqa", "P3"),
    ("hotpotqa", "P4"),
    ("humaneval", "P1"),
    ("humaneval", "P4"),
)
TASK_MAX_NEW_TOKENS = {"gsm8k": 512, "hotpotqa": 64, "humaneval": 512}
RESULT_SCHEMA_VERSION = 1
DEFAULT_PROMPT_ZIP = REPO_ROOT / "reproduction" / "prompts" / "提示词模板.zip"
DEFAULT_OUTPUT_ROOT = (
    SWAT_ROOT / "outputs/subproblem4_original_recipe/evaluations/swat_full_cache_no_am_v1"
)


def numeric_fields(row: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in row.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }


def add_values(destination: dict[str, list[float]], row: dict[str, Any]) -> None:
    for name, value in numeric_fields(row).items():
        destination[name].append(value)


def prior_state(path: Path):
    completed: set[str] = set()
    metrics: dict[str, list[float]] = defaultdict(list)
    timings: dict[str, list[float]] = defaultdict(list)
    communication: dict[str, list[float]] = defaultdict(list)
    tokens: dict[str, list[float]] = defaultdict(list)
    finishes: Counter = Counter()
    if not path.exists():
        return completed, metrics, timings, communication, tokens, finishes
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("result_schema_version") != RESULT_SCHEMA_VERSION:
                raise ValueError(f"Result schema mismatch in {path}")
            record_id = row["record_id"]
            if record_id in completed:
                raise ValueError(f"Duplicate record in {path}: {record_id}")
            completed.add(record_id)
            add_values(metrics, row["metrics"])
            add_values(timings, row["timing_ms"])
            add_values(communication, row["communication_bytes"])
            add_values(tokens, row["token_counts"])
            finishes[row["termination"]["finish_reason"]] += 1
    return completed, metrics, timings, communication, tokens, finishes


def cuda_timed(callable_: Callable[[], Any]) -> tuple[Any, float, float]:
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    value = callable_()
    finish.record()
    torch.cuda.synchronize()
    return value, float(start.elapsed_time(finish)), (time.perf_counter() - wall_start) * 1000


def tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        return tensor_bytes(value.key_cache) + tensor_bytes(value.value_cache)
    return 0


def model_metadata(model, path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "weight_bytes_in_loaded_dtype": sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
        "frozen": True,
    }


def runtime_environment() -> dict[str, Any]:
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    gpu = subprocess.run(
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
    import transformers

    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_index_physical": int(gpu[0]),
        "gpu_uuid": gpu[1],
        "gpu_name": gpu[2],
        "driver": gpu[3],
        "gpu_total_memory_mib": int(gpu[4]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": 1,
        "timing": "CUDA events for GPU stages; perf_counter for wall/CPU stages",
    }


@torch.inference_mode()
def generate_one(
    *,
    renderer,
    renderer_payload: dict[str, Any],
    sender_model,
    receiver_model,
    sender_tokenizer,
    receiver_tokenizer,
    sender_messages: list[dict[str, str]],
    receiver_suffix_messages: list[dict[str, str]] | None,
    max_length: int,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    position_mode = str(renderer_payload.get("position_mode") or "interp")
    token_alignment = getattr(renderer, "token_alignment", "native")
    causal = bool(renderer_payload.get("args", {}).get("causal_cross_attention"))

    assembly_started = time.perf_counter()
    context = encode_context_pair(
        sender_tokenizer,
        receiver_tokenizer,
        sender_messages,
        max_length,
        position_mode,
        token_alignment=token_alignment,
    )
    if not context["sender_ids"] or not context["receiver_ids"]:
        raise ValueError("SwaT context encoder produced an empty prompt")
    if receiver_suffix_messages is None:
        decode_ids = generation_prompt_ids(receiver_tokenizer)
        suffix_mode = "generation_prompt_only"
    else:
        decode_ids = chat_prompt_ids(
            receiver_tokenizer,
            receiver_suffix_messages,
            max_length,
            add_generation_prompt=True,
        )
        suffix_mode = "local_suffix_chat_prompt"
    prompt_assembly_cpu_ms = (time.perf_counter() - assembly_started) * 1000

    sender_device = str(next(sender_model.parameters()).device)
    sender_ids = torch.tensor([context["sender_ids"]], dtype=torch.long, device=sender_device)
    sender_mask = torch.tensor(
        [context.get("sender_valid", [1] * len(context["sender_ids"]))],
        dtype=torch.long,
        device=sender_device,
    )
    sender_positions = torch.arange(sender_ids.shape[1], device=sender_device).unsqueeze(0)

    def sender_prefill():
        return sender_model(
            input_ids=sender_ids,
            attention_mask=sender_mask,
            position_ids=sender_positions,
            use_cache=True,
            past_key_values=full_sender_cache(),
        ).past_key_values

    sender_cache, sender_prefill_cuda_ms, sender_prefill_wall_ms = cuda_timed(sender_prefill)
    def flatten_cache():
        return flatten_sender_cache(sender_model, sender_cache, sender_positions).to(device)

    memory, flatten_cuda_ms, flatten_wall_ms = cuda_timed(flatten_cache)
    # Transformers 5 DynamicCache no longer exposes key_cache/value_cache on the
    # top-level object.  The flattened tensor contains every source-layer K/V
    # element exactly once, so its storage size is the logical source KV size.
    source_cache_bytes = tensor_bytes(memory)
    del sender_cache

    receiver_ids = torch.tensor([context["receiver_ids"]], dtype=torch.long, device=device)
    receiver_valid = torch.tensor(
        [context.get("receiver_valid", [1] * len(context["receiver_ids"]))],
        dtype=torch.long,
        device=device,
    )
    receiver_positions = torch.arange(receiver_ids.shape[1], device=device).unsqueeze(0)
    virtual = torch.from_numpy(context["virtual_positions"][None].astype("float32")).to(device)
    limits = None
    if causal:
        limits = torch.from_numpy(context["cross_causal_limits"][None]).to(device)

    def render_cache():
        embeds = receiver_model.get_input_embeddings()(receiver_ids)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = renderer(
                embeds,
                receiver_positions,
                memory,
                virtual,
                None if token_alignment == "native" else sender_mask.to(device),
                limits,
                **(
                    {"receiver_mask": receiver_valid}
                    if token_alignment != "native"
                    else {}
                ),
            )
        return pred_to_cache(
            receiver_model,
            prediction,
            receiver_positions,
            next(receiver_model.parameters()).dtype,
        )

    pred_cache, conversion_cuda_ms, conversion_wall_ms = cuda_timed(render_cache)
    transformed_cache_bytes = tensor_bytes(pred_cache)
    first_key = pred_cache[0][0]
    cache_shape = {
        "layers": len(pred_cache),
        "key_per_layer": list(first_key.shape),
        "value_per_layer": list(pred_cache[0][1].shape),
        "dtype": str(first_key.dtype).replace("torch.", ""),
    }
    del memory

    metadata = {
        "schema_version": 1,
        "logical_prefix_tokens": int(receiver_ids.shape[1]),
        "cache": cache_shape,
        "suffix_mode": suffix_mode,
    }
    metadata_bytes = len(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    cache = tuple_to_dynamic_cache(pred_cache, receiver_model.config)
    receiver_len = int(receiver_ids.shape[1])
    attention_len = receiver_len
    suffix_prefill_cuda_ms = 0.0
    suffix_prefill_wall_ms = 0.0
    current_id = decode_ids[-1]
    if len(decode_ids) > 1:
        prefix = torch.tensor([decode_ids[:-1]], dtype=torch.long, device=device)
        attention_len += prefix.shape[1]

        def suffix_prefill():
            return receiver_model(
                input_ids=prefix,
                attention_mask=torch.cat(
                    [receiver_valid, receiver_valid.new_ones((1, attention_len - receiver_len))],
                    dim=1,
                ),
                position_ids=torch.arange(
                    receiver_len, receiver_len + prefix.shape[1], device=device
                ).unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
            ).past_key_values

        cache, suffix_prefill_cuda_ms, suffix_prefill_wall_ms = cuda_timed(suffix_prefill)

    output_ids: list[int] = []
    forward_calls: list[dict[str, float | int]] = []
    finish_reason = "cutoff"
    stop_ids = eos_ids(receiver_tokenizer)
    next_position = receiver_len + len(decode_ids) - 1
    decode_wall_started = time.perf_counter()
    for _ in range(max_new_tokens):
        attention_len += 1
        input_token = torch.tensor([[current_id]], dtype=torch.long, device=device)
        attention_mask = torch.cat(
            [receiver_valid, receiver_valid.new_ones((1, attention_len - receiver_len))],
            dim=1,
        )
        position_ids = torch.tensor([[next_position]], dtype=torch.long, device=device)

        def decode_forward():
            return receiver_model(
                input_ids=input_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )

        output, elapsed_ms, wall_ms = cuda_timed(decode_forward)
        forward_calls.append({"cuda_ms": elapsed_ms, "wall_ms": wall_ms, "input_tokens": 1})
        next_id = int(output.logits[:, -1, :].argmax(dim=-1).item())
        cache = output.past_key_values
        if next_id in stop_ids:
            finish_reason = "eos"
            break
        output_ids.append(next_id)
        current_id = next_id
        next_position += 1
    decode_wall_ms = (time.perf_counter() - decode_wall_started) * 1000
    decode_cuda_ms = sum(float(call["cuda_ms"]) for call in forward_calls)
    prediction = receiver_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return {
        "prediction": prediction,
        "finish_reason": finish_reason,
        "token_counts": {
            "sender_transferred_prefix_tokens": int(sender_ids.shape[1]),
            "receiver_rendered_prefix_tokens": receiver_len,
            "receiver_suffix_prompt_tokens": len(decode_ids),
            "receiver_output_tokens": len(output_ids),
        },
        "communication_bytes": {
            "source_original_kv_cache_bytes": source_cache_bytes,
            "transformed_receiver_kv_cache_bytes": transformed_cache_bytes,
            "wire_metadata_json_utf8_bytes": metadata_bytes,
            "logical_transmission_bytes": transformed_cache_bytes + metadata_bytes,
            "am_auxiliary_bytes": 0,
        },
        "timing_ms": {
            "prompt_assembly_cpu_ms": prompt_assembly_cpu_ms,
            "upstream_sender_prefill_cuda_ms": sender_prefill_cuda_ms,
            "upstream_sender_prefill_wall_ms": sender_prefill_wall_ms,
            "upstream_cache_flatten_move_cuda_ms": flatten_cuda_ms,
            "upstream_cache_flatten_move_wall_ms": flatten_wall_ms,
            "upstream_swat_conversion_cuda_ms": conversion_cuda_ms,
            "upstream_swat_conversion_wall_ms": conversion_wall_ms,
            "downstream_suffix_prefill_cuda_ms": suffix_prefill_cuda_ms,
            "downstream_suffix_prefill_wall_ms": suffix_prefill_wall_ms,
            "downstream_decode_cuda_ms": decode_cuda_ms,
            "downstream_decode_wall_ms": decode_wall_ms,
        },
        "cache_shape": cache_shape,
        "suffix_mode": suffix_mode,
        "decode_forward_calls": forward_calls,
    }


@torch.inference_mode()
def generate_one_am(
    *,
    renderer,
    renderer_payload: dict[str, Any],
    sender_model,
    receiver_model,
    sender_tokenizer,
    receiver_tokenizer,
    query_transfer,
    am_config: AttentionMatchingConfig,
    dataset: str,
    protocol: str,
    record_id: str,
    record_index: int,
    sender_messages: list[dict[str, str]],
    receiver_suffix_messages: list[dict[str, str]] | None,
    max_length: int,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    """Run the official SwaT->AM path while retaining the unified evaluator's metrics."""
    position_mode = str(renderer_payload.get("position_mode") or "interp")
    token_alignment = getattr(renderer, "token_alignment", "native")
    if token_alignment != "native":
        raise ValueError("The official renderer+AM path only supports native token alignment")

    assembly_started = time.perf_counter()
    context = encode_context_pair(
        sender_tokenizer,
        receiver_tokenizer,
        sender_messages,
        max_length,
        position_mode,
        token_alignment=token_alignment,
    )
    if not context["sender_ids"] or not context["receiver_ids"]:
        raise ValueError("SwaT context encoder produced an empty prompt")
    sample = C2CExample(
        id=record_id,
        dataset=dataset,
        split="test",
        phase1_messages=sender_messages,
        sender_messages=sender_messages,
        receiver_messages=receiver_suffix_messages or [],
        target_text=None,
        prompt_mode="split_prompt" if receiver_suffix_messages is not None else "same_prompt_continuation",
        metadata={
            "context_id": f"{dataset}:{protocol}:{record_id}",
            "task": dataset,
            "protocol": protocol,
        },
    )
    prompt_assembly_cpu_ms = (time.perf_counter() - assembly_started) * 1000
    seed_attention_matching_context(42, f"{dataset}:{protocol}", record_index, sample, device)

    timings: dict[str, Any] = {}
    prediction, stop_reason, generated_tokens = generate_attention_matching_renderer(
        renderer=renderer,
        sender_model=sender_model,
        receiver_model=receiver_model,
        sender_tokenizer=sender_tokenizer,
        receiver_tokenizer=receiver_tokenizer,
        query_transfer=query_transfer,
        sample=sample,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        am_config=am_config,
        decode_fn=generate_from_beta_cache,
        causal_cross_attention=bool(
            renderer_payload.get("args", {}).get("causal_cross_attention")
        ),
        decode_prefix="question" if receiver_suffix_messages is not None else "generation",
        position_mode=position_mode,
        timings=timings,
    )
    counts = re.search(
        r"renderer\+am(\d+)active/(\d+)allocated/(\d+)recv\+(\d+)",
        stop_reason,
    )
    if counts is None:
        raise RuntimeError(f"Could not parse AM cache counts from stop reason: {stop_reason}")
    active_entries, allocated_entries, receiver_len, suffix_tokens = map(int, counts.groups())

    receiver_config = receiver_model.config.get_text_config()
    sender_config = sender_model.config.get_text_config()
    receiver_layers = int(receiver_config.num_hidden_layers)
    receiver_kv_heads = int(receiver_config.num_key_value_heads)
    receiver_head_dim = int(
        getattr(receiver_config, "head_dim", receiver_config.hidden_size // receiver_config.num_attention_heads)
    )
    sender_layers = int(sender_config.num_hidden_layers)
    sender_kv_heads = int(sender_config.num_key_value_heads)
    sender_head_dim = int(
        getattr(sender_config, "head_dim", sender_config.hidden_size // sender_config.num_attention_heads)
    )
    element_bytes = int(next(receiver_model.parameters()).element_size())
    sender_element_bytes = int(next(sender_model.parameters()).element_size())

    source_cache_bytes = (
        len(context["sender_ids"])
        * sender_layers
        * sender_kv_heads
        * sender_head_dim
        * 2
        * sender_element_bytes
    )
    full_transformed_bytes = (
        receiver_len
        * receiver_layers
        * receiver_kv_heads
        * receiver_head_dim
        * 2
        * element_bytes
    )
    compact_kv_bytes = active_entries * receiver_head_dim * 2 * element_bytes
    beta_bytes = active_entries * element_bytes
    allocated_runtime_bytes = allocated_entries * (2 * receiver_head_dim + 1) * element_bytes
    metadata = {
        "schema_version": 1,
        "logical_prefix_tokens": receiver_len,
        "active_entries": active_entries,
        "receiver_layers": receiver_layers,
        "receiver_kv_heads": receiver_kv_heads,
        "receiver_head_dim": receiver_head_dim,
        "target_ratio": am_config.target_ratio,
        "head_budget_mode": am_config.head_budget_mode,
        "suffix_mode": "local_suffix_chat_prompt" if receiver_suffix_messages is not None else "generation_prompt_only",
    }
    metadata_bytes = len(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    head_count = receiver_layers * receiver_kv_heads
    finish_reason = stop_reason.split(":", 1)[0]

    return {
        "prediction": prediction,
        "finish_reason": finish_reason,
        "stop_reason_detail": stop_reason,
        "token_counts": {
            "sender_transferred_prefix_tokens": len(context["sender_ids"]),
            "receiver_rendered_prefix_tokens": receiver_len,
            "receiver_suffix_prompt_tokens": suffix_tokens,
            "receiver_output_tokens": generated_tokens,
            "am_active_entries_total": active_entries,
            "am_allocated_entries_total": allocated_entries,
            "am_mean_active_positions_per_layer_head": active_entries / head_count,
            "am_retained_fraction": active_entries / max(1, receiver_len * head_count),
        },
        "communication_bytes": {
            "source_original_kv_cache_bytes": source_cache_bytes,
            "transformed_receiver_kv_cache_bytes_before_am": full_transformed_bytes,
            "am_compact_kv_bytes": compact_kv_bytes,
            "am_auxiliary_bytes": beta_bytes,
            "am_allocated_runtime_bytes": allocated_runtime_bytes,
            "wire_metadata_json_utf8_bytes": metadata_bytes,
            "logical_transmission_bytes": compact_kv_bytes + beta_bytes + metadata_bytes,
        },
        "timing_ms": {
            "prompt_assembly_cpu_ms": prompt_assembly_cpu_ms,
            "upstream_sender_prefill_wall_ms": float(timings.get("sender_prefill_ms", 0.0)),
            "upstream_swat_conversion_query_transfer_am_wall_ms": float(timings.get("sender_am_ms", 0.0)),
            "upstream_total_wall_ms": float(timings.get("sender_ms", 0.0)),
            "downstream_suffix_prefill_wall_ms": float(timings.get("receiver_prefill_ms", 0.0)),
            "downstream_decode_wall_ms": float(timings.get("receiver_decode_ms", 0.0)),
            "downstream_total_wall_ms": float(timings.get("receiver_ms", 0.0)),
        },
        "cache_shape": {
            "layers": receiver_layers,
            "kv_heads": receiver_kv_heads,
            "head_dim": receiver_head_dim,
            "dtype": str(next(receiver_model.parameters()).dtype).replace("torch.", ""),
            "active_entries": active_entries,
            "allocated_entries": allocated_entries,
        },
        "suffix_mode": metadata["suffix_mode"],
        "decode_forward_calls": [],
    }


def summary_payload(
    *,
    pair: str,
    direction: str,
    dataset: str,
    protocol: str,
    completed_rows: int,
    expected_rows: int,
    excluded_rows: int,
    processed_rows_this_run: int,
    metrics,
    timings,
    communication,
    tokens,
    finishes,
    checkpoint: Path,
    checkpoint_step: int | None,
    renderer_parameters: int,
    renderer_loaded_weight_bytes: int,
    receiver_embedding_parameters: int,
    environment: dict[str, Any],
    model_deployments: dict[str, Any],
    predictions_path: Path,
    prompt_zip: Path,
    prompt_bundle: dict[str, Any],
    am_enabled: bool = False,
    am_target_ratio: float | None = None,
    am_budget_label: str | None = None,
    am_head_budget_mode: str | None = None,
    am_head_budget_path: Path | None = None,
    query_transfer_checkpoint: Path | None = None,
    query_transfer_parameters: int = 0,
    query_transfer_loaded_weight_bytes: int = 0,
) -> dict[str, Any]:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "method": "SwaT+AM" if am_enabled else "SwaT",
        "experiment_stage": "stage1_unified_test_and_base_overhead",
        "compression": (
            f"Attention Matching target_ratio={am_target_ratio:g} ({am_budget_label})"
            if am_enabled
            else "none (AM disabled)"
        ),
        "pair": pair,
        "direction": direction,
        "dataset": dataset,
        "protocol": protocol,
        "completed_rows": completed_rows,
        "expected_rows": expected_rows,
        "protocol_ineligible_rows_excluded": excluded_rows,
        "processed_rows_this_run": processed_rows_this_run,
        "quality_statistics": quality_statistics(metrics),
        "timing_statistics_ms": {key: distribution(value) for key, value in timings.items()},
        "communication_statistics_bytes": {
            key: distribution(value) for key, value in communication.items()
        },
        "token_statistics": {key: distribution(value) for key, value in tokens.items()},
        "termination_counts": dict(finishes),
        "module_parameters": {
            "swat_renderer": renderer_parameters,
            "query_transfer": query_transfer_parameters,
            "deduplicated_trainable_total": renderer_parameters + query_transfer_parameters,
            "receiver_frozen_embedding_table_deployed_upstream": receiver_embedding_parameters,
            "receiver_embedding_excluded_from_trainable_total": True,
            "renderer_weight_bytes_in_loaded_float32": renderer_loaded_weight_bytes,
            "query_transfer_weight_bytes_in_loaded_float32": query_transfer_loaded_weight_bytes,
            "checkpoint_file_bytes_including_training_state": checkpoint.stat().st_size,
        },
        "weights": {
            "renderer_checkpoint": str(checkpoint),
            "renderer_step": checkpoint_step,
            "query_transfer_checkpoint": (
                str(query_transfer_checkpoint) if query_transfer_checkpoint is not None else None
            ),
            "base_models_frozen": True,
        },
        "communication_representation": {
            "payload": (
                "AM-compacted receiver-format K/V, one beta per active entry, plus JSON metadata"
                if am_enabled
                else "complete transformed receiver-format KV cache plus JSON metadata"
            ),
            "am_enabled": am_enabled,
            "am_target_ratio": am_target_ratio,
            "am_budget_label": am_budget_label,
            "am_head_budget_mode": am_head_budget_mode,
            "am_head_budget_path": (
                str(am_head_budget_path) if am_head_budget_path is not None else None
            ),
            "network_transport_time_included": False,
            "byte_accounting": (
                "active K/V vectors + active beta scalars + serialized metadata; padded runtime allocation reported separately"
                if am_enabled
                else "tensor numel * element_size plus serialized metadata bytes"
            ),
        },
        "protocol_implementation": {
            "P1": "shared input cache is taken over by SwaT; receiver does not duplicate full-input prefill",
            "P2": "converted q+C_S prefix, followed by receiver-local q+C_R suffix",
            "P3": "converted q+C prefix, followed by receiver-local q suffix",
            "P4": "converted complete-task prefix, followed only by the generation prompt",
        },
        "prompt": {
            "archive": str(prompt_zip),
            "zip_sha256": sha256_file(prompt_zip),
            "schema": prompt_bundle["schema"],
            "version": prompt_bundle["version"],
            "policy": "receiver solver role and task template; thinking disabled",
        },
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "enable_thinking": False,
            "max_new_tokens": TASK_MAX_NEW_TOKENS[dataset],
        },
        "measurement_environment": environment,
        "model_deployments": model_deployments,
        "predictions": str(predictions_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=sorted(PAIR_RUNS), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-zip", type=Path, default=DEFAULT_PROMPT_ZIP)
    parser.add_argument(
        "--only-condition",
        action="append",
        help="Repeatable; for example: --only-condition gsm8k:P4 --only-condition hotpotqa:P3",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, help="Smoke-test override only")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--am-query-transfer", type=Path)
    parser.add_argument("--am-target-ratio", type=float)
    parser.add_argument("--am-budget-label")
    parser.add_argument(
        "--am-head-budget-mode", choices=("uniform", "precomputed"), default="uniform"
    )
    parser.add_argument("--am-head-budget-path", type=Path)
    parser.add_argument("--am-query-samples", type=int, default=512)
    parser.add_argument("--am-nnls-iters", type=int, default=2)
    parser.add_argument("--am-compaction-workers", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument(
        "--am-physical-gpu",
        default="0",
        help="Physical GPU exposed to AM evaluation; defaults to Device 0 (RTX 3090).",
    )
    parser.add_argument(
        "--am-required-gpu-substring",
        default="RTX 3090",
        help="Refuse AM evaluation unless the selected physical GPU name contains this text.",
    )
    args = parser.parse_args()

    am_enabled = args.am_target_ratio is not None
    if am_enabled:
        if not 0 < args.am_target_ratio <= 1:
            parser.error("--am-target-ratio must be in (0, 1]")
        if args.am_query_transfer is None or args.am_budget_label is None:
            parser.error("AM requires --am-query-transfer and --am-budget-label")
        if args.am_head_budget_mode == "precomputed" and args.am_head_budget_path is None:
            parser.error("precomputed AM budgets require --am-head-budget-path")
        if args.am_head_budget_mode == "uniform" and args.am_head_budget_path is not None:
            parser.error("--am-head-budget-path is only valid with precomputed budgets")
    elif args.am_query_transfer is not None or args.am_budget_label is not None or args.am_head_budget_path is not None:
        parser.error("AM options require --am-target-ratio")
    if am_enabled:
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was initialized before selecting the AM evaluation GPU")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.am_physical_gpu)
        selected_gpu_name = subprocess.run(
            [
                "nvidia-smi",
                f"--id={args.am_physical_gpu}",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if args.am_required_gpu_substring not in selected_gpu_name:
            raise RuntimeError(
                f"AM evaluation requires {args.am_required_gpu_substring!r}, selected {selected_gpu_name!r}"
            )

    selected = [
        (dataset, protocol)
        for dataset, protocol in CONDITIONS
        if args.only_condition is None or f"{dataset}:{protocol}" in args.only_condition
    ]
    if not selected:
        raise ValueError(f"Unknown condition(s) {args.only_condition!r}")

    run_name = PAIR_RUNS[args.pair]
    checkpoint = args.checkpoint or (
        SWAT_ROOT
        / "outputs/subproblem4_original_recipe"
        / run_name
        / "stage2/exit-checkpoint-24000/renderer.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    prompt_bundle = load_prompt_bundle(args.prompt_zip)
    renderer, renderer_payload = load_renderer_checkpoint(checkpoint, args.device)
    sender_path = Path(renderer_payload["sender_model"])
    receiver_path = Path(renderer_payload["receiver_model"])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    sender_tokenizer = load_tokenizer(sender_path)
    receiver_tokenizer = load_tokenizer(receiver_path)
    sender_model = load_model(sender_path, dtype, args.device)
    receiver_model = load_model(receiver_path, dtype, args.device)
    query_transfer_checkpoint = None
    query_transfer = None
    query_transfer_parameters = 0
    query_transfer_loaded_weight_bytes = 0
    am_config = None
    if am_enabled:
        query_transfer_checkpoint, query_transfer = load_query_transfer(
            args.am_query_transfer, args.device
        )
        query_transfer_parameters = sum(parameter.numel() for parameter in query_transfer.parameters())
        query_transfer_loaded_weight_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in query_transfer.parameters()
        )
        am_config = AttentionMatchingConfig(
            target_ratio=args.am_target_ratio,
            query_samples=args.am_query_samples,
            nnls_iters=args.am_nnls_iters,
            head_budget_mode=args.am_head_budget_mode,
            head_budget_shares=(
                load_head_budget_shares(str(args.am_head_budget_path))
                if args.am_head_budget_path is not None
                else None
            ),
            query_storage="cpu",
            attention_backend="eager",
            decode_backend="eager",
            compaction_workers=args.am_compaction_workers,
        )
    renderer_parameters = sum(parameter.numel() for parameter in renderer.parameters())
    renderer_loaded_weight_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in renderer.parameters()
    )
    receiver_embedding_parameters = sum(
        parameter.numel() for parameter in receiver_model.get_input_embeddings().parameters()
    )
    direction = f"{sender_path.name}_to_{receiver_path.name}"
    environment = runtime_environment()
    deployments = {
        "sender": model_metadata(sender_model, sender_path, "upstream"),
        "receiver": model_metadata(receiver_model, receiver_path, "downstream"),
    }
    am_summary_kwargs = {
        "am_enabled": am_enabled,
        "am_target_ratio": args.am_target_ratio,
        "am_budget_label": args.am_budget_label,
        "am_head_budget_mode": args.am_head_budget_mode if am_enabled else None,
        "am_head_budget_path": args.am_head_budget_path,
        "query_transfer_checkpoint": query_transfer_checkpoint,
        "query_transfer_parameters": query_transfer_parameters,
        "query_transfer_loaded_weight_bytes": query_transfer_loaded_weight_bytes,
    }
    pair_root = args.output_root / run_name
    pair_root.mkdir(parents=True, exist_ok=True)

    for dataset, protocol in selected:
        condition_root = pair_root / dataset / protocol
        condition_root.mkdir(parents=True, exist_ok=True)
        predictions_path = condition_root / "predictions.jsonl"
        summary_path = condition_root / "summary.json"
        completed, metrics, timings, communication, tokens, finishes = prior_state(predictions_path)
        expected_rows, excluded_rows = expected_and_excluded(dataset, protocol)
        if args.limit is not None:
            expected_rows = min(expected_rows, args.limit)
        processed_this_run = 0
        task_prompt = prompt_bundle["per_task"][dataset]

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

                sender_view, receiver_view = protocol_views(record, protocol, task_prompt["p0"])
                sender_messages = [
                    {"role": "system", "content": task_prompt["receiver_system"]},
                    {
                        "role": "user",
                        "content": sender_view + "\n\n" + task_prompt["receiver_task"],
                    },
                ]
                receiver_suffix_messages = None
                if protocol in {"P2", "P3"}:
                    receiver_suffix_messages = [
                        {"role": "system", "content": task_prompt["receiver_system"]},
                        {
                            "role": "user",
                            "content": receiver_view + "\n\n" + task_prompt["receiver_task"],
                        },
                    ]

                torch.cuda.reset_peak_memory_stats()
                inference_started = time.perf_counter()
                if am_enabled:
                    assert query_transfer is not None and am_config is not None
                    generated = generate_one_am(
                        renderer=renderer,
                        renderer_payload=renderer_payload,
                        sender_model=sender_model,
                        receiver_model=receiver_model,
                        sender_tokenizer=sender_tokenizer,
                        receiver_tokenizer=receiver_tokenizer,
                        query_transfer=query_transfer,
                        am_config=am_config,
                        dataset=dataset,
                        protocol=protocol,
                        record_id=record["record_id"],
                        record_index=eligible_ordinal - 1,
                        sender_messages=sender_messages,
                        receiver_suffix_messages=receiver_suffix_messages,
                        max_length=args.max_length,
                        max_new_tokens=args.max_new_tokens or TASK_MAX_NEW_TOKENS[dataset],
                        device=args.device,
                    )
                else:
                    generated = generate_one(
                        renderer=renderer,
                        renderer_payload=renderer_payload,
                        sender_model=sender_model,
                        receiver_model=receiver_model,
                        sender_tokenizer=sender_tokenizer,
                        receiver_tokenizer=receiver_tokenizer,
                        sender_messages=sender_messages,
                        receiver_suffix_messages=receiver_suffix_messages,
                        max_length=args.max_length,
                        max_new_tokens=args.max_new_tokens or TASK_MAX_NEW_TOKENS[dataset],
                        device=args.device,
                    )
                generated["timing_ms"]["end_to_end_inference_wall_ms"] = (
                    time.perf_counter() - inference_started
                ) * 1000
                scoring_started = time.perf_counter()
                result_metrics, execution = score_prediction(
                    dataset, generated["prediction"], record
                )
                generated["timing_ms"]["quality_scoring_wall_ms"] = (
                    time.perf_counter() - scoring_started
                ) * 1000

                result = {
                    "result_schema_version": RESULT_SCHEMA_VERSION,
                    "method": "SwaT+AM" if am_enabled else "SwaT",
                    "compression": (
                        f"am:{args.am_budget_label}:ratio={args.am_target_ratio:g}"
                        if am_enabled
                        else "none"
                    ),
                    "pair": args.pair,
                    "direction": direction,
                    "dataset": dataset,
                    "protocol": protocol,
                    "record_id": record["record_id"],
                    "task_id": record.get("task_id"),
                    "prediction": generated["prediction"],
                    "reference": record.get("answer"),
                    "metrics": result_metrics,
                    "humaneval_execution": execution,
                    "token_counts": generated["token_counts"],
                    "communication_bytes": generated["communication_bytes"],
                    "timing_ms": generated["timing_ms"],
                    "cache_shape": generated["cache_shape"],
                    "receiver_suffix_mode": generated["suffix_mode"],
                    "termination": {
                        "finish_reason": generated["finish_reason"],
                        "stop_reason_detail": generated.get("stop_reason_detail"),
                        "max_new_tokens": args.max_new_tokens
                        or TASK_MAX_NEW_TOKENS[dataset],
                        "decode_forward_calls": generated["decode_forward_calls"],
                    },
                    "protocol_compliance": {
                        "p2_eligible": record.get("p2_eligible"),
                        "p2_source_support_paragraphs": 1 if protocol == "P2" else None,
                        "p2_source_distractor_paragraphs": 4 if protocol == "P2" else None,
                        "p2_receiver_support_paragraphs": 1 if protocol == "P2" else None,
                        "p2_receiver_distractor_paragraphs": 4 if protocol == "P2" else None,
                        "p4_receiver_uses_generation_head_only": protocol == "P4",
                    },
                    "memory": {
                        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                    },
                }
                output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()
                completed.add(record["record_id"])
                processed_this_run += 1
                add_values(metrics, result_metrics)
                add_values(timings, generated["timing_ms"])
                add_values(communication, generated["communication_bytes"])
                add_values(tokens, generated["token_counts"])
                finishes[generated["finish_reason"]] += 1

                summary = summary_payload(
                    pair=args.pair,
                    direction=direction,
                    dataset=dataset,
                    protocol=protocol,
                    completed_rows=len(completed),
                    expected_rows=expected_rows,
                    excluded_rows=excluded_rows,
                    processed_rows_this_run=processed_this_run,
                    metrics=metrics,
                    timings=timings,
                    communication=communication,
                    tokens=tokens,
                    finishes=finishes,
                    checkpoint=checkpoint,
                    checkpoint_step=renderer_payload.get("step"),
                    renderer_parameters=renderer_parameters,
                    renderer_loaded_weight_bytes=renderer_loaded_weight_bytes,
                    receiver_embedding_parameters=receiver_embedding_parameters,
                    environment=environment,
                    model_deployments=deployments,
                    predictions_path=predictions_path,
                    prompt_zip=args.prompt_zip,
                    prompt_bundle=prompt_bundle,
                    **am_summary_kwargs,
                )
                if processed_this_run == 1 or len(completed) % 10 == 0 or len(completed) == expected_rows:
                    atomic_write_json(summary_path, summary)
                print(
                    json.dumps(
                        {
                            "pair": args.pair,
                            "dataset": dataset,
                            "protocol": protocol,
                            "record_id": record["record_id"],
                            "completed": len(completed),
                            "expected": expected_rows,
                            "metrics": result_metrics,
                            "finish_reason": generated["finish_reason"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        atomic_write_json(
            summary_path,
            summary_payload(
                pair=args.pair,
                direction=direction,
                dataset=dataset,
                protocol=protocol,
                completed_rows=len(completed),
                expected_rows=expected_rows,
                excluded_rows=excluded_rows,
                processed_rows_this_run=processed_this_run,
                metrics=metrics,
                timings=timings,
                communication=communication,
                tokens=tokens,
                finishes=finishes,
                checkpoint=checkpoint,
                checkpoint_step=renderer_payload.get("step"),
                renderer_parameters=renderer_parameters,
                renderer_loaded_weight_bytes=renderer_loaded_weight_bytes,
                receiver_embedding_parameters=receiver_embedding_parameters,
                environment=environment,
                model_deployments=deployments,
                predictions_path=predictions_path,
                prompt_zip=args.prompt_zip,
                prompt_bundle=prompt_bundle,
                **am_summary_kwargs,
            ),
        )


if __name__ == "__main__":
    main()
