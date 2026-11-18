from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .answers import score_answer
from .checkpoint import load_adapter_checkpoint
from .config import ExperimentConfig
from .data import append_jsonl, read_jsonl
from .prompts import (
    assistant_prefix_ids,
    chat_ids,
    question_batch,
    t2t_receiver_question,
)
from .runtime import (
    build_adapter,
    extract_cache,
    load_model,
    load_tokenizer,
    manual_generate,
    parse_dtype,
    set_seed,
)

VALID_MODES = ("receiver_only", "t2t", "context_aware", "context_unaware")


def estimated_tflops(
    cfg: ExperimentConfig,
    sender_prefill: int,
    sender_decode: int,
    receiver_prefill: int,
    receiver_decode: int,
) -> float:
    flops = 2 * cfg.sender.parameter_count * (sender_prefill + sender_decode)
    flops += 2 * cfg.receiver.parameter_count * (
        receiver_prefill + receiver_decode
    )
    return flops / 1e12


def _decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def evaluate(
    cfg: ExperimentConfig,
    modes: list[str],
    *,
    max_samples: int | None = None,
    max_new_tokens: int = 512,
    checkpoint: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    unknown = set(modes) - set(VALID_MODES)
    if unknown:
        raise ValueError(f"Unknown evaluation modes: {sorted(unknown)}")
    set_seed(cfg.training.seed)
    records = read_jsonl(cfg.paths.evaluation_data)
    if max_samples is not None:
        records = records[:max_samples]
    dtype = parse_dtype(cfg.adapter.dtype)
    sender_tokenizer = load_tokenizer(cfg.sender, local_files_only=True)
    receiver_tokenizer = load_tokenizer(cfg.receiver, local_files_only=True)
    receiver = load_model(
        cfg.receiver,
        cfg.training.receiver_device,
        dtype=dtype,
        local_files_only=True,
    )
    needs_sender = any(mode != "receiver_only" for mode in modes)
    sender = None
    if needs_sender:
        sender = load_model(
            cfg.sender,
            cfg.training.sender_device,
            dtype=dtype,
            local_files_only=True,
        )
    adapter = None
    if any(mode.startswith("context_") for mode in modes):
        adapter = build_adapter(cfg)
        load_adapter_checkpoint(
            checkpoint or cfg.paths.phase2_checkpoint,
            adapter,
            expected_direction=cfg.direction,
        )
        adapter.eval()

    eos_sender = (
        {sender_tokenizer.eos_token_id}
        if sender_tokenizer.eos_token_id is not None
        else set()
    )
    eos_receiver = (
        {receiver_tokenizer.eos_token_id}
        if receiver_tokenizer.eos_token_id is not None
        else set()
    )
    destination = Path(output_path or (Path(cfg.paths.output_dir) / "evaluation.jsonl"))
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "count": 0})

    for index, record in enumerate(records):
        question = record["question"]
        transformed = None
        sender_prefill = 0
        if any(mode.startswith("context_") for mode in modes):
            batch = question_batch(
                sender_tokenizer, [question], cfg.training.max_prompt_tokens
            )
            source_cache = extract_cache(
                sender,
                batch.input_ids,
                batch.attention_mask,
                batch.position_ids,
            )
            with torch.no_grad():
                transformed = adapter(source_cache, batch.position_ids)
            sender_prefill = int(batch.attention_mask.sum().item())

        for mode in modes:
            sender_decode = 0
            mode_sender_prefill = 0
            if mode == "receiver_only":
                prefix = chat_ids(
                    receiver_tokenizer,
                    question,
                    generation_prompt=True,
                    enable_thinking=cfg.training.enable_thinking,
                )
                generated, receiver_usage = manual_generate(
                    receiver,
                    prefix,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_receiver,
                )
            elif mode == "t2t":
                sender_prefix = chat_ids(
                    sender_tokenizer,
                    question,
                    generation_prompt=True,
                    enable_thinking=cfg.training.enable_thinking,
                )
                reasoning_ids, sender_usage = manual_generate(
                    sender,
                    sender_prefix,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_sender,
                )
                reasoning = _decode(sender_tokenizer, reasoning_ids)
                receiver_question = t2t_receiver_question(question, reasoning)
                prefix = chat_ids(
                    receiver_tokenizer,
                    receiver_question,
                    generation_prompt=True,
                    enable_thinking=cfg.training.enable_thinking,
                )
                generated, receiver_usage = manual_generate(
                    receiver,
                    prefix,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_receiver,
                )
                mode_sender_prefill = sender_usage["prefill_tokens"]
                sender_decode = sender_usage["decode_tokens"]
            elif mode == "context_aware":
                prefix = chat_ids(
                    receiver_tokenizer,
                    question,
                    generation_prompt=True,
                    enable_thinking=cfg.training.enable_thinking,
                )
                generated, receiver_usage = manual_generate(
                    receiver,
                    prefix,
                    past_layers=transformed,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_receiver,
                )
                mode_sender_prefill = sender_prefill
            else:
                prefix = assistant_prefix_ids(
                    receiver_tokenizer,
                    question,
                    enable_thinking=cfg.training.enable_thinking,
                )
                generated, receiver_usage = manual_generate(
                    receiver,
                    prefix,
                    past_layers=transformed,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_receiver,
                )
                mode_sender_prefill = sender_prefill

            prediction = _decode(receiver_tokenizer, generated)
            correct = score_answer(record, prediction)
            dataset = str(record.get("dataset", "unknown"))
            totals[f"{mode}:{dataset}"]["correct"] += int(correct)
            totals[f"{mode}:{dataset}"]["count"] += 1
            result = {
                "id": record.get("id", index),
                "dataset": dataset,
                "direction": cfg.direction,
                "mode": mode,
                "question": question,
                "gold_answer": record.get("gold_answer", record.get("answer")),
                "prediction": prediction,
                "correct": correct,
                "sender_prefill_tokens": mode_sender_prefill,
                "sender_decode_tokens": sender_decode,
                "receiver_prefill_tokens": receiver_usage["prefill_tokens"],
                "receiver_decode_tokens": receiver_usage["decode_tokens"],
            }
            result["estimated_tflops"] = estimated_tflops(
                cfg,
                result["sender_prefill_tokens"],
                result["sender_decode_tokens"],
                result["receiver_prefill_tokens"],
                result["receiver_decode_tokens"],
            )
            append_jsonl(destination, result)
            print(
                f"eval {index + 1}/{len(records)} mode={mode} "
                f"dataset={dataset} correct={correct}",
                flush=True,
            )

    summary = {
        key: {
            **counts,
            "accuracy": counts["correct"] / counts["count"],
        }
        for key, counts in sorted(totals.items())
    }
    append_jsonl(destination.with_name("evaluation_summary.jsonl"), summary)
    return summary
