from __future__ import annotations

import time
from pathlib import Path

import torch

from .checkpoint import load_adapter_checkpoint, save_adapter_checkpoint
from .config import ExperimentConfig
from .data import append_jsonl, read_jsonl, shuffled_batches
from .losses import reconstruction_loss
from .prompts import question_batch
from .runtime import (
    build_adapter,
    extract_cache,
    load_model,
    load_tokenizer,
    parse_dtype,
    set_seed,
)


def train_phase1(
    cfg: ExperimentConfig,
    *,
    max_steps: int | None = None,
    max_samples: int | None = None,
    resume: str | None = None,
) -> dict[str, float | int]:
    set_seed(cfg.training.seed)
    records = read_jsonl(cfg.paths.train_data)
    if max_samples is not None:
        records = records[:max_samples]
    required = {"question"}
    if any(not required.issubset(record) for record in records):
        raise ValueError("Every Phase-I record must contain 'question'")

    dtype = parse_dtype(cfg.adapter.dtype)
    tokenizer = load_tokenizer(cfg.sender, local_files_only=True)
    sender = load_model(
        cfg.sender,
        cfg.training.sender_device,
        dtype=dtype,
        local_files_only=True,
    )
    receiver = load_model(
        cfg.receiver,
        cfg.training.receiver_device,
        dtype=dtype,
        local_files_only=True,
    )
    adapter = build_adapter(cfg)
    if resume:
        load_adapter_checkpoint(resume, adapter, expected_direction=cfg.direction)
    adapter.train()

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = max_steps or cfg.training.phase1_steps
    accumulation = cfg.training.gradient_accumulation_steps
    batches = shuffled_batches(records, cfg.training.batch_size, cfg.training.seed)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "phase1_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    latest: dict[str, float | int] = {}

    for step in range(1, total_steps + 1):
        started = time.monotonic()
        batch_records = next(batches)
        token_batch = question_batch(
            tokenizer,
            [record["question"] for record in batch_records],
            cfg.training.max_prompt_tokens,
        )
        sender_cache = extract_cache(
            sender,
            token_batch.input_ids,
            token_batch.attention_mask,
            token_batch.position_ids,
        )
        target_cache = extract_cache(
            receiver,
            token_batch.input_ids,
            token_batch.attention_mask,
            token_batch.position_ids,
        )
        projected = adapter(sender_cache, token_batch.position_ids)
        loss, components = reconstruction_loss(
            projected, target_cache, token_batch.attention_mask
        )
        (loss / accumulation).backward()
        if step % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        latest = {
            "step": step,
            **components,
            "gate_mean": float(adapter.gate_values().detach().float().mean().cpu()),
            "tokens": int(token_batch.attention_mask.sum().item()),
            "seconds": time.monotonic() - started,
        }
        if step % cfg.training.log_every == 0:
            append_jsonl(log_path, latest)
            print(
                f"phase1 step={step}/{total_steps} "
                f"loss={components['reconstruction_loss']:.6f} "
                f"k={components['key_mse']:.6f} v={components['value_mse']:.6f}",
                flush=True,
            )
        if step % cfg.training.save_every == 0:
            save_adapter_checkpoint(
                cfg.paths.phase1_checkpoint,
                adapter,
                cfg,
                phase=1,
                step=step,
                metrics=latest,
            )

        del sender_cache, target_cache, projected, loss

    if total_steps % accumulation:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    save_adapter_checkpoint(
        cfg.paths.phase1_checkpoint,
        adapter,
        cfg,
        phase=1,
        step=total_steps,
        metrics=latest,
    )
    return latest

