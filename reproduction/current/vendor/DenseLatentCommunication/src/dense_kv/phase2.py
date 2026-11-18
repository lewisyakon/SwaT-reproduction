from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from .cache import tensors_to_dynamic_cache
from .checkpoint import load_adapter_checkpoint, save_adapter_checkpoint
from .config import ExperimentConfig
from .data import append_jsonl, read_jsonl, shuffled_batches
from .prompts import phase2_sequence, question_batch
from .runtime import (
    build_adapter,
    extract_cache,
    load_model,
    load_tokenizer,
    parse_dtype,
    set_seed,
)


def train_phase2(
    cfg: ExperimentConfig,
    *,
    max_steps: int | None = None,
    max_samples: int | None = None,
    resume: str | None = None,
) -> dict[str, float | int | str]:
    if cfg.training.batch_size != 1:
        raise ValueError("Phase II currently requires batch_size=1")
    set_seed(cfg.training.seed)
    rng = random.Random(cfg.training.seed)
    records = read_jsonl(cfg.paths.trace_data)
    if max_samples is not None:
        records = records[:max_samples]
    required = {"question", "trace"}
    if any(not required.issubset(record) for record in records):
        raise ValueError("Every Phase-II record must contain 'question' and 'trace'")

    dtype = parse_dtype(cfg.adapter.dtype)
    tokenizer = load_tokenizer(cfg.receiver, local_files_only=True)
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
    checkpoint = resume or cfg.paths.phase1_checkpoint
    load_adapter_checkpoint(checkpoint, adapter, expected_direction=cfg.direction)
    adapter.train()

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = max_steps or cfg.training.phase2_steps
    accumulation = cfg.training.gradient_accumulation_steps
    batches = shuffled_batches(records, 1, cfg.training.seed)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "phase2_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    latest: dict[str, float | int | str] = {}

    for step in range(1, total_steps + 1):
        started = time.monotonic()
        record = next(batches)[0]
        token_batch = question_batch(
            tokenizer, [record["question"]], cfg.training.max_prompt_tokens
        )
        sender_cache = extract_cache(
            sender,
            token_batch.input_ids,
            token_batch.attention_mask,
            token_batch.position_ids,
        )
        transformed = adapter(sender_cache, token_batch.position_ids)
        past = tensors_to_dynamic_cache(transformed)
        past_length = past.get_seq_length()

        context_aware = rng.random() < cfg.training.context_aware_probability
        input_ids, labels = phase2_sequence(
            tokenizer,
            record["question"],
            record["trace"],
            context_aware=context_aware,
            enable_thinking=cfg.training.enable_thinking,
            max_prompt_tokens=cfg.training.max_prompt_tokens,
            max_trace_tokens=cfg.training.max_trace_tokens,
        )
        device = receiver.device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        sequence_length = input_ids.shape[1]
        attention_mask = torch.ones(
            (1, past_length + sequence_length), dtype=torch.long, device=device
        )
        cache_position = torch.arange(
            past_length, past_length + sequence_length, device=device
        )
        outputs = receiver(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
            past_key_values=past,
            use_cache=False,
            return_dict=True,
        )
        loss = outputs.loss
        (loss / accumulation).backward()
        if step % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        latest = {
            "step": step,
            "generation_loss": float(loss.detach().float().cpu()),
            "mode": "context_aware" if context_aware else "context_unaware",
            "target_tokens": int((labels != -100).sum().item()),
            "gate_mean": float(adapter.gate_values().detach().float().mean().cpu()),
            "seconds": time.monotonic() - started,
        }
        if step % cfg.training.log_every == 0:
            append_jsonl(log_path, latest)
            print(
                f"phase2 step={step}/{total_steps} mode={latest['mode']} "
                f"loss={latest['generation_loss']:.6f}",
                flush=True,
            )
        if step % cfg.training.save_every == 0:
            save_adapter_checkpoint(
                cfg.paths.phase2_checkpoint,
                adapter,
                cfg,
                phase=2,
                step=step,
                metrics=latest,
            )
        del sender_cache, transformed, past, outputs, loss

    if total_steps % accumulation:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    save_adapter_checkpoint(
        cfg.paths.phase2_checkpoint,
        adapter,
        cfg,
        phase=2,
        step=total_steps,
        metrics=latest,
    )
    return latest
