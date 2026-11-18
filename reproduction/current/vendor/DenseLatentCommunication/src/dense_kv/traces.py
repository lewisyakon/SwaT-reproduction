from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig
from .data import append_jsonl, read_jsonl
from .prompts import chat_ids, trace_guidance_question
from .runtime import load_model, load_tokenizer, manual_generate, parse_dtype, set_seed


def generate_receiver_traces(
    cfg: ExperimentConfig,
    *,
    max_samples: int | None = None,
    max_new_tokens: int = 768,
    temperature: float = 0.0,
) -> int:
    set_seed(cfg.training.seed)
    records = read_jsonl(cfg.paths.train_data)
    if max_samples is not None:
        records = records[:max_samples]
    output_path = Path(cfg.paths.trace_data)
    completed: set[str] = set()
    if output_path.exists():
        completed = {
            str(record.get("id", record.get("question")))
            for record in read_jsonl(output_path)
        }

    tokenizer = load_tokenizer(cfg.receiver, local_files_only=True)
    receiver = load_model(
        cfg.receiver,
        cfg.training.receiver_device,
        dtype=parse_dtype(cfg.adapter.dtype),
        local_files_only=True,
    )
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    written = 0
    for index, record in enumerate(records):
        identity = str(record.get("id", record["question"]))
        if identity in completed:
            continue
        gold = str(record.get("gold_answer", record.get("answer", "")))
        guided = trace_guidance_question(record["question"], gold)
        prefix = chat_ids(
            tokenizer,
            guided,
            generation_prompt=True,
            enable_thinking=cfg.training.enable_thinking,
        )
        generated, usage = manual_generate(
            receiver,
            prefix,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_ids=eos_ids,
        )
        trace = tokenizer.decode(generated, skip_special_tokens=True).strip()
        output = dict(record)
        output.update(
            {
                "trace": trace,
                "receiver": cfg.receiver.model_id,
                "trace_prefill_tokens": usage["prefill_tokens"],
                "trace_decode_tokens": usage["decode_tokens"],
            }
        )
        append_jsonl(output_path, output)
        completed.add(identity)
        written += 1
        print(f"trace {index + 1}/{len(records)} tokens={len(generated)}", flush=True)
    return written
