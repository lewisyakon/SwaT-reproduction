from __future__ import annotations

from dataclasses import dataclass

import torch


def _apply_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
    **kwargs,
):
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def chat_ids(
    tokenizer,
    question: str,
    *,
    generation_prompt: bool,
    enable_thinking: bool = False,
) -> list[int]:
    result = _apply_chat_template(
        tokenizer,
        [{"role": "user", "content": question}],
        enable_thinking=enable_thinking,
        tokenize=True,
        add_generation_prompt=generation_prompt,
    )
    if isinstance(result, torch.Tensor):
        result = result.tolist()
    if result and isinstance(result[0], list):
        result = result[0]
    return list(result)


def assistant_prefix_ids(
    tokenizer, question: str, *, enable_thinking: bool = False
) -> list[int]:
    user = chat_ids(
        tokenizer,
        question,
        generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    complete = chat_ids(
        tokenizer,
        question,
        generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if complete[: len(user)] != user:
        raise ValueError("Tokenizer chat template does not have a stable user prefix")
    suffix = complete[len(user) :]
    if suffix:
        return suffix
    fallback = tokenizer.bos_token_id
    if fallback is None:
        fallback = tokenizer.eos_token_id
    if fallback is None:
        raise ValueError("Tokenizer provides neither an assistant prefix nor BOS/EOS")
    return [fallback]


@dataclass
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


def pad_token_sequences(
    sequences: list[list[int]],
    pad_token_id: int,
    *,
    max_length: int | None = None,
) -> TokenBatch:
    if not sequences:
        raise ValueError("At least one token sequence is required")
    if max_length is not None:
        sequences = [sequence[:max_length] for sequence in sequences]
    width = max(len(sequence) for sequence in sequences)
    if width == 0:
        raise ValueError("Token sequences may not be empty")
    input_ids = torch.full(
        (len(sequences), width), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, :length] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.clamp_(min=0)
    return TokenBatch(input_ids, attention_mask, position_ids)


def question_batch(tokenizer, questions: list[str], max_length: int) -> TokenBatch:
    sequences = [
        chat_ids(tokenizer, question, generation_prompt=False)
        for question in questions
    ]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no padding or EOS token")
    return pad_token_sequences(sequences, pad_id, max_length=max_length)


def phase2_sequence(
    tokenizer,
    question: str,
    trace: str,
    *,
    context_aware: bool,
    enable_thinking: bool,
    max_prompt_tokens: int,
    max_trace_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context_aware:
        prefix = chat_ids(
            tokenizer,
            question,
            generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    else:
        prefix = assistant_prefix_ids(
            tokenizer, question, enable_thinking=enable_thinking
        )
    prefix = prefix[:max_prompt_tokens]
    target = tokenizer.encode(trace, add_special_tokens=False)[:max_trace_tokens]
    if tokenizer.eos_token_id is not None and (
        not target or target[-1] != tokenizer.eos_token_id
    ):
        target.append(tokenizer.eos_token_id)
    if not target:
        raise ValueError("Trace tokenization produced no target tokens")
    input_ids = torch.tensor([prefix + target], dtype=torch.long)
    labels = torch.tensor([[-100] * len(prefix) + target], dtype=torch.long)
    return input_ids, labels


def trace_guidance_question(question: str, gold_answer: str) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Gold answer:\n"
        f"{gold_answer}\n\n"
        "Give a concise step-by-step derivation that reaches the gold answer. "
        "End by stating that answer clearly."
    )


def t2t_receiver_question(question: str, sender_reasoning: str) -> str:
    return (
        f"{question}\n\n"
        "Another model produced the following reasoning. Check it and then answer "
        f"the original question.\n\n{sender_reasoning}"
    )
