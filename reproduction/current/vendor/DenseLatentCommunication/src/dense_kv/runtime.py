from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch

from .adapter import DenseKVAdapter
from .cache import CacheTensors, cache_to_tensors, tensors_to_dynamic_cache
from .config import ExperimentConfig, ModelSpec


DTYPES = {
    "float32": torch.float32,
    "float": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def parse_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(DTYPES)}") from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_source(spec: ModelSpec) -> str:
    path = Path(spec.path)
    return str(path) if path.exists() else spec.model_id


def validate_hf_config(hf_config: Any, spec: ModelSpec) -> None:
    expected = {
        "num_hidden_layers": spec.num_layers,
        "num_key_value_heads": spec.num_kv_heads,
        "head_dim": spec.head_dim,
        "max_position_embeddings": spec.max_position_embeddings,
    }
    for key, expected_value in expected.items():
        actual = getattr(hf_config, key, None)
        if actual != expected_value:
            raise ValueError(
                f"{spec.model_id} config {key}={actual!r}, expected {expected_value!r}"
            )
    rope_theta = float(getattr(hf_config, "rope_theta", 10000.0))
    if rope_theta != spec.rope_theta:
        raise ValueError(
            f"{spec.model_id} rope_theta={rope_theta}, expected {spec.rope_theta}"
        )
    if getattr(hf_config, "rope_scaling", None) is not None:
        raise ValueError("This reproduction currently supports default, unscaled Qwen3 RoPE")


def load_tokenizer(spec: ModelSpec, *, local_files_only: bool = False):
    from transformers import AutoTokenizer

    source = model_source(spec)
    revision = None if Path(spec.path).exists() else spec.revision
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=revision,
        local_files_only=local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    spec: ModelSpec,
    device: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
):
    from transformers import AutoConfig, AutoModelForCausalLM

    source = model_source(spec)
    revision = None if Path(spec.path).exists() else spec.revision
    hf_config = AutoConfig.from_pretrained(
        source, revision=revision, local_files_only=local_files_only
    )
    validate_hf_config(hf_config, spec)
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
        "attn_implementation": "sdpa",
        "revision": revision,
    }
    if device != "cpu":
        kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if device == "cpu":
        model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_adapter(cfg: ExperimentConfig) -> DenseKVAdapter:
    adapter = DenseKVAdapter(
        cfg.sender,
        cfg.receiver,
        expansion=cfg.adapter.expansion,
        gate_init=cfg.adapter.gate_init,
    )
    return adapter.to(
        device=cfg.training.adapter_device,
        dtype=parse_dtype(cfg.adapter.dtype),
    )


@torch.no_grad()
def extract_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> CacheTensors:
    device = model.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)
    cache_position = torch.arange(input_ids.shape[1], device=device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )
    return cache_to_tensors(outputs.past_key_values)


@torch.no_grad()
def manual_generate(
    model,
    prefix_ids: list[int] | torch.Tensor,
    *,
    past_layers: CacheTensors | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_ids: set[int] | None = None,
) -> tuple[list[int], dict[str, int]]:
    device = model.device
    if isinstance(prefix_ids, list):
        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    else:
        input_ids = prefix_ids.to(device)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("manual_generate expects one non-empty token sequence")

    cache = tensors_to_dynamic_cache(past_layers) if past_layers is not None else None
    past_length = cache.get_seq_length() if cache is not None else 0
    attention_mask = torch.ones(
        (1, past_length + input_ids.shape[1]), dtype=torch.long, device=device
    )
    generated: list[int] = []
    current_ids = input_ids

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()
        if temperature <= 0.0:
            next_token = logits.argmax(dim=-1)
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = probabilities.sort(descending=True)
                cumulative = sorted_probs.cumsum(dim=-1)
                remove = cumulative - sorted_probs >= top_p
                sorted_probs[remove] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, sampled).squeeze(-1)
            else:
                next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        token = int(next_token.item())
        generated.append(token)
        if eos_token_ids and token in eos_token_ids:
            break
        current_ids = next_token[:, None]
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
            ],
            dim=-1,
        )

    return generated, {
        "prefill_tokens": int(input_ids.shape[1]),
        "decode_tokens": len(generated),
    }


def model_config_from_disk(path: str | Path) -> dict[str, Any]:
    with (Path(path) / "config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def adapter_parameter_count(cfg: ExperimentConfig) -> int:
    sender_dim = cfg.sender.head_dim
    receiver_dim = cfg.receiver.head_dim
    hidden = cfg.adapter.expansion * sender_dim
    per_mlp = sender_dim * hidden + hidden + hidden * receiver_dim + receiver_dim
    per_layer = 2 * cfg.receiver.num_kv_heads * per_mlp
    gates = cfg.receiver.num_kv_heads
    return cfg.receiver.num_layers * (per_layer + gates)
