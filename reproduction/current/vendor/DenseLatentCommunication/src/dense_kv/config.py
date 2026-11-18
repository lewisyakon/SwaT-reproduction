from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    path: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    max_position_embeddings: int
    parameter_count: int
    revision: str | None = None


@dataclass(frozen=True)
class AdapterSpec:
    expansion: int = 16
    gate_init: float = 0.5
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class TrainingSpec:
    sender_device: str = "cuda:0"
    receiver_device: str = "cuda:1"
    adapter_device: str = "cuda:1"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    phase1_steps: int = 2000
    phase2_steps: int = 2000
    max_prompt_tokens: int = 512
    max_trace_tokens: int = 768
    enable_thinking: bool = True
    context_aware_probability: float = 0.5
    seed: int = 42
    log_every: int = 1
    save_every: int = 100


@dataclass(frozen=True)
class PathSpec:
    train_data: str
    trace_data: str
    evaluation_data: str
    output_dir: str
    phase1_checkpoint: str
    phase2_checkpoint: str


@dataclass(frozen=True)
class ExperimentConfig:
    direction: str
    sender: ModelSpec
    receiver: ModelSpec
    adapter: AdapterSpec
    training: TrainingSpec
    paths: PathSpec

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reject_unknown(section: str, data: dict[str, Any], cls: type) -> None:
    expected = set(cls.__dataclass_fields__)
    unknown = set(data) - expected
    if unknown:
        raise ValueError(f"Unknown {section} keys: {sorted(unknown)}")


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    _reject_unknown("top-level", raw, ExperimentConfig)
    _reject_unknown("sender", raw["sender"], ModelSpec)
    _reject_unknown("receiver", raw["receiver"], ModelSpec)
    _reject_unknown("adapter", raw["adapter"], AdapterSpec)
    _reject_unknown("training", raw["training"], TrainingSpec)
    _reject_unknown("paths", raw["paths"], PathSpec)

    cfg = ExperimentConfig(
        direction=raw["direction"],
        sender=ModelSpec(**raw["sender"]),
        receiver=ModelSpec(**raw["receiver"]),
        adapter=AdapterSpec(**raw["adapter"]),
        training=TrainingSpec(**raw["training"]),
        paths=PathSpec(**raw["paths"]),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.sender.num_layers < 1 or cfg.receiver.num_layers < 1:
        raise ValueError("Both models must have at least one layer")
    if cfg.sender.num_kv_heads != cfg.receiver.num_kv_heads:
        raise ValueError(
            "The minimal reproduction uses direct KV-group alignment and therefore "
            "requires equal num_kv_heads"
        )
    if cfg.sender.head_dim < 2 or cfg.receiver.head_dim < 2:
        raise ValueError("head_dim must be at least 2")
    if cfg.sender.head_dim % 2 or cfg.receiver.head_dim % 2:
        raise ValueError("RoPE requires even head dimensions")
    if cfg.adapter.expansion < 1:
        raise ValueError("adapter expansion must be positive")
    if not 0.0 < cfg.adapter.gate_init < 1.0:
        raise ValueError("gate_init must be strictly between zero and one")
    if not 0.0 <= cfg.training.context_aware_probability <= 1.0:
        raise ValueError("context_aware_probability must be in [0, 1]")


def monotonic_layer_mapping(sender_layers: int, receiver_layers: int) -> list[int]:
    if sender_layers < 1 or receiver_layers < 1:
        raise ValueError("Layer counts must be positive")
    if receiver_layers == 1:
        return [0]
    if sender_layers == 1:
        return [0] * receiver_layers
    denominator = receiver_layers - 1
    return [
        (2 * layer * (sender_layers - 1) + denominator) // (2 * denominator)
        for layer in range(receiver_layers)
    ]
