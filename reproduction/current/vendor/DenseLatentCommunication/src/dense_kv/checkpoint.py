from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .adapter import DenseKVAdapter
from .config import ExperimentConfig


def save_adapter_checkpoint(
    path: str | Path,
    adapter: DenseKVAdapter,
    config: ExperimentConfig,
    *,
    phase: int,
    step: int,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "phase": phase,
        "step": step,
        "direction": config.direction,
        "config": config.as_dict(),
        "adapter_state_dict": {
            key: value.detach().cpu() for key, value in adapter.state_dict().items()
        },
        "metrics": metrics or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_adapter_checkpoint(
    path: str | Path,
    adapter: DenseKVAdapter,
    *,
    expected_direction: str | None = None,
) -> dict[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if state.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {state.get('format_version')}")
    if expected_direction is not None and state.get("direction") != expected_direction:
        raise ValueError(
            f"Checkpoint direction {state.get('direction')!r} does not match "
            f"{expected_direction!r}"
        )
    adapter.load_state_dict(state["adapter_state_dict"], strict=True)
    return state
