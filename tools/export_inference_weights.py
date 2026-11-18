#!/usr/bin/env python3
"""Export exact, optimizer-free SwaT checkpoints and build a checksum manifest."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
SWAT = Path("/mnt/hd2/lewis/SwaTReproduction")
SWAT_CODE = Path("/mnt/hd2/lewis/SwaT-Code")


ENTRIES = [
    (
        "renderer",
        "original/M1/renderer.pt",
        SWAT / "outputs/subproblem4_original_recipe/m1_qwen3_0.6b_to_qwen3_1.7b/stage2/exit-checkpoint-24000/renderer.pt",
        "Qwen3-0.6B",
        "Qwen3-1.7B",
    ),
    (
        "renderer",
        "original/M2/renderer.pt",
        SWAT / "outputs/subproblem4_original_recipe/m2_qwen3_0.6b_to_qwen3_4b/stage2/exit-checkpoint-24000/renderer.pt",
        "Qwen3-0.6B",
        "Qwen3-4B",
    ),
    (
        "renderer",
        "original/M3/renderer.pt",
        SWAT / "outputs/subproblem4_original_recipe/m3_qwen3_4b_to_qwen3_0.6b/stage2/exit-checkpoint-24000/renderer.pt",
        "Qwen3-4B",
        "Qwen3-0.6B",
    ),
    (
        "query_transfer",
        "am-query-transfer/M1/query_transfer.pt",
        SWAT / "outputs/subproblem4_original_recipe/am_query_transfer_original_recipe_v1/M1/exit-checkpoint-2000/query_transfer.pt",
        "Qwen3-0.6B",
        "Qwen3-1.7B",
    ),
    (
        "query_transfer",
        "am-query-transfer/M2/query_transfer.pt",
        SWAT / "outputs/subproblem4_original_recipe/am_query_transfer_original_recipe_v1/M2/exit-checkpoint-2000/query_transfer.pt",
        "Qwen3-0.6B",
        "Qwen3-4B",
    ),
    (
        "renderer",
        "swat-code/M1/renderer.pt",
        SWAT_CODE / "outputs/m1_qwen3_0.6b_to_qwen3_1.7b/stage2_code_adaptation/exit-checkpoint-4000/renderer.pt",
        "Qwen3-0.6B",
        "Qwen3-1.7B",
    ),
    (
        "renderer",
        "swat-code/M2/renderer.pt",
        SWAT_CODE / "outputs/m2_qwen3_0.6b_to_qwen3_4b/stage2_code_adaptation/exit-checkpoint-4000/renderer.pt",
        "Qwen3-0.6B",
        "Qwen3-4B",
    ),
    (
        "renderer",
        "swat-code/M3/renderer.pt",
        SWAT_CODE / "outputs/m3_qwen3_4b_to_qwen3_0.6b/stage2_code_adaptation/exit-checkpoint-4000/renderer.pt",
        "Qwen3-4B",
        "Qwen3-0.6B",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_stats(state: dict[str, torch.Tensor]) -> tuple[int, int, dict[str, int]]:
    dtypes: Counter[str] = Counter()
    parameters = 0
    tensor_bytes = 0
    for tensor in state.values():
        parameters += tensor.numel()
        size = tensor.numel() * tensor.element_size()
        tensor_bytes += size
        dtypes[str(tensor.dtype)] += size
    return parameters, tensor_bytes, dict(sorted(dtypes.items()))


def equal_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    if left.keys() != right.keys():
        raise ValueError("state-dict keys changed during export")
    for name in left:
        a, b = left[name], right[name]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            raise ValueError(f"tensor changed during export: {name}")


def main() -> None:
    output_root = REPO / "weights"
    records = []
    for kind, relative, source, sender, receiver in ENTRIES:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        if kind == "renderer":
            state_key = "renderer_state_dict"
            drop = ["optimizer_state_dict", "scheduler_state_dict"]
        else:
            state_key = "query_transfer_state_dict"
            drop = [
                "optimizer_state_dict",
                "losses",
                "torch_rng_state",
                "python_random_state",
            ]
        state = payload[state_key]
        parameters, tensor_bytes, dtypes = state_stats(state)
        exported = {key: value for key, value in payload.items() if key not in drop}
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(exported, temporary)
        os.replace(temporary, destination)
        verified = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
        equal_state(state, verified[state_key])
        records.append(
            {
                "id": relative.removesuffix(".pt").replace("/", "_"),
                "kind": kind,
                "path": str(Path("weights") / relative),
                "sender": sender,
                "receiver": receiver,
                "parameters": parameters,
                "tensor_bytes": tensor_bytes,
                "dtypes_by_bytes": dtypes,
                "file_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "source_path": str(source),
                "source_file_bytes": source.stat().st_size,
                "dropped_training_only_keys": drop,
                "load_api": (
                    "multiagent.model.cache_renderer.load_renderer_checkpoint"
                    if kind == "renderer"
                    else "multiagent.model.query_transfer.load_query_transfer"
                ),
            }
        )
        print(f"exported {destination}: {destination.stat().st_size} bytes")

    manifest = {
        "schema_version": 1,
        "created_at": "2026-11-18",
        "precision_policy": "Exact inference tensors; no dtype conversion",
        "entries": records,
        "total_file_bytes": sum(record["file_bytes"] for record in records),
        "note": "AM M3 is absent because its query-transfer training was not complete at export time.",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
