from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

CacheTensors = list[tuple[torch.Tensor, torch.Tensor]]


def cache_to_tensors(cache: Any) -> CacheTensors:
    if cache is None:
        raise ValueError("Model did not return a KV cache")
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        keys = cache.key_cache
        values = cache.value_cache
        if len(keys) != len(values):
            raise ValueError("Cache key/value layer counts differ")
        return list(zip(keys, values))
    if isinstance(cache, Iterable):
        layers = list(cache)
        if all(isinstance(layer, (tuple, list)) and len(layer) == 2 for layer in layers):
            return [(layer[0], layer[1]) for layer in layers]
    raise TypeError(f"Unsupported cache type: {type(cache)!r}")


def tensors_to_dynamic_cache(layers: CacheTensors):
    try:
        from transformers import DynamicCache
    except ImportError as exc:
        raise RuntimeError("transformers is required for cache injection") from exc

    cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(layers):
        cache.update(key, value, layer_idx)
    return cache


def clone_cache_tensors(layers: CacheTensors, *, detach: bool = False) -> CacheTensors:
    cloned: CacheTensors = []
    for key, value in layers:
        key_copy = key.clone()
        value_copy = value.clone()
        if detach:
            key_copy = key_copy.detach()
            value_copy = value_copy.detach()
        cloned.append((key_copy, value_copy))
    return cloned


def cache_shape(layers: CacheTensors) -> tuple[int, int, int, int, int]:
    if not layers:
        raise ValueError("Cache is empty")
    first_key, first_value = layers[0]
    if first_key.shape != first_value.shape or first_key.ndim != 4:
        raise ValueError("Cache layers must contain matching rank-4 key/value tensors")
    shape = tuple(first_key.shape)
    for key, value in layers:
        if tuple(key.shape) != shape or tuple(value.shape) != shape:
            raise ValueError("All cache layers must have the same shape")
    batch, heads, tokens, head_dim = shape
    return len(layers), batch, heads, tokens, head_dim

