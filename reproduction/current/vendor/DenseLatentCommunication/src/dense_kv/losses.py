from __future__ import annotations

import torch

from .cache import CacheTensors, cache_shape


def reconstruction_loss(
    projected: CacheTensors,
    target: CacheTensors,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    projected_shape = cache_shape(projected)
    target_shape = cache_shape(target)
    if projected_shape != target_shape:
        raise ValueError(
            f"Projected cache shape {projected_shape} != target {target_shape}"
        )
    layers, batch, heads, tokens, head_dim = projected_shape
    if attention_mask.shape != (batch, tokens):
        raise ValueError("attention_mask does not match cache batch/token dimensions")
    mask = attention_mask.to(projected[0][0].device, dtype=torch.float32)
    mask = mask[:, None, :, None]
    denominator = mask.sum() * heads * head_dim
    if denominator.item() == 0:
        raise ValueError("attention_mask contains no valid tokens")

    key_loss = torch.zeros((), device=projected[0][0].device, dtype=torch.float32)
    value_loss = torch.zeros_like(key_loss)
    for (projected_key, projected_value), (target_key, target_value) in zip(
        projected, target
    ):
        target_key = target_key.to(projected_key.device, projected_key.dtype)
        target_value = target_value.to(projected_value.device, projected_value.dtype)
        key_error = (projected_key.float() - target_key.float()).square() * mask
        value_error = (projected_value.float() - target_value.float()).square() * mask
        key_loss = key_loss + key_error.sum() / denominator
        value_loss = value_loss + value_error.sum() / denominator
    key_loss = key_loss / layers
    value_loss = value_loss / layers
    total = key_loss + value_loss
    return total, {
        "key_mse": float(key_loss.detach().cpu()),
        "value_mse": float(value_loss.detach().cpu()),
        "reconstruction_loss": float(total.detach().cpu()),
    }
