from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    midpoint = x.shape[-1] // 2
    return torch.cat((-x[..., midpoint:], x[..., :midpoint]), dim=-1)


def rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2:
        raise ValueError("RoPE head_dim must be even")
    positions = position_ids.to(device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
            / head_dim
        )
    )
    frequencies = positions.unsqueeze(-1) * inv_freq
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def apply_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [batch, heads, tokens, dim], got {tuple(x.shape)}")
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.shape[0] not in (1, x.shape[0]) or position_ids.shape[1] != x.shape[2]:
        raise ValueError(
            f"position_ids {tuple(position_ids.shape)} do not match cache {tuple(x.shape)}"
        )
    cos, sin = rope_cos_sin(
        position_ids,
        x.shape[-1],
        theta,
        dtype=x.dtype,
        device=x.device,
    )
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    sign = -1.0 if inverse else 1.0
    return x * cos + sign * rotate_half(x) * sin

