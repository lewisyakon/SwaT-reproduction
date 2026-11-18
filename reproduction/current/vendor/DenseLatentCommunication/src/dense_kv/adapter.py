from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .cache import CacheTensors, cache_shape
from .config import ModelSpec, monotonic_layer_mapping
from .rope import apply_rope


class HeadwiseMLP(nn.Module):
    """Independent two-layer MLPs, batched over KV heads for efficiency."""

    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight1 = nn.Parameter(torch.empty(num_heads, hidden_dim, input_dim))
        self.bias1 = nn.Parameter(torch.empty(num_heads, hidden_dim))
        self.weight2 = nn.Parameter(torch.empty(num_heads, output_dim, hidden_dim))
        self.bias2 = nn.Parameter(torch.empty(num_heads, output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for head in range(self.num_heads):
            nn.init.kaiming_uniform_(self.weight1[head], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight2[head], a=math.sqrt(5))
        bound1 = 1.0 / math.sqrt(self.input_dim)
        bound2 = 1.0 / math.sqrt(self.weight2.shape[-1])
        nn.init.uniform_(self.bias1, -bound1, bound1)
        nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.num_heads or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [batch, {self.num_heads}, tokens, {self.input_dim}], "
                f"got {tuple(x.shape)}"
            )
        hidden = torch.einsum("bhtd,hmd->bhtm", x, self.weight1)
        hidden = F.gelu(hidden + self.bias1[None, :, None, :])
        output = torch.einsum("bhtm,hom->bhto", hidden, self.weight2)
        return output + self.bias2[None, :, None, :]


class LayerTransform(nn.Module):
    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        gate_init: float,
    ) -> None:
        super().__init__()
        self.key_mlp = HeadwiseMLP(num_heads, input_dim, hidden_dim, output_dim)
        self.value_mlp = HeadwiseMLP(num_heads, input_dim, hidden_dim, output_dim)
        initial_logit = math.log(gate_init / (1.0 - gate_init))
        self.gate_logits = nn.Parameter(torch.full((num_heads,), initial_logit))

    def forward(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate_logits.sigmoid()[None, :, None, None]
        return gate * self.key_mlp(key), gate * self.value_mlp(value)


class DenseKVAdapter(nn.Module):
    def __init__(
        self,
        sender: ModelSpec,
        receiver: ModelSpec,
        *,
        expansion: int = 16,
        gate_init: float = 0.5,
    ) -> None:
        super().__init__()
        if sender.num_kv_heads != receiver.num_kv_heads:
            raise ValueError("Direct KV-group alignment requires matching KV-head counts")
        self.sender_spec = sender
        self.receiver_spec = receiver
        self.expansion = expansion
        self.gate_init = gate_init
        mapping = monotonic_layer_mapping(sender.num_layers, receiver.num_layers)
        self.register_buffer("layer_mapping", torch.tensor(mapping, dtype=torch.long))
        hidden_dim = expansion * sender.head_dim
        self.transforms = nn.ModuleList(
            LayerTransform(
                receiver.num_kv_heads,
                sender.head_dim,
                hidden_dim,
                receiver.head_dim,
                gate_init,
            )
            for _ in range(receiver.num_layers)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        sender_cache: CacheTensors,
        position_ids: torch.Tensor,
    ) -> CacheTensors:
        layers, batch, heads, tokens, head_dim = cache_shape(sender_cache)
        expected = (
            self.sender_spec.num_layers,
            self.sender_spec.num_kv_heads,
            self.sender_spec.head_dim,
        )
        if (layers, heads, head_dim) != expected:
            raise ValueError(
                "Sender cache does not match configured layers/heads/head_dim: "
                f"got {(layers, heads, head_dim)}, expected {expected}"
            )
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape != (batch, tokens):
            raise ValueError(
                f"Expected position_ids {(batch, tokens)}, got {tuple(position_ids.shape)}"
            )

        output: CacheTensors = []
        positions = position_ids.to(self.device)
        for receiver_layer, transform in enumerate(self.transforms):
            sender_layer = int(self.layer_mapping[receiver_layer].item())
            key, value = sender_cache[sender_layer]
            key = key.to(device=self.device, dtype=self.dtype)
            value = value.to(device=self.device, dtype=self.dtype)
            rope_dtype = key.dtype
            content_key = apply_rope(
                key.float(), positions, self.sender_spec.rope_theta, inverse=True
            ).to(rope_dtype)
            projected_key, projected_value = transform(content_key, value)
            receiver_key = apply_rope(
                projected_key.float(),
                positions,
                self.receiver_spec.rope_theta,
            ).to(projected_key.dtype)
            output.append((receiver_key, projected_value))
        return output

    def gate_values(self) -> torch.Tensor:
        return torch.stack([layer.gate_logits.sigmoid() for layer in self.transforms])
