"""Inference state. MLA keeps latents; GDN keeps recurrent and convolution state."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MLACache:
    latent: torch.Tensor  # [batch, past_length, kv_rank], after RMSNorm
    rotary_key: torch.Tensor  # [batch, past_length, rotary_dim], after RoPE


@dataclass(frozen=True)
class GDNCache:
    state: torch.Tensor  # [batch, heads, value_dim, key_dim], FP32
    conv_q: torch.Tensor  # [batch, kernel_width - 1, heads * key_dim]
    conv_k: torch.Tensor
    conv_v: torch.Tensor


@dataclass(frozen=True)
class LayerCache:
    mla: MLACache
    gdn: GDNCache | None = None


@dataclass(frozen=True)
class ModelCache:
    layers: tuple[LayerCache, ...]
    document_ids: torch.Tensor  # [batch, past_length], internal contiguous runs
    last_segment_ids: torch.Tensor
    last_position_ids: torch.Tensor

    @property
    def length(self) -> int:
        return self.document_ids.shape[1]
