"""Pre-RMSNorm decoder blocks with separate attention and expert routers."""

import torch
from torch import nn
from torch.nn import functional as F

from .cache import LayerCache
from .config import LaMoEConfig
from .feedforward import MixtureOfExperts, SwiGLU
from .gated_deltanet import GatedDeltaNet
from .mla import MultiHeadLatentAttention
from .normalization import RMSNorm


class DecoderLayer(nn.Module):
    def __init__(self, config: LaMoEConfig, layer_number: int):
        super().__init__()
        self.layer_number = layer_number
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mla = MultiHeadLatentAttention(config)
        self.gdn = (
            GatedDeltaNet(config) if layer_number in config.hybrid_layers else None
        )
        self.attention_router = (
            nn.Linear(config.hidden_size, 1, bias=True)
            if self.gdn is not None
            else None
        )
        self.ffn = (
            MixtureOfExperts(config)
            if layer_number % 2 == 0
            else SwiGLU(config.hidden_size, config.dense_ffn_width)
        )

    def forward(self, x, layout, cache: LayerCache | None = None, use_cache=False):
        u = self.attention_norm(x)
        attention, mla_cache = self.mla(
            u, layout, None if cache is None else cache.mla, use_cache
        )
        gdn_cache, gate_mean = None, None
        if self.gdn is not None:
            recurrent, gdn_cache = self.gdn(
                u, layout, None if cache is None else cache.gdn, use_cache
            )
            with torch.autocast(device_type=x.device.type, enabled=False):
                gate = F.linear(
                    u.float(),
                    self.attention_router.weight.float(),
                    self.attention_router.bias.float(),
                ).sigmoid()
            gate = gate.to(attention.dtype)
            if getattr(self, "collect_diagnostics", False):
                from lamoe.train.diagnostics import distribution

                valid = layout.valid
                self.diagnostics = distribution(gate[valid])
                self.diagnostics.update(
                    weighted_mla_rms=float(
                        ((1 - gate) * attention)
                        .detach()
                        .float()[valid]
                        .square()
                        .mean()
                        .sqrt()
                    ),
                    weighted_gdn_rms=float(
                        (gate * recurrent)
                        .detach()
                        .float()[valid]
                        .square()
                        .mean()
                        .sqrt()
                    ),
                )
            attention = (1 - gate) * attention + gate * recurrent
            gate_mean = (
                gate.detach().float().squeeze(-1) * layout.valid
            ).sum() / layout.valid.sum().clamp_min(1)
        x = x + attention
        v = self.ffn_norm(x)
        statistics = None
        if isinstance(self.ffn, MixtureOfExperts):
            ffn, statistics = self.ffn(v, layout.valid)
        else:
            ffn = self.ffn(v)
        x = (x + ffn) * layout.valid[..., None]
        updated = LayerCache(mla_cache, gdn_cache) if use_cache else None
        return x, updated, statistics, gate_mean
