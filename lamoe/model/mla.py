"""Multi-head latent attention with a compressed inference cache."""

import torch
from torch import nn
from torch.nn import functional as F

from .cache import MLACache
from .config import LaMoEConfig
from .normalization import RMSNorm
from .rotary import RotaryEmbedding


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: LaMoEConfig):
        super().__init__()
        self.config = config
        d, heads = config.hidden_size, config.mla_heads
        self.q_down = nn.Linear(d, config.q_rank, bias=False)
        self.q_norm = RMSNorm(config.q_rank, config.rms_norm_eps)
        self.q_up = nn.Linear(
            config.q_rank,
            heads * (config.qk_content_dim + config.qk_rope_dim),
            bias=False,
        )
        self.kv_down = nn.Linear(d, config.kv_rank, bias=False)
        self.kv_norm = RMSNorm(config.kv_rank, config.rms_norm_eps)
        self.k_up = nn.Linear(config.kv_rank, heads * config.qk_content_dim, bias=False)
        self.v_up = nn.Linear(config.kv_rank, heads * config.value_head_dim, bias=False)
        self.k_rope = nn.Linear(d, config.qk_rope_dim, bias=False)
        self.out_proj = nn.Linear(heads * config.value_head_dim, d, bias=False)
        self.rope = RotaryEmbedding(config.qk_rope_dim, config.rope_theta)
        self.head_gate = (
            nn.Linear(d, heads, bias=True) if config.mla_head_gate else None
        )

    def forward(self, u, layout, cache: MLACache | None = None, use_cache=False):
        cfg = self.config
        batch, length, _ = u.shape
        query = self.q_up(self.q_norm(self.q_down(u))).reshape(
            batch, length, cfg.mla_heads, -1
        )
        content_q, rotary_q = query.split((cfg.qk_content_dim, cfg.qk_rope_dim), dim=-1)
        query = torch.cat((content_q, self.rope(rotary_q, layout.position_ids)), dim=-1)
        latent = self.kv_norm(self.kv_down(u))
        rotary_k = self.rope(self.k_rope(u).unsqueeze(2), layout.position_ids).squeeze(
            2
        )
        if cache is not None:
            if (
                cache.latent.dtype != latent.dtype
                or cache.rotary_key.dtype != rotary_k.dtype
            ):
                raise ValueError(
                    "Keep the same precision/autocast mode for every cached call"
                )
            latent = torch.cat((cache.latent, latent), dim=1)
            rotary_k = torch.cat((cache.rotary_key, rotary_k), dim=1)
        key = self.k_up(latent).reshape(batch, -1, cfg.mla_heads, cfg.qk_content_dim)
        key = torch.cat(
            (key, rotary_k.unsqueeze(2).expand(-1, -1, cfg.mla_heads, -1)), dim=-1
        )
        value = self.v_up(latent).reshape(batch, -1, cfg.mla_heads, cfg.value_head_dim)
        if cfg.mla_backend == "flash_varlen" and not use_cache:
            result = flash_varlen(query, key, value, layout)
        else:
            result = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=layout.attention_bias,
                is_causal=layout.is_causal,
                dropout_p=0.0,
            ).transpose(1, 2)
        if self.head_gate is not None:
            with torch.autocast(device_type=u.device.type, enabled=False):
                gate = F.linear(
                    u.float(),
                    self.head_gate.weight.float(),
                    self.head_gate.bias.float(),
                ).sigmoid()
            result = result * gate.to(result.dtype).unsqueeze(-1)
        result = self.out_proj(result.reshape(batch, length, -1))
        result = result * layout.valid[..., None]
        updated = MLACache(latent.detach(), rotary_k.detach()) if use_cache else None
        return result, updated


def flash_varlen(q, k, v, layout):
    """FA2 packed independent causal documents, never an unmasked fallback."""
    if q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("flash_varlen requires CUDA BF16/FP16 autocast")
    from flash_attn import flash_attn_varlen_func

    indices = layout.valid.flatten().nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return v * 0
    starts = layout.reset.clone()
    starts[:, 0] = layout.valid[:, 0]
    boundaries = starts.flatten()[indices].nonzero(as_tuple=True)[0]
    cu = torch.cat((boundaries, boundaries.new_tensor([indices.numel()]))).to(
        torch.int32
    )
    maximum = int((cu[1:] - cu[:-1]).max())
    pack = lambda t: t.flatten(0, 1).index_select(0, indices).contiguous()
    out = flash_attn_varlen_func(
        pack(q), pack(k), pack(v), cu, cu, maximum, maximum, dropout_p=0.0, causal=True
    )
    return torch.zeros_like(v.flatten(0, 1)).index_copy(0, indices, out).reshape_as(v)
