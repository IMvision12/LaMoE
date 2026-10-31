"""Gated DeltaNet branch with explicit decay, write, and document resets.

The differentiable PyTorch recurrence is the CPU correctness implementation.
CUDA auto mode requires FLA's chunk_gated_delta_rule; no slow silent fallback.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .cache import GDNCache
from .config import LaMoEConfig
from .normalization import RMSNorm


class CausalDepthwiseConv(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(torch.empty(channels, 1, width))
        nn.init.uniform_(self.weight, -1 / math.sqrt(width), 1 / math.sqrt(width))

    def forward(self, x, layout, history=None):
        batch, length, channels = x.shape
        keep = self.width - 1
        if history is None:
            history = x.new_zeros(batch, keep, channels)
        full = torch.cat((history, x * layout.valid[..., None]), dim=1)
        result = torch.zeros_like(x)
        for tap in range(self.width):
            lag = self.width - tap - 1
            permitted = layout.valid & (layout.position_ids >= lag)
            result = (
                result
                + full[:, tap : tap + length]
                * self.weight[:, 0, tap].to(x.dtype)
                * permitted[..., None]
            )
        tail = full[:, -keep:] if keep else full[:, :0]
        if keep:
            lags = torch.arange(keep - 1, -1, -1, device=x.device)
            tail = tail * (lags[None, :] <= layout.position_ids[:, -1:])[..., None]
        return F.silu(result), tail


def reference_delta_rule(q, k, v, log_decay, beta, layout, initial_state=None):
    """Sequential FP32 reference; state is [batch, heads, value, key]."""
    batch, length, heads, key_dim = q.shape
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v = q.float(), k.float(), v.float()
        state = (
            q.new_zeros(batch, heads, v.shape[-1], key_dim)
            if initial_state is None
            else initial_state.float()
        )
        outputs = []
        for t in range(length):
            state = torch.where(layout.reset[:, t, None, None, None], 0.0, state)
            decayed = state * log_decay[:, t].exp()[..., None, None]
            error = v[:, t] - (decayed * k[:, t, :, None, :]).sum(-1)
            updated = (
                decayed
                + beta[:, t, :, None, None] * error[..., None] * k[:, t, :, None, :]
            )
            state = torch.where(layout.valid[:, t, None, None, None], updated, state)
            read = (state * q[:, t, :, None, :]).sum(-1) * (key_dim**-0.5)
            outputs.append(read * layout.valid[:, t, None, None])
        return torch.stack(outputs, dim=1), state


def fla_delta_rule(
    q, k, v, log_decay, beta, layout, initial_state=None, use_cache=False
):
    """Unpad runs into independent FLA sequences; restore output batch shape.

    API reference: fla/ops/gated_delta_rule/chunk.py, checked 2026-09-27.
    The CUDA path still requires numerical comparison and profiling on the A100.
    """
    if q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            "FLA requires CUDA and BF16/FP16 autocast; use reference for CPU checks"
        )
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        raise RuntimeError(
            "Install a compatible flash-linear-attention on the GPU host, or explicitly select gdn_backend='reference' for debugging"
        ) from exc
    batch, length, heads, key_dim = q.shape
    indices = layout.valid.reshape(-1).nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return v * 0, v.new_zeros(
            batch, heads, v.shape[-1], key_dim, dtype=torch.float32
        )
    starts = layout.reset.clone()
    starts[:, 0] = layout.valid[:, 0]
    flat_starts = starts.reshape(-1)[indices].nonzero(as_tuple=True)[0]
    cu_seqlens = torch.cat((flat_starts, flat_starts.new_tensor([indices.numel()]))).to(
        torch.int32
    )

    def pack(x):
        return x.flatten(0, 1).index_select(0, indices).unsqueeze(0).contiguous()

    state_in = None
    if initial_state is not None:
        original_starts = indices[flat_starts]
        rows = original_starts // length
        continuing = (original_starts % length == 0) & ~layout.reset[rows, 0]
        state_in = initial_state[rows] * continuing[:, None, None, None]
    out, state_out = chunk_gated_delta_rule(
        q=pack(q),
        k=pack(k),
        v=pack(v),
        g=pack(log_decay),
        beta=pack(beta),
        scale=key_dim**-0.5,
        initial_state=state_in,
        output_final_state=use_cache,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
    )
    out = v.new_zeros(batch * length, heads, v.shape[-1]).index_copy(
        0, indices, out.squeeze(0)
    )
    final = None
    if use_cache:
        # Cached inference is unpadded, so every row has at least one sequence.
        last_sequence = starts.sum(dim=1).cumsum(0) - 1
        final = state_out[last_sequence].float()
    return out.reshape_as(v), final


class GatedDeltaNet(nn.Module):
    def __init__(self, config: LaMoEConfig):
        super().__init__()
        self.config = config
        d, heads = config.hidden_size, config.gdn_heads
        key_width, value_width = (
            heads * config.gdn_key_dim,
            heads * config.gdn_value_dim,
        )
        self.q_proj = nn.Linear(d, key_width, bias=False)
        self.k_proj = nn.Linear(d, key_width, bias=False)
        self.v_proj = nn.Linear(d, value_width, bias=False)
        self.z_proj = nn.Linear(d, value_width, bias=False)
        self.a_proj = nn.Linear(d, heads, bias=False)
        self.beta_proj = nn.Linear(d, heads, bias=False)
        self.out_proj = nn.Linear(value_width, d, bias=False)
        self.conv_q = CausalDepthwiseConv(key_width, config.gdn_conv_width)
        self.conv_k = CausalDepthwiseConv(key_width, config.gdn_conv_width)
        self.conv_v = CausalDepthwiseConv(value_width, config.gdn_conv_width)
        self.A_log = nn.Parameter(torch.empty(heads))
        self.dt_bias = nn.Parameter(torch.empty(heads))
        self.output_norm = RMSNorm(config.gdn_value_dim, config.rms_norm_eps)
        self.reset_dynamics()

    def reset_dynamics(self):
        with torch.no_grad():
            self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1, 16).log())
            delta = (
                torch.empty_like(self.dt_bias)
                .uniform_(math.log(0.001), math.log(0.1))
                .exp()
            )
            self.dt_bias.copy_(delta + torch.log(-torch.expm1(-delta)))
            for conv in (self.conv_q, self.conv_k, self.conv_v):
                nn.init.uniform_(
                    conv.weight, -1 / math.sqrt(conv.width), 1 / math.sqrt(conv.width)
                )

    def forward(self, u, layout, cache: GDNCache | None = None, use_cache=False):
        cfg = self.config
        batch, length, _ = u.shape
        q, hist_q = self.conv_q(
            self.q_proj(u), layout, None if cache is None else cache.conv_q
        )
        k, hist_k = self.conv_k(
            self.k_proj(u), layout, None if cache is None else cache.conv_k
        )
        v, hist_v = self.conv_v(
            self.v_proj(u), layout, None if cache is None else cache.conv_v
        )
        q = q.reshape(batch, length, cfg.gdn_heads, cfg.gdn_key_dim)
        k = k.reshape_as(q)
        v = v.reshape(batch, length, cfg.gdn_heads, cfg.gdn_value_dim)
        with torch.autocast(device_type=u.device.type, enabled=False):
            q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(v.dtype)
            k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(v.dtype)
            decay_input = F.linear(u.float(), self.a_proj.weight.float())
            log_decay = -self.A_log.float().exp() * F.softplus(
                decay_input + self.dt_bias.float()
            )
            beta = F.linear(u.float(), self.beta_proj.weight.float()).sigmoid()
        initial = None if cache is None else cache.state
        backend = cfg.gdn_backend
        if backend == "auto":
            backend = "fla" if u.device.type == "cuda" else "reference"
        if backend == "fla":
            result, state = fla_delta_rule(
                q, k, v, log_decay, beta, layout, initial, use_cache
            )
        else:
            result, state = reference_delta_rule(
                q, k, v, log_decay, beta, layout, initial
            )
        z = self.z_proj(u).reshape_as(v)
        result = self.output_norm(result.to(v.dtype)) * F.silu(z)
        result = self.out_proj(result.flatten(2)) * layout.valid[..., None]
        updated = None
        if use_cache:
            updated = GDNCache(
                state.detach(), hist_q.detach(), hist_k.detach(), hist_v.detach()
            )
        return result, updated
