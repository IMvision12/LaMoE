"""SwiGLU and a dropless top-k MoE with one always-active shared expert."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import LaMoEConfig


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass(frozen=True)
class RouterStatistics:
    assignment_counts: torch.Tensor
    probability_sum: torch.Tensor
    z_loss_sum: torch.Tensor
    token_count: int


class MixtureOfExperts(nn.Module):
    """Portable expert dispatch. Grouped GPU dispatch remains a performance task."""

    def __init__(self, config: LaMoEConfig):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.shared_expert = SwiGLU(config.hidden_size, config.shared_ffn_width)
        self.experts = nn.ModuleList(
            [
                SwiGLU(config.hidden_size, config.expert_ffn_width)
                for _ in range(config.num_experts)
            ]
        )

    def forward(self, x, valid):
        flat = x.reshape(-1, x.shape[-1])
        token_indices = valid.reshape(-1).nonzero(as_tuple=True)[0]
        tokens = flat.index_select(0, token_indices)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(tokens.float(), self.router.weight.float())
            chosen_logits, chosen_experts = logits.topk(
                self.config.experts_per_token, dim=-1
            )
            weights = chosen_logits.softmax(dim=-1)
            statistics = RouterStatistics(
                torch.bincount(
                    chosen_experts.flatten(), minlength=self.config.num_experts
                ).float(),
                logits.softmax(dim=-1).sum(0),
                logits.logsumexp(dim=-1).square().sum(),
                tokens.shape[0],
            )
        result = self.shared_expert(tokens)
        # No capacity limit: every selected token/expert pair is evaluated exactly once.
        if self.config.moe_backend == "batched" and tokens.shape[0]:
            result = result + self._batched(tokens, chosen_experts, weights)
        for expert_id, expert in enumerate(
            self.experts if self.config.moe_backend == "reference" else []
        ):
            selected_tokens, slots = (chosen_experts == expert_id).nonzero(
                as_tuple=True
            )
            if selected_tokens.numel():
                expert_output = expert(tokens.index_select(0, selected_tokens))
                contribution = expert_output * weights[selected_tokens, slots, None].to(
                    expert_output.dtype
                )
                result = result.index_add(0, selected_tokens, contribution)
        output = torch.zeros_like(flat).index_copy(
            0, token_indices, result.to(flat.dtype)
        )
        return output.reshape_as(x), statistics

    def _batched(self, tokens, choices, weights):
        # Dropless padded BMM. Only active experts are stacked so empty experts
        # retain grad=None and do not change optimizer momentum/decay behavior.
        e = choices.flatten()
        order = e.argsort(stable=True)
        counts = torch.bincount(e, minlength=len(self.experts))
        active = counts.nonzero(as_tuple=True)[0].tolist()
        c = counts[active]
        width = int(c.max())
        group = torch.repeat_interleave(torch.arange(len(active), device=e.device), c)
        start = c.cumsum(0) - c
        slot = torch.arange(e.numel(), device=e.device) - torch.repeat_interleave(
            start, c
        )
        tok = order // self.config.experts_per_token
        location = group * width + slot
        padded = tokens.new_zeros(len(active) * width, tokens.shape[-1]).index_copy(
            0, location, tokens[tok]
        )
        padded = padded.reshape(len(active), width, -1)

        def weight(name):
            return torch.stack(
                [getattr(self.experts[i], name).weight for i in active]
            ).transpose(1, 2)

        hidden = F.silu(torch.bmm(padded, weight("gate_proj"))) * torch.bmm(
            padded, weight("up_proj")
        )
        out = torch.bmm(hidden, weight("down_proj")).flatten(0, 1)[location]
        out = out * weights.flatten()[order, None].to(out.dtype)
        return torch.zeros_like(tokens).index_add(0, tok, out.to(tokens.dtype))
