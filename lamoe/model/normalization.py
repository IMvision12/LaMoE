"""RMS normalization with stable accumulation and no additive bias."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        accumulation_dtype = (
            torch.float64 if x.dtype == torch.float64 else torch.float32
        )
        value = x.to(accumulation_dtype)
        value = value * torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (value * self.weight.to(accumulation_dtype)).to(x.dtype)
