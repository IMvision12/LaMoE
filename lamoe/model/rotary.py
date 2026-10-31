"""RoPE on the decoupled rotary coordinates of MLA only."""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x, positions):
        # Recompute the small frequency vector in FP32 even after model.bfloat16().
        with torch.autocast(device_type=x.device.type, enabled=False):
            inv_freq = self.theta ** (
                -torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)
                / self.dim
            )
            angle = positions.float()[..., None] * inv_freq
            cosine, sine = angle.cos()[:, :, None, :], angle.sin()[:, :, None, :]
            even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
            rotated = torch.stack(
                (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
            )
        return rotated.flatten(-2).to(x.dtype)
