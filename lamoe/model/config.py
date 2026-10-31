"""Architecture configuration for the LaMoE v0.4 specification."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class LaMoEConfig:
    vocab_size: int = 32_000
    hidden_size: int = 768
    num_layers: int = 16
    max_position_embeddings: int = 2048
    mla_heads: int = 12
    q_rank: int = 192
    kv_rank: int = 192
    qk_content_dim: int = 48
    qk_rope_dim: int = 16
    value_head_dim: int = 64
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    hybrid_layers: tuple[int, ...] = (4, 8, 12, 16)  # One-based.
    gdn_heads: int = 4
    gdn_key_dim: int = 64
    gdn_value_dim: int = 64
    gdn_conv_width: int = 4
    gdn_backend: str = "auto"
    mla_backend: str = "sdpa"
    moe_backend: str = "reference"
    fusion_gate_init: float = 0.1
    mla_head_gate: bool = False
    head_gate_bias_init: float | None = None
    dense_ffn_width: int = 2048
    num_experts: int = 8
    experts_per_token: int = 2
    expert_ffn_width: int = 1024
    shared_ffn_width: int = 512
    initializer_std: float = 0.02
    router_initializer_std: float = 0.01
    load_balance_coef: float = 0.01
    router_z_coef: float = 0.001

    def __post_init__(self):
        object.__setattr__(self, "hybrid_layers", tuple(self.hybrid_layers))
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is int and (type(value) is not int or value <= 0):
                raise ValueError(f"{field.name} must be a positive integer")
        if self.qk_rope_dim % 2:
            raise ValueError("qk_rope_dim must be even")
        if self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token exceeds num_experts")
        if len(set(self.hybrid_layers)) != len(self.hybrid_layers) or any(
            type(i) is not int or not 1 <= i <= self.num_layers
            for i in self.hybrid_layers
        ):
            raise ValueError(
                "hybrid_layers must contain unique, one-based layer numbers"
            )
        if self.gdn_backend not in ("auto", "reference", "fla"):
            raise ValueError("gdn_backend must be auto, reference, or fla")
        if self.mla_backend not in ("sdpa", "flash_varlen"):
            raise ValueError("mla_backend must be sdpa or flash_varlen")
        if self.moe_backend not in ("reference", "batched"):
            raise ValueError("moe_backend must be reference or batched")
        if not 0 < self.fusion_gate_init < 1:
            raise ValueError("fusion_gate_init must be strictly between zero and one")
        for name in (
            "rope_theta",
            "rms_norm_eps",
            "initializer_std",
            "router_initializer_std",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("load_balance_coef", "router_z_coef"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.mla_head_gate and self.head_gate_bias_init is None:
            raise ValueError(
                "Choose head_gate_bias_init explicitly when enabling the experimental MLA gate"
            )
        if self.head_gate_bias_init is not None and not math.isfinite(
            self.head_gate_bias_init
        ):
            raise ValueError("head_gate_bias_init must be finite")

    @property
    def num_moe_layers(self) -> int:
        return self.num_layers // 2

    @classmethod
    def tiny(cls, **overrides):
        """Small dimensions for development; preserves the four-layer pattern."""
        values = {
            "vocab_size": 128,
            "hidden_size": 32,
            "num_layers": 4,
            "max_position_embeddings": 64,
            "mla_heads": 2,
            "q_rank": 8,
            "kv_rank": 8,
            "qk_content_dim": 12,
            "qk_rope_dim": 4,
            "value_head_dim": 16,
            "hybrid_layers": (4,),
            "gdn_heads": 2,
            "gdn_key_dim": 8,
            "gdn_value_dim": 8,
            "dense_ffn_width": 64,
            "num_experts": 4,
            "expert_ffn_width": 32,
            "shared_ffn_width": 16,
            "gdn_backend": "reference",
        }
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
