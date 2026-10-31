"""LaMoE architecture; imports PyTorch only when this package is imported."""

from .cache import ModelCache
from .config import LaMoEConfig
from .modeling_lamoe import LaMoEForCausalLM, LaMoEOutput

__all__ = ["LaMoEConfig", "LaMoEForCausalLM", "LaMoEOutput", "ModelCache"]
