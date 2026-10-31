"""Optional Transformers integration. Importing does not load a model or use the Hub."""

from .configuration_lamoe import LaMoEConfig
from .modeling_lamoe import LaMoEForCausalLM


def register():
    """Register the locally installed implementation with Transformers AutoClasses."""
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("lamoe", LaMoEConfig, exist_ok=True)
    AutoModelForCausalLM.register(LaMoEConfig, LaMoEForCausalLM, exist_ok=True)


__all__ = ["LaMoEConfig", "LaMoEForCausalLM", "register"]
