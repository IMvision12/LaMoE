"""Transformers configuration for the unchanged LaMoE architecture."""

from dataclasses import fields

from transformers import PreTrainedConfig

from ..model.config import LaMoEConfig as CoreConfig


class LaMoEConfig(PreTrainedConfig):
    model_type = "lamoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "num_layers",
        "num_attention_heads": "mla_heads",
    }

    def __init__(self, **kwargs):
        defaults = CoreConfig().to_dict()
        defaults["gdn_backend"] = "reference"
        values = {name: kwargs.pop(name, default) for name, default in defaults.items()}
        validated = CoreConfig(**values)
        if kwargs.pop("tie_word_embeddings", True) is not True:
            raise ValueError("LaMoE requires tied embedding/output weights")
        if kwargs.pop("is_encoder_decoder", False):
            raise ValueError("LaMoE is a decoder-only model")
        kwargs.pop("is_decoder", None)
        use_cache = kwargs.pop("use_cache", True)
        super().__init__(
            tie_word_embeddings=True,
            is_encoder_decoder=False,
            is_decoder=True,
            bos_token_id=kwargs.pop("bos_token_id", 2),
            eos_token_id=kwargs.pop("eos_token_id", [3, 10]),
            pad_token_id=kwargs.pop("pad_token_id", 0),
            **kwargs,
        )
        for name, value in validated.to_dict().items():
            setattr(self, name, value)
        self.use_cache = use_cache

    @property
    def num_moe_layers(self):
        return self.num_layers // 2

    def to_core_config(self):
        return CoreConfig(
            **{item.name: getattr(self, item.name) for item in fields(CoreConfig)}
        )
