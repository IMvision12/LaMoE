"""Transformers adapter for the original LaMoE modules and weight names."""

import copy

import torch
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from .caching_lamoe import LaMoECache
from .configuration_lamoe import LaMoEConfig
from ..model.masking import build_layout
from ..model.modeling_lamoe import LaMoEForCausalLM as CoreModel

logger = logging.get_logger(__name__)


class LaMoEForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LaMoEConfig
    base_model_prefix = ""
    main_input_name = "input_ids"
    _no_split_modules = ["DecoderLayer"]
    _tied_weights_keys = {"lm_head.weight": "embedding.weight"}
    _supports_sdpa = False

    def __init__(self, config):
        super().__init__(config)
        core = CoreModel(config.to_core_config())
        # Keep the checkpoint's original parameter names, so its safetensors
        # file can be copied byte-for-byte without rewriting trained weights.
        self.embedding = core.embedding
        self.layers = core.layers
        self.final_norm = core.final_norm
        self.lm_head = core.lm_head
        self.register_buffer("balance_counts", core.balance_counts)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        if value.weight.shape != self.embedding.weight.shape:
            raise ValueError(
                "Vocabulary resizing is not supported by this frozen export"
            )
        self.embedding = value
        self.lm_head.weight = value.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        if value.weight.shape != self.embedding.weight.shape:
            raise ValueError(
                "Output embedding shape disagrees with the frozen vocabulary"
            )
        self.lm_head = value
        self.embedding.weight = value.weight

    def _validate_cache(self, cache, input_ids):
        return CoreModel._validate_cache(self, cache, input_ids)

    def reset_router_counts(self):
        return CoreModel.reset_router_counts(self)

    def accumulate_router_counts(self, output):
        return CoreModel.accumulate_router_counts(self, output)

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
        logits_to_keep=0,
        segment_ids=None,
        token_type_ids=None,
        **kwargs,
    ):
        if inputs_embeds is not None or input_ids is None:
            raise ValueError(
                "LaMoE currently requires input_ids; inputs_embeds is unsupported"
            )
        if output_attentions or output_hidden_states:
            raise NotImplementedError(
                "This adapter does not expose attention maps or intermediate hidden states"
            )
        if token_type_ids is not None and bool((token_type_ids != 0).any()):
            raise ValueError("LaMoE does not use token type IDs")
        # Current Transformers may pass num_items_in_batch with labels. The
        # original model computes its own valid-target loss normalization.
        unknown = set(kwargs) - {"num_items_in_batch"}
        if unknown:
            raise TypeError(f"Unsupported LaMoE forward arguments: {sorted(unknown)}")
        if past_key_values is not None and not isinstance(past_key_values, LaMoECache):
            raise TypeError("Use LaMoECache, not a standard attention KV cache")
        core_past = None if past_key_values is None else past_key_values.state
        use_cache = bool(self.config.use_cache if use_cache is None else use_cache)
        if labels is not None or self.training:
            if core_past is not None:
                raise ValueError(
                    "Training/labeled loss does not accept an inference cache"
                )
            use_cache = False
        elif torch.is_grad_enabled() and core_past is None:
            use_cache = False
        current_mask = attention_mask
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            past_length = 0 if core_past is None else core_past.length
            if attention_mask.shape != (
                input_ids.shape[0],
                past_length + input_ids.shape[1],
            ):
                raise ValueError(
                    "attention_mask must cover the current input or full cached context"
                )
            if core_past is not None and not bool(
                attention_mask[:, :past_length].all()
            ):
                raise ValueError("Cached prefixes must be unpadded")
            current_mask = attention_mask[:, -input_ids.shape[1] :]
        if (
            current_mask is not None
            and not bool(current_mask.bool().all())
            and use_cache
        ):
            if core_past is not None:
                raise ValueError("Padded inputs cannot be added to an existing cache")
            logger.warning_once(
                "LaMoE padded batches use full-sequence decoding without a cache."
            )
            use_cache = False
        if position_ids is not None:
            canonical = build_layout(
                input_ids,
                current_mask,
                segment_ids,
                core_past,
                materialize_attention=False,
            ).position_ids
            if position_ids.shape != canonical.shape or not torch.equal(
                position_ids, canonical
            ):
                raise ValueError(
                    "Explicit position_ids must match LaMoE's document-aware positions"
                )
        out = CoreModel.forward(
            self,
            input_ids,
            attention_mask=current_mask,
            segment_ids=segment_ids,
            labels=labels,
            past_key_values=core_past,
            use_cache=use_cache,
            logits_to_keep=0 if labels is not None else logits_to_keep,
        )
        cache = None
        if out.past_key_values is not None:
            cache = (
                past_key_values
                if past_key_values is not None
                else LaMoECache(max_length=self.config.max_position_embeddings)
            )
            cache.state = out.past_key_values
        if not return_dict:
            result = (out.logits, cache) if cache is not None else (out.logits,)
            return ((out.loss,) + result) if out.loss is not None else result
        return CausalLMOutputWithPast(
            loss=out.loss, logits=out.logits, past_key_values=cache
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=True,
        **kwargs,
    ):
        length = 0 if past_key_values is None else past_key_values.get_seq_length()
        if length and input_ids.shape[1] > length:
            input_ids = input_ids[:, length:]
        if length and input_ids.shape[1] != 1:
            # A caller can directly forward a chunk with a cache, but the public
            # generate interface appends one token per decoding iteration.
            raise ValueError(
                "Generation with a cache requires one newly appended token"
            )
        result = dict(
            input_ids=input_ids.contiguous(),
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            logits_to_keep=kwargs.get("logits_to_keep", 1),
        )
        if kwargs.get("position_ids") is not None:
            result["position_ids"] = kwargs["position_ids"][:, -input_ids.shape[1] :]
        return result

    @torch.no_grad()
    def generate(self, inputs=None, generation_config=None, **kwargs):
        cfg = copy.deepcopy(generation_config or self.generation_config)
        if kwargs.get("assistant_model") is not None or cfg.is_assistant:
            raise NotImplementedError(
                "Assisted/speculative decoding requires rollback of recurrent states and is unsupported"
            )
        if kwargs.get("cache_implementation", cfg.cache_implementation) is not None:
            raise ValueError(
                "LaMoE uses its own recurrent/latent cache; cache_implementation must be unset"
            )
        ids = kwargs.get("input_ids", inputs)
        if ids is None:
            ids = torch.tensor([[self.config.bos_token_id]], device=self.device)
            inputs = ids
        past = kwargs.get("past_key_values")
        prefix_length = ids.shape[1]
        if past is not None:
            prefix_length = max(prefix_length, past.get_seq_length() + 1)
        remaining = self.config.max_position_embeddings - prefix_length
        if remaining <= 0:
            raise ValueError(
                "Prompt reaches/exceeds LaMoE's context; it will not be truncated"
            )
        requested = kwargs.get("max_new_tokens", cfg.max_new_tokens)
        if requested is not None:
            if type(requested) is not int or requested < 1:
                raise ValueError("max_new_tokens must be a positive integer")
            if requested > remaining:
                logger.warning_once(
                    "LaMoE caps output to the remaining context allowance."
                )
            kwargs["max_new_tokens"] = min(requested, remaining)
        else:
            kwargs["max_length"] = min(
                kwargs.get("max_length", cfg.max_length),
                self.config.max_position_embeddings,
            )
        model_kwargs = cfg.update(**kwargs)
        return super().generate(inputs=inputs, generation_config=cfg, **model_kwargs)
