"""LaMoE causal language model, initialization, checkpoint IO, and diagnostics."""

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from .cache import ModelCache
from .config import LaMoEConfig
from .feedforward import MixtureOfExperts, RouterStatistics, SwiGLU
from .gated_deltanet import GatedDeltaNet
from .layers import DecoderLayer
from .losses import causal_language_model_loss, chunked_linear_loss, router_losses
from .masking import build_layout
from .normalization import RMSNorm


@dataclass
class LaMoEOutput:
    logits: torch.Tensor | None
    loss: torch.Tensor | None
    ce_loss: torch.Tensor | None
    ce_loss_sum: torch.Tensor | None
    target_token_count: torch.Tensor | None
    input_token_count: torch.Tensor
    balance_loss: torch.Tensor
    router_z_loss: torch.Tensor
    router_statistics: tuple[RouterStatistics, ...]
    fusion_gate_means: tuple[torch.Tensor, ...]
    past_key_values: ModelCache | None
    _counts_accumulated: bool = field(default=False, repr=False)


class LaMoEForCausalLM(nn.Module):
    def __init__(self, config: LaMoEConfig | None = None):
        super().__init__()
        self.config = config or LaMoEConfig()
        self.gradient_checkpointing = False
        cfg = self.config
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, number) for number in range(1, cfg.num_layers + 1)]
        )
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.register_buffer(
            "balance_counts",
            torch.zeros(cfg.num_moe_layers, cfg.num_experts, dtype=torch.float32),
        )
        self.reset_parameters()

    def reset_parameters(self):
        cfg = self.config
        seen = set()
        # The tied matrix is initialized once, as well as counted once.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if id(module.weight) not in seen:
                    nn.init.normal_(module.weight, std=cfg.initializer_std)
                    seen.add(id(module.weight))
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        residual_std = cfg.initializer_std / math.sqrt(2 * cfg.num_layers)
        for module in self.modules():
            if isinstance(module, SwiGLU):
                nn.init.normal_(module.down_proj.weight, std=residual_std)
            elif isinstance(module, GatedDeltaNet):
                module.reset_dynamics()
                nn.init.normal_(module.out_proj.weight, std=residual_std)
            elif isinstance(module, MixtureOfExperts):
                nn.init.normal_(module.router.weight, std=cfg.router_initializer_std)
        for layer in self.layers:
            nn.init.normal_(layer.mla.out_proj.weight, std=residual_std)
            if layer.attention_router is not None:
                nn.init.zeros_(layer.attention_router.weight)
                nn.init.constant_(
                    layer.attention_router.bias,
                    math.log(cfg.fusion_gate_init / (1 - cfg.fusion_gate_init)),
                )
            if layer.mla.head_gate is not None:
                nn.init.zeros_(layer.mla.head_gate.weight)
                nn.init.constant_(layer.mla.head_gate.bias, cfg.head_gate_bias_init)
        self.reset_router_counts()

    def _validate_cache(self, cache, input_ids):
        cfg = self.config
        batch = input_ids.shape[0]
        if not isinstance(cache, ModelCache) or len(cache.layers) != cfg.num_layers:
            raise ValueError("Cache must contain one entry per decoder layer")
        if (
            cache.document_ids.ndim != 2
            or cache.document_ids.shape[0] != batch
            or cache.length == 0
        ):
            raise ValueError("Cache batch shape or length is invalid")
        tensors = [cache.document_ids, cache.last_segment_ids, cache.last_position_ids]
        if cache.last_segment_ids.shape != (
            batch,
        ) or cache.last_position_ids.shape != (batch,):
            raise ValueError("Cache metadata shape is invalid")
        if any(t.dtype != torch.long for t in tensors):
            raise ValueError("Cache metadata must use int64")
        for layer, item in zip(self.layers, cache.layers):
            if item.mla.latent.shape != (
                batch,
                cache.length,
                cfg.kv_rank,
            ) or item.mla.rotary_key.shape != (batch, cache.length, cfg.qk_rope_dim):
                raise ValueError("MLA cache shape does not match model configuration")
            tensors.extend((item.mla.latent, item.mla.rotary_key))
            if (item.gdn is None) != (layer.gdn is None):
                raise ValueError("GDN cache placement does not match hybrid layers")
            if item.gdn is not None:
                g = item.gdn
                key_shape = (
                    batch,
                    cfg.gdn_conv_width - 1,
                    cfg.gdn_heads * cfg.gdn_key_dim,
                )
                val_shape = (
                    batch,
                    cfg.gdn_conv_width - 1,
                    cfg.gdn_heads * cfg.gdn_value_dim,
                )
                if (
                    g.state.shape
                    != (batch, cfg.gdn_heads, cfg.gdn_value_dim, cfg.gdn_key_dim)
                    or g.state.dtype != torch.float32
                ):
                    raise ValueError("GDN cache must contain FP32 value-by-key states")
                if (
                    g.conv_q.shape != key_shape
                    or g.conv_k.shape != key_shape
                    or g.conv_v.shape != val_shape
                ):
                    raise ValueError("GDN convolution cache shape is invalid")
                tensors.extend((g.state, g.conv_q, g.conv_k, g.conv_v))
        if any(t.device != input_ids.device for t in tensors):
            raise ValueError("Cache and input_ids must be on the same device")

    def forward(
        self,
        input_ids,
        *,
        attention_mask=None,
        segment_ids=None,
        labels=None,
        past_key_values: ModelCache | None = None,
        use_cache=False,
        logits_to_keep=0,
        loss_chunk_size=0,
        return_logits=True,
    ):
        """Run packed pretraining/SFT or unpadded cached inference.

        segment_ids change at document boundaries within each row. Labels are
        unshifted input IDs, with -100 at unsupervised positions. logits_to_keep
        is an inference-only option (zero means every position).
        """
        if (
            input_ids.ndim != 2
            or input_ids.dtype != torch.long
            or not all(input_ids.shape)
        ):
            raise ValueError(
                "input_ids must be a nonempty int64 [batch, length] tensor"
            )
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if use_cache and (
            self.training or torch.is_grad_enabled() or labels is not None
        ):
            raise ValueError(
                "Caching is for model.eval() inside torch.no_grad()/inference_mode(), without labels"
            )
        if self.training and (
            self.embedding.weight.dtype != torch.float32
            or self.balance_counts.dtype != torch.float32
        ):
            raise ValueError(
                "Keep model parameters/buffers in FP32 for training; use BF16 autocast"
            )
        if (
            type(logits_to_keep) is not int
            or logits_to_keep < 0
            or (logits_to_keep and labels is not None)
        ):
            raise ValueError(
                "logits_to_keep must be nonnegative and cannot be used with labels"
            )
        if past_key_values is not None:
            self._validate_cache(past_key_values, input_ids)
        past_length = 0 if past_key_values is None else past_key_values.length
        if past_length + input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError(
                "Input plus cache exceeds max_position_embeddings; start a new context"
            )
        if type(loss_chunk_size) is not int or loss_chunk_size < 0:
            raise ValueError("loss_chunk_size must be a nonnegative integer")
        if not return_logits and (labels is None or not loss_chunk_size):
            raise ValueError("Omitting logits requires labels and chunked loss")
        layout = build_layout(
            input_ids,
            attention_mask,
            segment_ids,
            past_key_values,
            materialize_attention=use_cache
            or self.config.mla_backend != "flash_varlen",
        )
        if use_cache and not layout.valid.all():
            raise ValueError("Cached inference currently requires unpadded inputs")
        x = self.embedding(input_ids) * layout.valid[..., None]
        caches, routers, gates = [], [], []
        for index, layer in enumerate(self.layers):
            past_layer = (
                None if past_key_values is None else past_key_values.layers[index]
            )
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint

                x, updated, statistics, gate = checkpoint(
                    layer, x, layout, past_layer, use_cache, use_reentrant=False
                )
            else:
                x, updated, statistics, gate = layer(x, layout, past_layer, use_cache)
            if use_cache:
                caches.append(updated)
            if statistics is not None:
                routers.append(statistics)
            if gate is not None:
                gates.append(gate)
        x = self.final_norm(x)
        logits = (
            self.lm_head(x[:, -logits_to_keep:] if logits_to_keep else x)
            if return_logits
            else None
        )
        zero = x.float().sum() * 0
        history_counts = (
            self.balance_counts
            if self.training
            else torch.zeros_like(self.balance_counts)
        )
        balance, router_z = router_losses(routers, history_counts, zero)
        loss = ce = ce_sum = target_count = None
        if labels is not None:
            if loss_chunk_size:
                ce, ce_sum, target_count = chunked_linear_loss(
                    x, self.lm_head.weight, labels, layout, loss_chunk_size
                )
            else:
                ce, ce_sum, target_count = causal_language_model_loss(
                    logits, labels, layout
                )
            loss = (
                ce
                + self.config.load_balance_coef * balance
                + self.config.router_z_coef * router_z
            )
        updated_cache = None
        if use_cache:
            all_docs = (
                layout.document_ids
                if past_key_values is None
                else torch.cat(
                    (past_key_values.document_ids, layout.document_ids), dim=1
                )
            )
            updated_cache = ModelCache(
                tuple(caches),
                all_docs.detach(),
                layout.segment_ids[:, -1].detach(),
                layout.position_ids[:, -1].detach(),
            )
        return LaMoEOutput(
            logits,
            loss,
            ce,
            ce_sum,
            target_count,
            layout.valid.sum(),
            balance,
            router_z,
            tuple(routers),
            tuple(gates),
            updated_cache,
        )

    @torch.no_grad()
    def accumulate_router_counts(self, output: LaMoEOutput):
        """Call once per training microbatch, outside forward/checkpoint replay."""
        if not self.training:
            raise ValueError("Router count accumulation is only for training")
        if output._counts_accumulated:
            raise ValueError("This microbatch's router counts were already accumulated")
        if len(output.router_statistics) != self.config.num_moe_layers:
            raise ValueError("Output router count does not match this model")
        for buffer, stats in zip(self.balance_counts, output.router_statistics):
            buffer.add_(stats.assignment_counts.detach())
        output._counts_accumulated = True

    @torch.no_grad()
    def reset_router_counts(self):
        """Call after every optimizer step, including a discarded/aborted window."""
        self.balance_counts.zero_()

    def parameter_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        inactive = (
            self.config.num_moe_layers
            * (self.config.num_experts - self.config.experts_per_token)
            * 3
            * self.config.hidden_size
            * self.config.expert_ffn_width
        )
        return {
            "total_parameters": total,
            "active_path_parameters": total - inactive,
            "layers": self.config.num_layers,
            "hybrid_layers": list(self.config.hybrid_layers),
            "moe_layers": list(range(2, self.config.num_layers + 1, 2)),
            "mla_head_gate": self.config.mla_head_gate,
            "tied_embeddings": self.lm_head.weight is self.embedding.weight,
        }

    def save_pretrained(self, directory: str | Path):
        """Local model/config snapshot, not a resumable trainer checkpoint."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory / "config.json")
        from safetensors.torch import save_model

        save_model(self, str(directory / "model.safetensors"))

    @classmethod
    def from_pretrained(cls, directory: str | Path):
        directory = Path(directory)
        model = cls(LaMoEConfig.load(directory / "config.json"))
        if (directory / "model.safetensors").exists():
            from safetensors.torch import load_model

            load_model(model, str(directory / "model.safetensors"), strict=True)
            return model
        state = torch.load(
            directory / "model.pt", map_location="cpu", weights_only=True
        )
        if not torch.equal(state["embedding.weight"], state["lm_head.weight"]):
            raise ValueError(
                "Checkpoint embedding and output weights disagree despite weight tying"
            )
        model.load_state_dict(state, strict=True)
        return model
