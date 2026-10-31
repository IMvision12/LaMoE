# LaMoE model

Implementation of the LaMoE v0.4 architecture specification. All dimensions are
configurable through `LaMoEConfig`; the defaults describe the released 247.65M model.
This directory contains model code, not a tokenizer or a training launcher.

## Architecture

The model uses 16 pre-RMSNorm decoder blocks at width 768, a 32,000-token
vocabulary, tied embedding/output weights, and an initial context limit of 2,048.
Each group repeats with independent weights:

| Layer within group | Sequence mixing | Feed-forward |
| --- | --- | --- |
| 1 | MLA | Dense SwiGLU, width 2,048 |
| 2 | MLA | Shared expert plus top-2 of 8 routed experts |
| 3 | MLA | Dense SwiGLU, width 2,048 |
| 4 | Soft blend of MLA and Gated DeltaNet | Shared expert plus top-2 of 8 routed experts |

The attention router produces a scalar `g` per token and blends `(1-g)*MLA +
g*GDN`. Both branches execute and maintain history. The independent MoE router
selects FFN experts. Routed experts have width 1,024; the always-active shared
expert has width 512. No token is dropped due to expert capacity.

MLA uses 12 heads, query/KV ranks 192, content Q/K width 48, rotary Q/K width 16,
and value width 64 per head. Only the decoupled rotary coordinates receive RoPE.
GDN uses four heads with 64-by-64 states, bias-free causal convolutions of width
4, FP32 recurrence, and output normalization shared across its heads.

| Variant | Stored parameters | Active-path convention |
| --- | ---: | ---: |
| Baseline hybrid | 247,650,340 | 134,404,132 |
| Optional MLA head gate | 247,797,988 | 134,551,780 |

The active-path convention excludes unselected routed experts and includes the
entire embedding matrix. It does not measure FLOPs or wall time.

## Files

| File | Responsibility |
| --- | --- |
| `config.py` | Validated architecture settings and small development configuration |
| `normalization.py`, `rotary.py` | RMSNorm and decoupled RoPE |
| `mla.py` | MLA projections, attention, optional head gate, compressed KV cache |
| `gated_deltanet.py` | Boundary-aware convolutions, recurrence, optional FLA adapter |
| `feedforward.py` | SwiGLU, shared expert, dropless expert dispatch, router statistics |
| `layers.py` | Residual blocks and the separate attention/FFN routers |
| `masking.py`, `cache.py` | Document boundaries, causal masks, inference state |
| `losses.py` | Masked next-token CE, expert balance loss, router z-loss |
| `modeling_lamoe.py` | Complete causal LM, initialization, count buffers, local checkpoints |
| `__main__.py` | Parameter summary on the meta device, without weight allocation |

## Development usage

Run from the repository root with Python 3.10+ and PyTorch installed. Model
dependencies are in the root `requirements-model.txt`; data collection does not
require PyTorch. A CUDA installation should use the PyTorch wheel appropriate
to that host. Local validation uses PyTorch 2.14.0 CPU on Windows.

```python
import torch
from lamoe.model import LaMoEConfig, LaMoEForCausalLM

model = LaMoEForCausalLM(LaMoEConfig.tiny())  # 4 layers for local development
tokens = torch.randint(model.config.vocab_size, (2, 16))
output = model(tokens, labels=tokens)
output.loss.backward()
print(output.logits.shape, output.ce_loss.item())
print(model.parameter_summary())
```

For the full architecture use `LaMoEConfig()`. Keep FP32 master parameters during
training and wrap the forward pass in BF16 autocast. The constructor initializes
output projections with `0.02 / sqrt(2*num_layers)`, FFN routers with standard
deviation 0.01, and attention fusion with `g=0.1`. No extra normalization is
applied after the residual output projections.

```bash
python -m lamoe.model
python -m lamoe.model --tiny
python -m lamoe.model --mla-head-gate-bias 0.0
```

## Labels, packing, and padding

Pass **unshifted** labels; the model shifts them once for next-token prediction.
Set label positions to `-100` for prompts or other unsupervised tokens. CE excludes
padding and transitions between documents. For example:

```python
tokens = torch.randint(128, (1, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
labels = tokens.clone()
labels[:, :2] = -100
output = model(tokens, segment_ids=segments, labels=labels)
```

Each change in `segment_ids` starts a new document; a valid token following a
padding gap also starts a new document. IDs are local to each batch row and need
not match other rows. The model derives internal run IDs, resets positions, blocks
cross-document MLA attention, and resets GDN recurrence and convolution history.
Padding uses an `attention_mask` of shape `[batch, length]` containing 0/1.
Input IDs, including padded positions, must remain valid vocabulary IDs.

Router losses use all valid input tokens, including SFT prompts. CE uses its own
target mask. `target_token_count`, `ce_loss_sum`, and `input_token_count` are
returned so a trainer can weight unequal microbatches correctly. The last input
position has no next-token target within that batch. EOS labels are trained when
they occur inside the same document; EOS-to-next-document transitions are masked.

## Router losses and accumulation

The returned training loss is `CE + 0.01*balance + 0.001*router_z`, averaged over
MoE layers. Balance combines the detached running assignment counts plus the
current microbatch with the current microbatch's mean router probabilities.
Uniform usage gives balance loss 1. With one microbatch, this is microbatch balance.

Forward has no routing-buffer mutations. For accumulation, call
`model.accumulate_router_counts(output)` exactly once after each microbatch's
forward/backward, then call `model.reset_router_counts()` after the optimizer
step. Reset counts on an abandoned accumulation window too. The buffers are part
of `state_dict`; save them if resuming inside a window. Checkpoint recomputation
must never call the accumulation method. Calling it twice on one output raises.

For unequal microbatches, scale CE by valid target counts and auxiliary losses
by valid input counts across the update. Dividing the combined loss by a fixed
number of accumulation steps is correct only when the relevant counts are equal.
One can combine the separately returned losses to apply these weights.

## Cached inference

Use `model.eval()` inside `torch.inference_mode()`. The initial implementation
supports **unpadded** cached batches and rejects padding instead of guessing
history semantics. Keep the same autocast precision across all cached calls.

```python
model.eval()
with torch.inference_mode():
    first = model(tokens, use_cache=True, logits_to_keep=1)
    next_id = first.logits[:, -1].argmax(-1, keepdim=True)
    second = model(
        next_id, past_key_values=first.past_key_values, use_cache=True, logits_to_keep=1
    )
```

MLA caches normalized compressed KV latents and rotated shared keys; expanded
keys and values are reconstructed per call. GDN caches FP32 value-by-key states
and raw projection convolution histories. Cached chunks can contain document
boundaries. Omitted segment IDs continue the previous final document; pass a
different segment ID or start a new cache to begin another document. Both branches
update even if the attention gate is near zero or one. Old document latents may
remain allocated but are masked out. Total cached context is capped at 2,048 by
default; no implicit eviction or sliding window is implemented.

## Optional MLA head gate

Enable it explicitly with `LaMoEConfig(mla_head_gate=True,
head_gate_bias_init=0.0)`. The bias must be provided when enabling the gate; zero
is an example calibration setting, not an established optimum. It produces one
sigmoid weight per MLA head before `W_O` in all layers. Zero gate weights and zero
bias initialize the scale to 0.5. It is separate from the attention fusion router
and FFN expert router, adds 147,648 parameters, and does not skip computation.

## Backends and remaining GPU work

MLA uses [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
Ordinary unpadded inputs use its causal path. Packing and cached chunks use an
explicit boolean mask with the correct cache offset. Backend selection and
performance depend on the installed PyTorch/CUDA version.

`gdn_backend="auto"` uses the sequential reference on CPU and requires
[FLA's chunked gated delta rule](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py)
on CUDA. The adapter unpads document runs, passes independent initial states,
sets the query scale explicitly, and uses value-by-key state layout. It performs
decay and beta activation outside the kernel. Choose `gdn_backend="reference"`
explicitly to debug on CUDA. The reference token loop is too slow for the main
pretraining run. FLA is optional and is not installed by collection requirements.

The FLA adapter has not been exercised on an A100 here. Pin a compatible FLA
version after validating forward, backward, packed resets, and decoding against
the reference. Also benchmark grouped expert dispatch and fused/chunked output
loss before committing the Colab budget. Optional `moe_backend="batched"`
provides padded batched expert GEMMs; `mla_backend="flash_varlen"` uses FA2
document-packed attention. The trainer uses checkpointed chunked linear CE
without retaining full vocabulary logits. Reference paths remain available;
optimized paths carry no throughput promise until profiled on the target host.

The [training package](../train/README.md) provides binary data loading, Muon + AdamW,
LR schedules, single-device pretraining, validation and safetensors resume
checkpoints. SFT/GRPO remain separate work. Unselected experts with `grad=None`
skip optimizer momentum and decay in that trainer. Optional non-reentrant
activation checkpointing is enabled through `model.gradient_checkpointing`.

## Local model snapshots

`model.save_pretrained("checkpoints/example")` saves `config.json` and `model.safetensors`.
The loader still accepts legacy model-only `model.pt` exports. Full training
resume uses the trainer's separate optimizer/RNG/data-state checkpoints.
`LaMoEForCausalLM.from_pretrained("checkpoints/example")` loads locally with
`weights_only=True`, validates tied weights, and restores the model and router
buffers. This is not a complete training resume checkpoint: a trainer must also
save optimizer, schedule, data cursor, random state, and token counters. Snapshot
paths can be overwritten by subsequent saves; use distinct names when needed.
