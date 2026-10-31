# Transformers support

The optional adapter supports `AutoConfig`, `AutoTokenizer` and
`AutoModelForCausalLM`. The tested version is Transformers **5.17.0** with PyTorch
**2.11.0** on CPU. A separate WSL smoke test used PyTorch **2.13.0+cu129** on
an RTX 4060 laptop GPU with FP32 weights, reference GDN and SDPA. Native inference
remains available independently.

## Install in your own environment

```sh
python -m pip install -e ".[transformers]"
```

This installs an optional dependency set; importing the ordinary `lamoe` package
does not require Transformers. No installation is performed by importing the adapter.

## Load a Hub or exported model

The Transformers-compatible export is publicly available at
[`IMvision12/lamoe-247`](https://huggingface.co/IMvision12/lamoe-247). It
contains model weights, tokenizer and configuration files; Python
implementation code is distributed through this public source repository.
Install the LaMoE source and register its local implementation before loading:

```python
import torch
from lamoe.transformers import register
from transformers import AutoModelForCausalLM, AutoTokenizer

register()
path = "IMvision12/lamoe-247"  # Or a local downloaded model-files folder.
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=False, dtype=torch.float32,
).eval().to(device)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain binary search."}],
    tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
).to(device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

`trust_remote_code=True` alone cannot supply an architecture whose Python code is
absent from the Hub. The installed source registration is required. Pin the Hub
`revision` and source version for a fixed code/weight pair. No tools or generated
code execute through the model or template.

To use the implementation installed from this source repository instead:

```python
from lamoe.transformers import register
from transformers import AutoModelForCausalLM

register()
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/downloaded-model-files", trust_remote_code=False, local_files_only=True,
)
```

Registration is explicit; it does not download or load a model. The source
repository contains no weights or tokenizer payload. Supply them separately.

## Export a native local checkpoint

```sh
python -m lamoe.transformers.export --checkpoint /path/to/native-checkpoint --tokenizer /path/to/tokenizer.json --output /path/to/new-transformers-export
```

The output must be empty and separate from original inputs. The exporter copies
weights and the frozen tokenizer byte-for-byte, preserves original parameter
names, adds Transformers metadata/chat formatting and writes a checksum manifest.
The default output has no Python files or remote-code AutoClass mappings. It does
not initialize a model, train, upload files or run Git. Original files are unchanged.
Add `--expected-sha256` to require a specific source weight fingerprint.
The exporter includes the source license. Review the model card and measured
results when distributing a model export.

An optional `--include-code` export can produce a standalone remote-code bundle
for separate local testing. The public Hub layout uses the default model-files
export and keeps the implementation in this GitHub source repository.

## Verified behavior and boundaries

- Cached/uncached greedy generation, beam search, left-padded batches,
  chunked cache continuation, save/load, tied embeddings and labeled loss.
- Custom caches preserve MLA latents, GDN recurrent states and convolution history.
  Padded batches fall back to full-sequence decoding without caching.
- Native chat token IDs and formatting match the frozen renderer for tested
  histories. Tool arguments use dictionaries. Tool definitions require an
  explicit initial system message; there is no automatic identity/system prompt.
- Literal native control-token strings in message/tool content are rejected by
  the string template. Use the native token-level renderer for that special case.
- Context remains 2,048 for the released model. Prompts are never truncated;
  generation is capped to the remaining context.
- FLA through this wrapper, quantization, distributed device maps and other Transformers versions
  remain untested in this adapter. Foreign/static/offloaded caches and assisted
  decoding are rejected. Intermediate hidden states, attention maps, arbitrary
  position IDs, vocabulary resizing and `inputs_embeds` are not exposed.

The actual identity-corrected checkpoint was checked locally on CPU: all 487
stored tensors matched exactly, logits matched the original runtime exactly for
one prompt, and three greedy output tokens agreed. This is an inference parity
check, not a new benchmark campaign or proof of every possible generation mode.

The WSL CUDA smoke test generated 64 greedy tokens for "Explain binary search."
using the local exported files and registered source. Parameters, inputs, logits,
outputs and caches were on CUDA, and logits remained finite. The first three
tokens matched the earlier CPU check. This checks one FP32 reference-GDN/SDPA
generation path; it does not establish FLA support or a performance benchmark.

## Installed documentation

Wheel installs include guides, architecture assets, selected graphs, aggregate
results and configurations under `share/lamoe/` in the installation prefix.
The source archive includes the original `docs/`, `configs/`, `results/` and
requirements paths. Source checkouts use the paths shown throughout this guide.
