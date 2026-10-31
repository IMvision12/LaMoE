# Validation

Validation covers synthetic CPU checks, inference parity for the released
checkpoint, and one WSL CUDA smoke test. The test suite is not included in this
source distribution.

## Completed checks

- **Model and runtime:** parameter counts, cached/full decoding equivalence,
  document-boundary resets, weight serialization, and teacher-forced scoring.
- **Data preparation:** fixed dataset IDs, revision pinning, file checksums,
  path validation, token/index integrity, assistant-only labels, split isolation,
  and interrupted preparation. Hub-loader checks used mocked calls and local
  fixtures, rather than complete dataset downloads.
- **Training mechanics:** synthetic gradient checks, token-loader resume, and
  interrupted versus uninterrupted SFT resume with matching final weights.
- **Transformers integration:** local AutoClass registration, chat formatting,
  cache behavior, generation, save/load, and unchanged-weight export. For the
  released checkpoint, all 487 stored tensors matched the original checkpoint;
  logits for one prompt and three greedy output tokens also matched on CPU.
- **CUDA inference:** the released export generated 64 greedy tokens for one
  binary-search prompt on an RTX 4060 laptop GPU under WSL. Parameters, inputs,
  outputs and caches were on CUDA, and logits remained finite. The first three
  tokens matched the CPU check.
- **Packaging and documentation:** wheel and source-archive builds, package
  contents, Python source syntax, CLI help, and local documentation/image links.

## Tested configurations

| Configuration | Software and backend |
| --- | --- |
| Windows CPU | Python 3.13.14; PyTorch 2.11.0+cu128 using CPU tensors; Transformers 5.17.0; reference GDN |
| WSL CUDA | PyTorch 2.13.0+cu129; Transformers 5.17.0; FP32 weights; reference GDN and SDPA |

## Limits

Synthetic checks and one CUDA prompt do not establish correctness for every
input, generation mode, dependency version or hardware configuration. FLA through
the Transformers wrapper, FlashAttention, quantization and distributed device
maps remain unverified. The standalone CPU scoring checks do not constitute a
full integration test of every evaluation task.

The benchmark tables contain the completed measurements described in
[EVALUATION.md](EVALUATION.md). Inference parity and software validation are
separate from benchmark scoring; this record adds no benchmark results. Training
from the released source and Hub archives has not been validated end to end.
