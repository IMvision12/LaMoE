# Distribution scope

The LaMoE source repository is public and distributed under Apache-2.0. Model
artifacts and training datasets are maintained separately. Availability below
was checked on October 31, 2026.

## Included

- Full model implementation and full/tiny configurations.
- Raw and tokenized pretraining readers, separate main/identity SFT preparation,
  local cleaning, deduplication, tokenization, packing and screening.
- Single-device pretraining, Muon/AdamW, token schedules, validation and resume.
- Native chat rendering, cached inference, portable SFT and evaluation interfaces.
- Optional Transformers configuration/model/cache adapter and checkpoint exporter.
- Aggregate benchmark results, all 57 MMLU subjects and 27 BBH task variants.
- Selected benchmark graphs, architecture diagrams, offline explorer and guides.
- Apache-2.0 source license.

## Separate Hub artifacts

The Transformers-compatible model export is publicly available at
[`IMvision12/lamoe-247`](https://huggingface.co/IMvision12/lamoe-247).
It includes weights, tokenizer/configuration files and the chat template, while
Python implementation code stays in this public source repository.
It contains the final identity-corrected `step-00129185` weights and supporting
inference files, documentation and results. It does not contain predecessor
weights or optimizer/RNG/loader state for training resume.

All three training dataset repositories are private and need authorized access:

- `IMvision12/LaMoE-Dataset`: uploaded raw pretraining archive.
- `IMvision12/lamoe-sft`: uploaded main and identity configurations.
- `IMvision12/LaMoE-Tokenized`: configured tokenized-data destination; currently
  metadata only, without a complete token snapshot.

## Excluded

- Corpus collectors, scraping, acquisition registries and private data builders.
- Dataset upload helpers and private historical curricula.
- Weights, tokenizer payloads, training examples and token shards.
- Optimizer/RNG states, raw responses, grading tests, caches, logs and credentials.
- Historical queue controllers, development tools and regression test suites.

The portable SFT baseline and proportional token planner support new experiments.
Historical curriculum transitions require their original continuation settings.
The software checks are recorded in [VALIDATION.md](VALIDATION.md).
Benchmark protocols and supplementary tables are described in
[EVALUATION.md](EVALUATION.md).
