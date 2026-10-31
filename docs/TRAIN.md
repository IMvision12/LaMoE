# Preprocessing and pretraining

The public trainer is single-device and reads local `lamoe-bin-v1` token shards.
It includes document-aware packing, RMS-matched Muon plus AdamW (or an explicit
AdamW baseline), token-based scheduling, diagnostics, validation and exact
update-boundary checkpoints. It does not provision a GPU or collect a corpus.

## Load the uploaded Hugging Face dataset

The public data-loading entry point reads
[`IMvision12/LaMoE-Dataset`](https://huggingface.co/datasets/IMvision12/LaMoE-Dataset).
It loads existing uploaded files only. No upstream dataset discovery, scraping,
collection, dataset construction or uploading code is included.

```sh
python -m lamoe.preprocessing snapshot --repo-id IMvision12/LaMoE-Dataset --output /path/to/snapshot
python -m lamoe.preprocessing clean --snapshot /path/to/snapshot/manifest.json --output /path/to/clean
```

Install the declared `preprocessing` extra in your own environment before using
this entry point. The dataset must be readable with your existing Hugging Face
access. The loader does not log in, create repositories or change permissions.
The raw reader accepts only `IMvision12/LaMoE-Dataset`. The tokenized reader
accepts only `IMvision12/LaMoE-Tokenized`, and SFT preparation accepts only
`IMvision12/lamoe-sft`. Benchmark evaluation keeps its official references.

On the first call, `main` resolves to a full immutable revision. Later calls to
the same output directory reuse that revision, even if the upload changes.
Use `--revision` with a full 40-character revision hash to select a specific
version, and a new output directory to change versions.

The existing upload format is `state/progress.json` plus the Parquet shard paths
listed there. The reader verifies completion and counts, keeps only the metadata
needed for loading, and downloads each referenced Parquet file when preprocessing
reads it. Every shard must match its saved SHA256. It does not download or execute
remote Python scripts, follow upstream source links, or fetch an entire dataset
blindly. The generated snapshot/cache belongs outside the public source folder.

## Load existing tokenized pretraining data

Use [`IMvision12/LaMoE-Tokenized`](https://huggingface.co/datasets/IMvision12/LaMoE-Tokenized)
when you want the already tokenized corpus:

```sh
python -m lamoe.preprocessing token-snapshot --repo-id IMvision12/LaMoE-Tokenized --output /path/to/token-snapshot
```

This path expects a complete `lamoe-bin-v1` `manifest.json`, its uint16 binary
shards, non-pickle document indexes and any referenced continuation selection
arrays. It downloads only manifest-declared payloads and their matching frozen
tokenizer. If the tokenizer is not stored with the dataset, pass
`--tokenizer /path/to/matching/tokenizer.json`; its SHA256 must match the manifest.

The original token manifest is published locally only after all files, index
boundaries and totals pass verification. Its bytes and SHA256 are preserved, so
existing plans remain bound to the same manifest. Original continuation metadata
is retained; private historical training contracts are not reconstructed or
bypassed. Point the trainer at `/path/to/token-snapshot/manifest.json`.

Reusing an output directory resumes the same pinned version and verifies retained
files. Downloads use a local cache, with hard links where supported to avoid a
second payload copy. A changed existing payload is rejected rather than replaced.
No tokenization or training runs through `token-snapshot`.

Main SFT and identity data use the separate [SFT loading guide](SFT.md).

## Local data contract

For offline use and synthetic development fixtures, the same pipeline also
accepts already downloaded Parquet files and a local `manifest.json`:

```json
{
  "sources": {"demo": "general"},
  "shards": [{"path": "demo.parquet", "source": "demo"}]
}
```

Each Parquet row must contain `text`, `source`, `domain`, `split`, `utf8_bytes`
and `text_sha256` (SHA256 of UTF-8 text). Splits are `train` or `validation`.
Domains are `general`, `code`, `cyber` or `math`. Optional `split_group` keeps
related training material out of held-out groups. Input file paths are relative
to the supplied root; traversal, missing files and checksum changes are rejected.
The local input path accepts no remote URLs or collection registry.

```sh
python -m lamoe.preprocessing snapshot --raw-dir /path/to/local-parquet --output /path/to/snapshot
python -m lamoe.preprocessing clean --snapshot /path/to/snapshot/manifest.json --output /path/to/clean
python -m lamoe.preprocessing train-tokenizer --clean-manifest /path/to/clean/manifest.json --output /path/to/tokenizer --vocab-size 32000
python -m lamoe.preprocessing tokenize --clean-manifest /path/to/clean/manifest.json --tokenizer /path/to/tokenizer --output /path/to/tokens
```

The tokenizer sampler retains the original 35/25/30/10 code/cyber/general/math
shares and expects training text from all four domains. The public token planner
uses proportional measured source counts, with an exact single-pass budget.
It is a source-agnostic replacement for the private collector's planner and does
not recreate historical source quotas. For an explicit repetition distribution,
provide a separately reviewed plan satisfying the loader's pass limits.
Pass limits default to one for every source. An optional `source_max_passes`
mapping in the token manifest declares positive integer training limits per
logical source; validation always stays single-pass. The manifest hash binds
these settings to plans and exact resumes. There are no source-name-specific
repetition rules in the public code.

Do not retrain a tokenizer for existing weights. Existing checkpoints require
their original frozen tokenizer and special-token mapping.

Cleaning keeps text bytes, applies conservative quality/credential-pattern checks,
validation-first exact/approximate deduplication, optional evaluation overlap
screening and group exclusion. These are limited filters, not an exhaustive
privacy, licensing or contamination audit. Tokenization supports code FIM and
writes uint16 tokens plus non-pickle indexes with hashes and resumable manifests.

## Screening and plans

Full pretraining requires a complete evaluation-screening report bound to the
same token manifest. Supply a frozen local JSONL evaluation file with the fields
supported by `screen.py`; include prompts, answers and reference/test material
appropriate to your intended evaluation.

```sh
python -m lamoe.preprocessing.screen --clean-manifest /path/to/clean/manifest.json --token-manifest /path/to/tokens/manifest.json --evaluation-jsonl /path/to/frozen-evaluation.jsonl --output /path/to/screen
```

The screen uses exact whitespace-normalized matches and case-folded 13-word
matches. Original shards are retained; the loader skips excluded documents.
This is lexical screening of the supplied set, not proof of absent overlap.

`--training-tokens` on the trainer builds a plan from post-screen counts. The
standalone preprocessing `plan` command uses unscreened counts, so do not pass
that plan to screened training without binding the reviewed post-screen quotas
and `screening_sha256`. Packing is optional: the training loader packs online.

## Model and hardware acceptance

`configs/lamoe_247m.json` contains full dimensions; `configs/tiny.json` is a
CPU development configuration. Full CUDA training requires a passed kernel
report for the actual GPU, environment, model, batch, sequence length and
checkpointing choice. `python -m lamoe.train.calibrate --help` describes the
synthetic acceptance/profile command. Running it performs work on the selected
GPU; it is not part of the CPU release checks.

Choose compatible FLA/FlashAttention/PyTorch builds yourself. Pin the versions
validated on your host rather than assuming an untested combination works.

## Pretraining entry point

```sh
python -m lamoe.train --token-manifest /path/to/tokens/manifest.json --output /path/to/new-run --model-config configs/lamoe_247m.json --training-tokens 100000000 --screening-report /path/to/screen/report.json --kernel-report /path/to/kernel-report.json --device cuda --precision bf16 --gdn-backend fla --mla-backend sdpa --moe-backend reference --sequence-length 2048 --batch-size 1 --accumulation-steps 8
```

The token horizon above is an example, not a validated training recipe or a
reproduction of the 25.5B run. Choose it only after measuring resources. The
trainer retains a historical, locally calculated budget ledger with configurable
`--cu-per-hour`, `--pretraining-cu` and runtime limits. It does not read account
balances or purchase/provision resources. These estimates are operational limits,
not a claim of actual cloud cost. Default sessions are six hours with a saving
reserve; use `--help` to review all settings before starting.

Default optimization is explicit RMS-Muon for eligible matrices and AdamW for
embeddings, norms, routers and other parameters. The standard schedule uses
cumulative valid input tokens: 2% warmup, 78% stable, 20% cosine cooldown to 10%
of peak. Input tokens and supervised targets are different quantities. Loss
combines target-weighted CE with input-weighted router losses. FP32 master
weights are retained during BF16 autocast; chunked CE avoids full logits.

## Resume and retention

Use `--resume auto` with the same output folder and unchanged recipe to restore
weights, both optimizers, RNG state, exact data cursors, token counts and schedule.
Checksums are verified. `--max-steps` is a safety pause. Do not reset manifests,
budget ledgers or source ordering to get past an integrity error.

The historical pretraining trainer has checkpoint retention: after a verified
save it retains protected/latest/best/final checkpoints and removes eligible
older verified checkpoints. Review `checkpoint.py` before a real run and keep
independent copies of checkpoints you need. Release preparation never invokes
retention or modifies any original checkpoint.

The optional historical final-decay/filtered-corpus transition depends on the
original private continuation contract. That contract is not distributed; those
transition options fail closed. Fresh public runs and exact same-recipe resume
do not reconstruct or bypass it. Such a contract must also explicitly declare
`introduced_sources` and `introduced_sources_start_tokens`; private source names
and thresholds are not inferred from hardcoded dataset identifiers.
The portable SFT baseline is documented in [SFT.md](SFT.md).
