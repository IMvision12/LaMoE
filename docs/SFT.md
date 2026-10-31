# SFT loading, tokenization and local fine-tuning

The dataset-upload helper is maintained privately outside this repository.

## Load and tokenize the uploaded dataset

The SFT reader accepts only
[`IMvision12/lamoe-sft`](https://huggingface.co/datasets/IMvision12/lamoe-sft).
Its `main` and `identity` configurations stay separate; each retains its original
train, validation and test splits. Identity replay may reuse main-training records,
so the two configurations are never silently merged.

Install the `preprocessing` extra in your own environment and use your existing
Hugging Face access for the private dataset. The reader does not log in or change
account/repository settings.

```sh
python -m lamoe.sft prepare --repo-id IMvision12/lamoe-sft --config main --tokenizer /path/to/tokenizer.json --output /path/to/main-sft
python -m lamoe.sft prepare --repo-id IMvision12/lamoe-sft --config identity --tokenizer /path/to/tokenizer.json --output /path/to/identity-sft
```

Preparation pins an immutable revision, checks the uploaded manifest and every
JSONL file's size and SHA256, then streams tokenization using the supplied frozen
tokenizer. Native role/tool rendering and assistant-only, unshifted labels are
shared with the local SFT reader. It writes `tokens/train.parquet`,
`tokens/validation.parquet` and `tokens/test.parquet`, plus a resumable manifest
binding the dataset revision, configuration, tokenizer hash and context limit.
The default context is 2,048; overlong records fail without truncation or dropping.

Duplicate IDs and cross-split ID/tokenized-content overlap are rejected. Completed
splits are checked on resume. Interrupted publication resumes from the recorded
partial-file hash. Changing configuration, tokenizer, context or pinned revision
requires a new output directory. The raw JSONL cache is local; no remote Python
scripts, data builders, tool execution or training run during preparation.

Pretraining uses `IMvision12/LaMoE-Dataset` for raw text or
`IMvision12/LaMoE-Tokenized` for existing tokenized shards; see [TRAIN.md](TRAIN.md).

## Portable training entry point

`python -m lamoe.sft` is a portable AdamW baseline. It does not reconstruct the
historical SFT/identity curricula from the public source alone. The `prepare`
subcommand loads/tokenizes data; the `train` subcommand uses local prepared files.
The original form with training flags directly after `lamoe.sft` remains supported.

## Inputs

Supply distinct local training and validation files, including the Parquet outputs
of `prepare`. UTF-8
JSONL is read one physical line at a time; Parquet is also supported. A record
can contain messages:

```json
{"id":"example-1","messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
```

Or pre-tokenized `input_ids` and same-length, **unshifted** `labels`, where
unsupervised positions are `-100`. Every other label must equal its corresponding
input token. A row needs at least one target after position zero. Prompts and role
headers are masked; assistant content and end markers are supervised. Message
rows must end in an assistant turn. Native tool-call rendering is supported as
text; no tool executes.

Rows outside the checkpoint context are rejected without truncation. The
loader checks vocabulary IDs, duplicate IDs and train/validation overlap by ID
and tokenized content. These checks do not prove semantic deduplication or
complete evaluation-set decontamination. The frozen tokenizer must match the
source checkpoint; vocabulary size alone does not establish its identity.

## Run

```sh
python -m lamoe.sft train --checkpoint /path/to/source-checkpoint --tokenizer /path/to/tokenizer.json --train-file /path/to/main-sft/tokens/train.parquet --validation-file /path/to/main-sft/tokens/validation.parquet --output /path/to/new-sft-run --device cpu --epochs 3 --batch-size 1 --accumulation 4 --learning-rate 0.000005
```

CPU reference recurrence is for development. Select CUDA and `--gdn-backend fla`
only in a separately validated compatible environment. Choose your own training
budget and hyperparameters; this command is not the original SFT recipe.

The baseline uses constant learning rate AdamW, zero weight decay, betas
`(0.9, 0.95)` and global gradient clipping. Unequal microbatches weight CE by
supervised next-token targets and router terms by valid input tokens. Router
counts accumulate once per microbatch and reset at each update boundary.
Validation reports target-weighted cross entropy, not a capability score.

## Checkpoints and resume

An immutable `recipe.json` binds the input files, tokenizer, source weights/config
and settings. Each completed epoch writes `epoch-0001/`, containing model/config,
optimizer/RNG state and progress. Checkpoints are published by directory rename;
existing checkpoints are never overwritten and no retention deletion runs.
Each published checkpoint has a complete marker with file hashes, verified
before resume.

```sh
python -m lamoe.sft train --checkpoint /path/to/source-checkpoint --tokenizer /path/to/tokenizer.json --train-file /path/to/main-sft/tokens/train.parquet --validation-file /path/to/main-sft/tokens/validation.parquet --output /path/to/new-sft-run --resume /path/to/new-sft-run/epoch-0001 --device cpu --epochs 3 --batch-size 1 --accumulation 4 --learning-rate 0.000005
```

Use exactly the original recipe, including the total epoch count. Resume is at
completed epoch boundaries only; interruption mid-epoch requires replay from the
last completed epoch. Partial directories are preserved for inspection. Trainer
state is loaded with `weights_only=True`; only use checkpoints from sources you
trust. Historical controllers have different checkpoint contracts.
