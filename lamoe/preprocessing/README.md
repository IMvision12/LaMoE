# Dataset loading and preprocessing

Load the already uploaded `IMvision12/LaMoE-Dataset` with the read-only loader:

```sh
python -m lamoe.preprocessing snapshot --repo-id IMvision12/LaMoE-Dataset --output .lamoe/prepared/snapshot
python -m lamoe.preprocessing clean --snapshot .lamoe/prepared/snapshot/manifest.json --output .lamoe/prepared/clean
```

The first call resolves an immutable dataset revision. Reusing the same output
directory keeps that revision. The loader verifies uploaded metadata and checks
each Parquet shard's SHA256 when read. This raw reader accepts only
`IMvision12/LaMoE-Dataset`. It performs no collection or uploading.

To load existing tokenized pretraining shards from `IMvision12/LaMoE-Tokenized`:

```sh
python -m lamoe.preprocessing token-snapshot --output .lamoe/prepared/tokens
```

The tokenized reader verifies the native binary/index format and preserves the
original token manifest. It does not re-tokenize the data. Main and identity SFT
records come only from `IMvision12/lamoe-sft`, through `python -m lamoe.sft prepare`.

Local Parquet fixtures are supported through `--raw-dir`. Tokenization, packing,
screening and resumable token loading remain available. Existing model weights
must always use their matching frozen tokenizer.

See [the complete loading and training guide](../../docs/TRAIN.md) for the data
contract, optional dependencies, CLI commands and validation limits.
