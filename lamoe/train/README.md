# LaMoE pretraining

Single-device training reads local `lamoe-bin-v1` token shards produced after
loading raw text from `IMvision12/LaMoE-Dataset` or existing tokenized shards from
`IMvision12/LaMoE-Tokenized`. SFT/identity records use `IMvision12/lamoe-sft` through
the separate SFT entry point. Data collection and uploading
are not part of this repository.

The trainer includes document-aware loading, Muon/AdamW optimization, token-based
schedules, evaluation, diagnostics and exact update-boundary checkpoint resume.
Real GPU training requires explicit hardware/kernel validation and a frozen
screening report. Importing the modules does not start a training run.

See [the pretraining guide](../../docs/TRAIN.md), [portable SFT](../../docs/SFT.md),
and [benchmark evaluation](../../docs/EVALUATION.md). Hardware acceptance must
be verified for the actual machine and software build used for training.
