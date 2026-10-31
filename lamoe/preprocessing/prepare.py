#!/usr/bin/env python3
"""Load the uploaded IMvision12 dataset and preprocess its checked local files."""

import argparse
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lamoe.preprocessing.common import read_json
from lamoe.preprocessing.tokenizer_profiles import DEFAULT_PROFILE, PROFILES
from lamoe.preprocessing.snapshot import DEFAULT_REPO
from lamoe.hub_datasets import TOKENIZED_REPO


def positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser(
        "snapshot", help="Pin the LaMoE pretraining dataset or validate local Parquet"
    )
    inputs = snapshot.add_mutually_exclusive_group()
    inputs.add_argument(
        "--repo-id",
        choices=[DEFAULT_REPO],
        help=f"Pretraining dataset (default: {DEFAULT_REPO})",
    )
    inputs.add_argument(
        "--raw-dir", type=Path, help="Optional already downloaded local fixture"
    )
    snapshot.add_argument(
        "--revision",
        default="main",
        help="Full revision hash, or main resolved once and pinned",
    )
    snapshot.add_argument("--output", type=Path, required=True)
    token_snapshot = sub.add_parser(
        "token-snapshot", help="Load the existing tokenized pretraining dataset"
    )
    token_snapshot.add_argument(
        "--repo-id", choices=[TOKENIZED_REPO], default=TOKENIZED_REPO
    )
    token_snapshot.add_argument("--revision", default="main")
    token_snapshot.add_argument("--output", type=Path, required=True)
    token_snapshot.add_argument(
        "--tokenizer",
        type=Path,
        help="Matching local tokenizer if not stored with the tokenized dataset",
    )
    clean = sub.add_parser(
        "clean", help="Quality filters, exact/near deduplication, evaluation overlap"
    )
    clean.add_argument("--snapshot", type=Path, required=True)
    clean.add_argument("--output", type=Path, required=True)
    clean.add_argument("--no-near-dedup", action="store_true")
    clean.add_argument("--evaluation-jsonl", type=Path)
    clean.add_argument("--min-chars", type=positive, default=40)
    clean.add_argument("--max-hours", type=float)
    tokenizer = sub.add_parser(
        "train-tokenizer", help="Train the training-only byte BPE"
    )
    tokenizer.add_argument("--clean-manifest", type=Path, required=True)
    tokenizer.add_argument("--output", type=Path, required=True)
    tokenizer.add_argument("--vocab-size", type=positive, default=32000)
    tokenizer.add_argument("--sample-bytes", type=positive, default=200_000_000)
    tokenizer.add_argument("--seed", type=int, default=122209)
    tokenizer.add_argument(
        "--profile",
        choices=PROFILES,
        default=DEFAULT_PROFILE,
        help=f"Pre-tokenization recipe (default: {DEFAULT_PROFILE}); use bytelevel to resume legacy training",
    )
    compare = sub.add_parser(
        "compare-tokenizers",
        help="Compare both 32K BPE profiles on identical train/development samples",
    )
    compare.add_argument("--clean-manifest", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--vocab-size", type=positive, default=32000)
    compare.add_argument("--sample-bytes", type=positive, default=50_000_000)
    compare.add_argument("--evaluation-bytes", type=positive, default=5_000_000)
    compare.add_argument("--seed", type=int, default=122209)
    compare.add_argument(
        "--freeze",
        type=Path,
        help="Publish the selected tokenizer here if the evidence/correctness gates pass",
    )
    tokenize = sub.add_parser(
        "tokenize", help="Tokenize documents, optional FIM, EOS, exact counts"
    )
    tokenize.add_argument("--clean-manifest", type=Path, required=True)
    tokenize.add_argument("--tokenizer", type=Path, required=True)
    tokenize.add_argument("--output", type=Path, required=True)
    tokenize.add_argument("--fim-rate", type=float, default=0.2)
    tokenize.add_argument("--seed", type=int, default=122209)
    tokenize.add_argument("--max-hours", type=float)
    tokenize.add_argument(
        "--max-shard-tokens",
        type=positive,
        default=64_000_000,
        help="Target tokens per binary shard; documents are kept whole",
    )
    tokenize.add_argument("--max-shard-documents", type=positive, default=100_000)
    plan = sub.add_parser(
        "plan", help="Choose exact source quotas from measured training tokens"
    )
    plan.add_argument("--token-manifest", type=Path, required=True)
    plan.add_argument("--training-tokens", type=positive, required=True)
    plan.add_argument("--output", type=Path, required=True)
    pack = sub.add_parser("pack", help="Build model-compatible fixed-length packs")
    pack.add_argument("--token-manifest", type=Path, required=True)
    pack.add_argument("--plan", type=Path)
    pack.add_argument("--split", choices=("train", "validation"), default="train")
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--sequence-length", type=positive, default=2048)
    pack.add_argument("--sequences-per-shard", type=positive, default=128)
    pack.add_argument("--seed", type=int, default=122209)
    pack.add_argument("--max-hours", type=float)
    report = sub.add_parser("report", help="Print an existing stage manifest")
    report.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    if getattr(args, "max_hours", None) is not None:
        import math

        if not math.isfinite(args.max_hours) or args.max_hours <= 0:
            parser.error("--max-hours must be positive and finite")
    if args.command == "snapshot":
        from lamoe.preprocessing.snapshot import prepare, prepare_hf

        if args.raw_dir:
            if args.revision != "main":
                parser.error("--revision is only used for a Hugging Face dataset")
            result = prepare(args.output, args.raw_dir)
        else:
            result = prepare_hf(
                args.output, args.repo_id or DEFAULT_REPO, args.revision
            )
    elif args.command == "token-snapshot":
        from lamoe.preprocessing.tokenized_hub import prepare

        result = prepare(args.output, args.repo_id, args.revision, args.tokenizer)
    elif args.command == "clean":
        from lamoe.preprocessing.cleaning import clean

        result = clean(
            args.snapshot,
            args.output,
            not args.no_near_dedup,
            args.evaluation_jsonl,
            args.min_chars,
            args.max_hours,
        )
    elif args.command == "train-tokenizer":
        from lamoe.preprocessing.tokenizer import train_tokenizer

        result = train_tokenizer(
            args.clean_manifest,
            args.output,
            args.vocab_size,
            args.sample_bytes,
            args.seed,
            args.profile,
        )
    elif args.command == "compare-tokenizers":
        from lamoe.preprocessing.tokenizer_comparison import compare_tokenizers

        result = compare_tokenizers(
            args.clean_manifest,
            args.output,
            args.vocab_size,
            args.sample_bytes,
            args.evaluation_bytes,
            args.seed,
            args.freeze,
        )
    elif args.command == "tokenize":
        from lamoe.preprocessing.tokenization import tokenize

        result = tokenize(
            args.clean_manifest,
            args.tokenizer,
            args.output,
            args.fim_rate,
            args.seed,
            args.max_hours,
            args.max_shard_tokens,
            args.max_shard_documents,
        )
    elif args.command == "plan":
        from lamoe.preprocessing.packing import make_plan

        result = make_plan(args.token_manifest, args.output, args.training_tokens)
    elif args.command == "pack":
        from lamoe.preprocessing.packing import pack

        result = pack(
            args.token_manifest,
            args.output,
            args.plan,
            args.split,
            args.sequence_length,
            args.sequences_per_shard,
            args.seed,
            args.max_hours,
        )
    else:
        result = read_json(args.manifest)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in {
                    "shards",
                    "cursors",
                    "rng_state",
                    "source_statistics",
                    "completed_inputs",
                }
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
