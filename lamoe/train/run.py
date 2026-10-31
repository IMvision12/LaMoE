"""CLI: python lamoe/train/run.py ... or python -m lamoe.train ..."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lamoe.train.trainer import TrainConfig, train


def main():
    parser = argparse.ArgumentParser(
        description="LaMoE single-device binary-corpus pretraining"
    )
    parser.add_argument("--token-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-steps",
        default=1_000_000_000,
        type=int,
        help="Optimizer updates, not microbatches",
    )
    for name in ("model-config", "plan", "resume", "screening-report", "kernel-report"):
        parser.add_argument("--" + name)
    for name in ("tiny", "gradient-checkpointing", "extend-completed-plan"):
        parser.add_argument("--" + name, action="store_true")
    defaults = TrainConfig("", "", 1)
    for name in (
        "sequence_length",
        "batch_size",
        "accumulation_steps",
        "warmup_steps",
        "seed",
        "eval_every",
        "eval_batches",
        "save_every",
        "log_every",
        "muon_ns_steps",
        "loss_chunk_size",
        "eval_tokens_per_domain",
        "eval_every_tokens",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(defaults, name)
        )
    for name in (
        "learning_rate",
        "muon_learning_rate",
        "muon_weight_decay",
        "muon_momentum",
        "min_lr_ratio",
        "weight_decay",
        "beta2",
        "max_grad_norm",
        "max_runtime_hours",
        "cu_per_hour",
        "pretraining_cu",
        "save_margin_seconds",
        "save_every_minutes",
        "min_free_gb",
        "eval_every_hours",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=getattr(defaults, name)
        )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--optimizer",
        choices=("muon_adamw", "adamw"),
        default="muon_adamw",
        help="Hybrid is the default; adamw is an explicit baseline",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--gdn-backend", choices=("auto", "reference", "fla"))
    parser.add_argument("--mla-backend", choices=("sdpa", "flash_varlen"))
    parser.add_argument("--moe-backend", choices=("reference", "batched"))
    parser.add_argument("--training-tokens", type=int)
    parser.add_argument("--final-decay-tokens", type=int, default=0)
    parser.add_argument("--final-decay-start-tokens", type=int)
    parser.add_argument("--training-deadline-unix", type=float)
    args = parser.parse_args()
    print(json.dumps(train(TrainConfig(**vars(args)))), flush=True)


if __name__ == "__main__":
    main()
