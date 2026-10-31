# Public release: uses a local, source-agnostic token planner.
"""Validate immutable input recipes, prepare mixture, and record host software."""

import importlib.metadata
import platform
from lamoe.preprocessing.common import (
    complete_manifest,
    file_hash,
    read_json,
    atomic_json,
)
from lamoe.preprocessing.screen import load_screening
from lamoe.preprocessing.planning import plan_tokens


def prepare_plan(config, root):
    if config.tiny:
        return
    if not config.screening_report:
        raise ValueError(
            "Full pretraining requires --screening-report; build it with python -m lamoe.preprocessing.screen"
        )
    screen = load_screening(config.screening_report, config.token_manifest)
    manifest = complete_manifest(config.token_manifest)
    if not config.plan:
        if not config.training_tokens:
            raise ValueError(
                "Supply --training-tokens after throughput calibration, or a frozen --plan"
            )
        counts = {
            n: v["tokens"] - screen["removed_tokens"].get("train/" + n, 0)
            for n, v in manifest["counts"]["train"].items()
        }
        plan = plan_tokens(counts, config.training_tokens)
        plan.update(
            token_manifest_sha256=file_hash(config.token_manifest),
            screening_sha256=file_hash(config.screening_report),
        )
        target = root / "mixture.json"
        if target.exists() and read_json(target) != plan:
            raise ValueError("Existing mixture differs; do not change a running recipe")
        atomic_json(target, plan)
        config.plan = str(target)


def environment():
    versions = {}
    for name in (
        "torch",
        "triton",
        "flash-attn",
        "flash-linear-attention",
        "fla-core",
        "safetensors",
        "numpy",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages=versions,
    )


def check_kernel_report(config, model_config):
    import torch

    if not config.kernel_report:
        raise ValueError(
            "Full training requires --kernel-report from python -m lamoe.train.calibrate --full on this GPU"
        )
    report = read_json(config.kernel_report)
    if not report.get("passed") or not report.get("full_model_profile"):
        raise ValueError("Kernel report is not a passed full-model profile")
    if config.device != "cuda" or report["gpu"] != torch.cuda.get_device_name():
        raise ValueError("Kernel report GPU differs from current host")
    if report["environment"] != environment():
        raise ValueError(
            "Software environment changed since kernel acceptance; rerun calibration"
        )
    expected = model_config.to_dict()
    expected["hybrid_layers"] = list(expected["hybrid_layers"])
    if expected["gdn_backend"] == "auto":
        expected["gdn_backend"] = "fla"
    if (
        report["model"] != expected
        or report["sequence_length"] != config.sequence_length
        or report["batch_size"] != config.batch_size
        or report.get("gradient_checkpointing", False) != config.gradient_checkpointing
    ):
        raise ValueError(
            "Profile model, physical batch or sequence length differs from training"
        )
    return file_hash(config.kernel_report)
