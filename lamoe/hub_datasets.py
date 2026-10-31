"""The three supported training-data repositories and safe pinned file loading."""

import re
from pathlib import PurePosixPath

PRETRAINING_REPO = "IMvision12/LaMoE-Dataset"
TOKENIZED_REPO = "IMvision12/LaMoE-Tokenized"
SFT_REPO = "IMvision12/lamoe-sft"
TRAINING_REPOS = frozenset({PRETRAINING_REPO, TOKENIZED_REPO, SFT_REPO})


def identity(repo_id, expected, revision, *, allow_main=False):
    if expected not in TRAINING_REPOS or repo_id != expected:
        raise ValueError(f"This data loader only accepts {expected}")
    if allow_main and revision == "main":
        return
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A full immutable dataset revision is required")


def relative_name(value, suffixes):
    if not isinstance(value, str):
        raise ValueError("Invalid dataset file path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or path.as_posix() != value
        or path.suffix not in suffixes
    ):
        raise ValueError("Unsafe or unsupported dataset file path")
    return value


def checksum(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid dataset file SHA256")
    return value


def hub_client():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ImportError("Hub data loading requires the preprocessing extra") from exc
    return HfApi, hf_hub_download
