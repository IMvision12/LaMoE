"""Atomic update-boundary checkpoints, including both optimizer states/ownership."""

import os
import copy
import random
import tempfile
import shutil
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import load_model, save_model

from lamoe.preprocessing.common import atomic_json, file_hash, read_json


def verify_checkpoint(path):
    path = Path(path)
    marker = read_json(path / "checkpoint.json")
    if marker.get("complete") is not True or marker.get("version") != 1:
        raise ValueError("Incomplete or unsupported checkpoint")
    for name in (
        "model.safetensors",
        "config.json",
        "training_state.pt",
        "progress.json",
    ):
        if file_hash(path / name) != marker["sha256"][name]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")


def latest_checkpoint(root):
    root = Path(root).resolve()
    pointer = root / "latest.json"
    if not pointer.exists():
        if (root / "run.json").exists() or list(root.glob("step-*")):
            raise ValueError(
                "Existing run has no latest pointer; inspect before continuing"
            )
        return None
    candidate = (root / read_json(pointer)["checkpoint"]).resolve()
    if candidate.parent != root or not candidate.name.startswith("step-"):
        raise ValueError("Invalid latest checkpoint path")
    verify_checkpoint(candidate)
    return str(candidate)


def retain_checkpoints(root, keep=2, protected_extra=()):
    """Only remove verified trainer checkpoints, never arbitrary output folders.

    Keep newest two, best validation, and stage-final. Called only after a
    newer checkpoint has been completely published and verified.
    """
    root = Path(root).resolve()
    paths = sorted(p for p in root.glob("step-*") if p.is_dir() and not p.is_symlink())
    protected = set(paths[-keep:]) | {Path(p).resolve() for p in protected_extra}
    for name in ("latest.json", "best.json", "base-final.json"):
        if (root / name).exists():
            protected.add((root / read_json(root / name)["checkpoint"]).resolve())
    for path in paths:
        if path.resolve().parent != root or path in protected:
            continue
        if len(path.name) != 13 or not path.name[5:].isdigit():
            continue
        verify_checkpoint(path)
        expected = {
            "checkpoint.json",
            "model.safetensors",
            "config.json",
            "training_state.pt",
            "progress.json",
        }
        if {p.name for p in path.iterdir()} != expected or any(
            p.is_symlink() for p in path.iterdir()
        ):
            continue
        shutil.rmtree(path)


def save_checkpoint(output, model, optimizer, loader, state, recipe):
    root = Path(output)
    destination = root / f"step-{state['step']:08d}"
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=root))
    # save_model handles tied tensors without storing the embedding twice.
    save_model(model, str(staging / "model.safetensors"))
    model.config.save(staging / "config.json")
    training = {
        "optimizer": optimizer.state_dict(),  # Versioned Muon/AdamW bundle, not one optimizer.
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if next(model.parameters()).is_cuda
        else [],
    }
    np_state = np.random.get_state()
    training["numpy_rng"] = (
        np_state[0],
        np_state[1].tolist(),
        np_state[2],
        np_state[3],
        np_state[4],
    )
    torch.save(training, staging / "training_state.pt")
    atomic_json(
        staging / "progress.json",
        {"state": state, "recipe": recipe, "loader": loader.state_dict()},
    )
    files = ("model.safetensors", "config.json", "training_state.pt", "progress.json")
    for name in files:
        with (staging / name).open("rb+") as stream:
            os.fsync(stream.fileno())
    atomic_json(
        staging / "checkpoint.json",
        {
            "complete": True,
            "version": 1,
            "sha256": {name: file_hash(staging / name) for name in files},
        },
    )
    os.rename(staging, destination)
    atomic_json(root / "latest.json", {"checkpoint": destination.name})
    return destination


def extended_loader_state(progress, recipe):
    """Allow only larger quotas after a completed plan; preserve all cursors/RNG.

    The completed WSD progress remains 1.0, so continuation retains the final
    learning rate. This is an explicit new token plan, not an exact recipe resume.
    """
    old = progress["recipe"]
    state = progress["state"]
    if (
        state["input_tokens"] != old["planned_tokens"]
        or state["lr_progress"] != 1.0
        or recipe["planned_tokens"] <= old["planned_tokens"]
    ):
        raise ValueError("Only a completed token plan can be extended")
    previous = old["data"]["quotas"]
    quotas = recipe["data"]["quotas"]
    saved = progress["loader"]
    if (
        set(quotas) != set(previous)
        or any(type(v) is not int or v < previous[n] for n, v in quotas.items())
        or sum(quotas.values()) != recipe["planned_tokens"]
        or saved["recipe"] != old["data"]
        or saved["consumed"] != previous
    ):
        raise ValueError(
            "Extension requires completed, monotonically increased source quotas"
        )
    allowed = copy.deepcopy(old)
    allowed["planned_tokens"] = recipe["planned_tokens"]
    allowed["settings"]["training_tokens"] = recipe["planned_tokens"]
    allowed["data"]["quotas"] = copy.deepcopy(quotas)
    if allowed != recipe:
        raise ValueError(
            "Plan extension cannot change model, optimizer, data identity or other settings"
        )
    restored = copy.deepcopy(saved)
    restored["recipe"] = copy.deepcopy(recipe["data"])
    return restored


def load_checkpoint(
    path,
    model,
    optimizer,
    loader,
    recipe,
    *,
    extend_completed_plan=False,
    allow_final_decay=False,
):
    path = Path(path)
    marker = read_json(path / "checkpoint.json")
    if marker.get("complete") is not True or marker.get("version") != 1:
        raise ValueError("Incomplete or unsupported checkpoint")
    for name in (
        "model.safetensors",
        "config.json",
        "training_state.pt",
        "progress.json",
    ):
        if file_hash(path / name) != marker["sha256"][name]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    progress = read_json(path / "progress.json")
    loader_state = progress["loader"]
    if progress["recipe"] != recipe:
        if allow_final_decay:
            if extend_completed_plan:
                raise ValueError("Cannot combine final decay with plan extension")
            from .final_decay import validate_recipe_transition

            loader_state = validate_recipe_transition(progress, recipe)
        elif extend_completed_plan:
            loader_state = extended_loader_state(progress, recipe)
        else:
            raise ValueError("Training recipe changed; use original settings to resume")
    load_model(model, str(path / "model.safetensors"), strict=True, device="cpu")
    training = torch.load(
        path / "training_state.pt", map_location="cpu", weights_only=True
    )
    optimizer.load_state_dict(training["optimizer"])
    loader.load_state_dict(loader_state)
    random.setstate(training["python_rng"])
    if "numpy_rng" in training:
        n = training["numpy_rng"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(training["torch_rng"])
    if next(model.parameters()).is_cuda:
        if len(training["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed across resume")
        torch.cuda.set_rng_state_all(training["cuda_rng"])
    if model.balance_counts.any():
        raise ValueError("Checkpoint is not at an optimizer-update boundary")
    return progress["state"]
