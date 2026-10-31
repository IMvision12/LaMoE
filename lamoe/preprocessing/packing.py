# Public release: uses a local, source-agnostic token planner.
"""Exact source quotas and resumable fixed-length packs with document boundaries."""

import os
import random
import time
from pathlib import Path

import numpy as np

from lamoe.preprocessing.planning import plan_tokens

from .common import (
    VERSION,
    atomic_json,
    complete_manifest,
    file_hash,
    read_json,
    stage_lock,
)
from .token_store import TokenCursor, require_binary
from .source_policy import pass_limits


def make_plan(token_manifest, output, training_tokens):
    manifest = complete_manifest(token_manifest)
    require_binary(manifest)
    counts = {s: v["tokens"] for s, v in manifest["counts"].get("train", {}).items()}
    plan = plan_tokens(counts, training_tokens)
    plan["token_manifest_sha256"] = file_hash(token_manifest)
    plan["tokenizer_sha256"] = manifest["tokenizer"]["sha256"]
    atomic_json(output, plan)
    return plan


def _tuple(value):
    return tuple(_tuple(x) for x in value) if isinstance(value, list) else value


def pack(
    token_manifest,
    output,
    plan_path=None,
    split="train",
    sequence_length=2048,
    sequences_per_shard=128,
    seed=122209,
    max_hours=None,
):
    token_manifest, output = Path(token_manifest), Path(output)
    tokens = complete_manifest(token_manifest)
    require_binary(tokens)
    if split not in ("train", "validation"):
        raise ValueError("Invalid packing split")
    if split == "train":
        if plan_path is None:
            raise ValueError("Training packing requires an explicit --plan")
        plan = read_json(plan_path)
        if plan["token_manifest_sha256"] != file_hash(token_manifest):
            raise ValueError("Plan is for another token manifest")
        quotas = {
            s: v["sampled_tokens"]
            for s, v in plan["source_plan"].items()
            if v["sampled_tokens"]
        }
    else:
        if plan_path:
            raise ValueError("Validation packs do not accept a training plan")
        quotas = {
            s: v["tokens"]
            for s, v in tokens["counts"].get("validation", {}).items()
            if v["tokens"]
        }
    if not quotas or any(type(v) is not int or v <= 0 for v in quotas.values()):
        raise ValueError("No positive integer token quotas")
    limits = pass_limits(tokens, split)
    for source, count in quotas.items():
        available = tokens["counts"].get(split, {}).get(source, {}).get("tokens", 0)
        cap = limits.get(source, 1)
        if count > cap * available:
            raise ValueError(f"Quota exceeds permitted passes for {source}")
    if sequence_length < 2 or sequence_length > 2**31 - 1 or sequences_per_shard < 1:
        raise ValueError("Invalid packing dimensions")
    recipe = {
        "version": VERSION,
        "token_manifest_sha256": file_hash(token_manifest),
        "quotas": quotas,
        "split": split,
        "sequence_length": sequence_length,
        "sequences_per_shard": sequences_per_shard,
        "seed": seed,
        "reader_order": "shard-sha256-document-pcg64-v1",
    }
    deadline = time.monotonic() + max_hours * 3600 if max_hours else float("inf")
    names = sorted(quotas)
    with stage_lock(output):
        state_path = output / "manifest.json"
        if state_path.exists():
            state = read_json(state_path)
            if state["recipe"] != recipe:
                raise ValueError("Packed stage inputs/settings changed")
            for shard in state["shards"]:
                if file_hash(output / shard["path"]) != shard["sha256"]:
                    raise ValueError("Packed shard checksum mismatch")
        else:
            state = {
                "complete": False,
                "recipe": recipe,
                "source_ids": {n: i for i, n in enumerate(names)},
                "tokenizer_sha256": tokens["tokenizer"]["sha256"],
                "vocab_size": tokens["tokenizer"]["vocab_size"],
                "special_tokens": tokens["tokenizer"]["special_tokens"],
                "consumed": {n: 0 for n in names},
                "cursors": {
                    n: {"shard": 0, "document": 0, "offset": 0, "passes": 0}
                    for n in names
                },
                "shards": [],
                "target_tokens": 0,
                "input_tokens": 0,
                "rng_state": random.Random(seed).getstate(),
            }
        if state["complete"]:
            return state
        rng = random.Random()
        rng.setstate(_tuple(state["rng_state"]))
        cursors = {}
        try:
            for n in names:
                cursors[n] = TokenCursor(
                    token_manifest.parent, tokens, split, n, state["cursors"][n], seed
                )
            while any(state["consumed"][n] < quotas[n] for n in names):
                if time.monotonic() >= deadline:
                    atomic_json(state_path, state)
                    return state
                ids_rows, seg_rows, source_rows = [], [], []
                for _ in range(sequences_per_shard):
                    ids = np.full(
                        sequence_length, state["special_tokens"]["<|pad|>"], dtype="<u2"
                    )
                    segments = np.full(sequence_length, -1, dtype="<i4")
                    source_ids = np.full(sequence_length, -1, dtype="<i2")
                    offset, segment = 0, 0
                    while offset < sequence_length:
                        remaining = [quotas[n] - state["consumed"][n] for n in names]
                        total = sum(remaining)
                        if total == 0:
                            break
                        point = rng.randrange(total)
                        for source, weight in zip(names, remaining):
                            if point < weight:
                                break
                            point -= weight
                        take = min(
                            sequence_length - offset,
                            quotas[source] - state["consumed"][source],
                        )
                        values = cursors[source].take(take)
                        end = offset + len(values)
                        ids[offset:end] = values
                        segments[offset:end] = segment
                        source_ids[offset:end] = state["source_ids"][source]
                        state["consumed"][source] += len(values)
                        state["input_tokens"] += len(values)
                        state["target_tokens"] += max(0, len(values) - 1)
                        offset, segment = end, segment + 1
                    if offset == 0:
                        break
                    ids_rows.append(ids)
                    seg_rows.append(segments)
                    source_rows.append(source_ids)
                name = f"part-{len(state['shards']):06d}.npz"
                temp = output / (name + ".partial")
                with temp.open("wb") as stream:
                    np.savez(
                        stream,
                        input_ids=np.stack(ids_rows),
                        segment_ids=np.stack(seg_rows),
                        source_ids=np.stack(source_rows),
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, output / name)
                state["shards"].append(
                    {
                        "path": name,
                        "sha256": file_hash(output / name),
                        "sequences": len(ids_rows),
                    }
                )
                state["rng_state"] = rng.getstate()
                state["complete"] = all(
                    state["consumed"][n] == quotas[n] for n in names
                )
                atomic_json(state_path, state)
                print(
                    f"Packed {state['input_tokens']:,}/{sum(quotas.values()):,} input tokens",
                    flush=True,
                )
            return state
        finally:
            for cursor in cursors.values():
                cursor.close()
