"""Pinned continuous decay and logical-row-preserving filtered-corpus resume."""

import copy
import hashlib
import numpy as np
from lamoe.preprocessing.common import file_hash
from lamoe.preprocessing.source_policy import continuation_policy
from lamoe.preprocessing.token_store import (
    TokenCursor,
    verify_shard,
    local_path,
    TOKEN_DTYPE,
)

START = 17_036_158_976
END = 25_500_000_000


class FilteredCursor(TokenCursor):
    """Persist original shard/document ordinals; skip removed physical rows."""

    def __init__(self, root, manifest, split, source, state, seed, **kwargs):
        self.physical = {s["path"]: s for s in manifest["shards"]}
        logical = dict(manifest, shards=manifest["continuation"]["original_shards"])
        super().__init__(root, logical, split, source, state, seed, **kwargs)

    def _open(self):
        position = self.state["shard"]
        if not 0 <= position < len(self.shards):
            raise ValueError("Invalid original shard ordinal")
        if self.loaded == position:
            return
        self.close()
        old = self.shards[position]
        new = self.physical.get(old["path"])
        count = old["documents"]
        removed = np.empty(0, dtype=np.int64)
        if old.get("selection"):
            path = local_path(self.root, old["selection"])
            if file_hash(path) != old["selection_sha256"]:
                raise ValueError("Selection checksum changed")
            with np.load(path, allow_pickle=False) as selection:
                removed = selection["rows"].astype(np.int64)
        if len(np.unique(removed)) != len(removed) or (
            len(removed) and (removed.min() < 0 or removed.max() >= count)
        ):
            raise ValueError("Invalid filtered-row map")
        keep = np.ones(count, dtype=bool)
        keep[removed] = False
        mapping = np.full(count, -1, dtype=np.int64)
        mapping[keep] = np.arange(int(keep.sum()))
        key = int.from_bytes(
            hashlib.sha256(f"{self.seed}:documents:{old['path']}".encode()).digest()[
                :8
            ],
            "big",
        )
        self.order = mapping[
            np.random.Generator(np.random.PCG64(key)).permutation(count)
        ]
        if new is None:
            if keep.any():
                raise ValueError("Missing retained shard")
        else:
            if new["documents"] != int(keep.sum()):
                raise ValueError("Original-to-filtered row count mismatch")
            if position not in self.verified:
                verify_shard(self.root, new)
                self.verified.add(position)
            self.index = np.load(
                local_path(self.root, new["index_path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            self.tokens = np.memmap(
                local_path(self.root, new["path"]), dtype=TOKEN_DTYPE, mode="r"
            )
        self.loaded = position

    def _advance(self):
        self.state["offset"] = 0
        self.state["document"] += 1
        if self.state["document"] == len(self.order):
            self.state["document"] = 0
            self.state["shard"] += 1
            if self.state["shard"] == len(self.shards):
                self.state["shard"] = 0
                self.state["passes"] += 1

    def take(self, maximum):
        if maximum <= 0:
            raise ValueError("Positive fragment required")
        while True:
            self._open()
            ordinal = self.state["document"]
            if not 0 <= ordinal < len(self.order):
                raise ValueError("Invalid original document ordinal")
            if self.order[ordinal] >= 0:
                break
            if self.state["offset"]:
                raise ValueError("Cannot discard a partially consumed document")
            self._advance()
        record = self.index[self.order[ordinal]]
        offset, length = self.state["offset"], int(record["length"])
        if not 0 <= offset < length:
            raise ValueError("Invalid original partial-document offset")
        end = min(length, offset + maximum)
        start = int(record["offset"])
        values = self.tokens[start + offset : start + end].copy()
        self.state["offset"] = end
        if end == length:
            self._advance()
        return values


def validate_transition(progress, recipe, contract):
    """Only pinned rollback may enter this new recipe; later resumes are exact."""
    if progress != contract["source_progress"] or recipe != contract["target_recipe"]:
        raise ValueError(
            "Continuation source or target recipe differs from pinned authorization"
        )
    old = progress["recipe"]
    if (
        progress["state"]["step"] != 519904
        or progress["state"]["input_tokens"] != START
    ):
        raise ValueError("Wrong rollback checkpoint")
    allowed = copy.deepcopy(old)
    allowed["planned_tokens"] = END
    allowed["settings"]["training_tokens"] = END
    allowed["schedule"] = recipe["schedule"]
    allowed["data"] = recipe["data"]
    if allowed != recipe:
        raise ValueError("Unapproved optimizer/model/settings change")
    if recipe["schedule"] != dict(
        kind="final-linear-token-decay-v1",
        start_input_tokens=START,
        end_input_tokens=END,
        decay_input_tokens=END - START,
        start_multiplier=0.1,
        rounding="exact_input_token_endpoint",
    ):
        raise ValueError("Wrong continuous decay")
    saved = copy.deepcopy(progress["loader"])
    saved["recipe"] = copy.deepcopy(recipe["data"])
    introduced, _ = continuation_policy(
        recipe["data"].get("continuation"), recipe["data"]["quotas"]
    )
    if not introduced:
        raise ValueError("Private continuation requires explicit source metadata")
    for name in introduced:
        saved["consumed"][name] = 0
        saved["cursors"][name] = dict(shard=0, document=0, offset=0, passes=0)
    return saved
