"""CPU memory-mapped, online packing with checkpointable source/document cursors."""

import copy
import random
from pathlib import Path

import numpy as np
import torch

from lamoe.preprocessing.common import complete_manifest, file_hash, read_json
from lamoe.preprocessing.token_store import TokenCursor, require_binary
from lamoe.preprocessing.source_policy import continuation_policy, pass_limits


def nested_tuple(value):
    return tuple(nested_tuple(x) for x in value) if isinstance(value, list) else value


class TokenBatchLoader:
    """Finite iterator of packed CPU batches; no worker prefetch to lose on resume.

    Uses the corpus distribution once by default, or exact input-token quotas
    from a preprocessing plan. A fragment starts a new segment, so neither
    attention nor next-token loss crosses documents. Context-window cuts lose
    one next-token target; token quotas count INPUT tokens, not CE targets.
    """

    def __init__(
        self,
        manifest_path,
        *,
        split="train",
        sequence_length=2048,
        batch_size=1,
        seed=122209,
        plan_path=None,
        domain=None,
        screening_report=None,
    ):
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        if sequence_length < 2 or batch_size < 1:
            raise ValueError("sequence_length >= 2 and batch_size >= 1 required")
        self.path = Path(manifest_path)
        self.manifest = complete_manifest(self.path)
        require_binary(self.manifest)
        counts = self.manifest["counts"].get(split, {})
        counts = copy.deepcopy(counts)
        self.screen = None
        if screening_report:
            from lamoe.preprocessing.screen import load_screening

            self.screen = load_screening(screening_report, self.path)
            for n, v in counts.items():
                v["tokens"] -= self.screen["removed_tokens"].get(split + "/" + n, 0)
                if v["tokens"] < 0:
                    raise ValueError("Invalid screened token counts")
        self.excluded = (
            set(self.screen["excluded_document_ids"]) if self.screen else set()
        )
        quotas = {n: v["tokens"] for n, v in counts.items() if v["tokens"]}
        if domain is not None:
            quotas = {
                n: v
                for n, v in quotas.items()
                if self.manifest["sources"].get(n) == domain
            }
        manifest_hash = file_hash(self.path)
        if plan_path:
            if split != "train":
                raise ValueError("A training plan cannot be used for validation")
            plan = read_json(plan_path)
            if plan["token_manifest_sha256"] != manifest_hash:
                raise ValueError("Plan belongs to a different token manifest")
            if plan.get("screening_sha256") != (
                file_hash(screening_report) if screening_report else None
            ):
                raise ValueError("Plan screening identity changed")
            quotas = {
                n: v["sampled_tokens"]
                for n, v in plan["source_plan"].items()
                if v["sampled_tokens"]
            }
        if not quotas or any(type(v) is not int or v <= 0 for v in quotas.values()):
            raise ValueError("No positive integer token quotas")
        limits = pass_limits(self.manifest, split)
        self.introduced_sources, self.introduced_start = continuation_policy(
            self.manifest.get("continuation") if split == "train" else None,
            self.manifest["sources"],
        )
        for name, quota in quotas.items():
            available = counts.get(name, {}).get("tokens", 0)
            cap = limits.get(name, 1)
            if quota > cap * available:
                raise ValueError(f"Quota exceeds allowed passes for {name}")
        if (
            self.introduced_sources
            and sum(
                quota
                for name, quota in quotas.items()
                if name not in self.introduced_sources
            )
            < self.introduced_start
        ):
            raise ValueError(
                "Existing-source quotas cannot reach the continuation threshold"
            )
        self.names = sorted(quotas)
        self.sequence_length, self.batch_size = sequence_length, batch_size
        self.vocab_size = self.manifest["tokenizer"]["vocab_size"]
        self.pad_id = self.manifest["tokenizer"]["special_tokens"]["<|pad|>"]
        if not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("Invalid pad token")
        self.recipe = {
            "version": 1,
            "manifest_sha256": manifest_hash,
            "split": split,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "seed": seed,
            "quotas": quotas,
            "order": "shard-sha256-document-pcg64-v1",
            "screening_sha256": file_hash(screening_report)
            if screening_report
            else None,
        }
        if self.manifest.get("continuation") and split == "train":
            self.recipe["continuation"] = {
                k: v
                for k, v in self.manifest["continuation"].items()
                if k != "original_shards"
            }
        self.state = {
            "consumed": {n: 0 for n in self.names},
            "cursors": {
                n: {"shard": 0, "document": 0, "offset": 0, "passes": 0}
                for n in self.names
            },
        }
        self.rng = random.Random(seed)
        self.cursors = {}
        self._open_cursors()

    def _open_cursors(self):
        for name in self.names:
            cursor_type = TokenCursor
            if self.introduced_sources and name not in self.introduced_sources:
                from .continuation import FilteredCursor

                cursor_type = FilteredCursor
            self.cursors[name] = cursor_type(
                self.path.parent,
                self.manifest,
                self.recipe["split"],
                name,
                self.state["cursors"][name],
                self.recipe["seed"],
                excluded=self.excluded,
                empty_shards=self.screen["empty_shards"] if self.screen else (),
            )

    @property
    def exhausted(self):
        return all(
            self.state["consumed"][n] == self.recipe["quotas"][n] for n in self.names
        )

    def state_dict(self):
        return copy.deepcopy(
            dict(recipe=self.recipe, **self.state, rng_state=self.rng.getstate())
        )

    def load_state_dict(self, state):
        if state["recipe"] != self.recipe:
            raise ValueError(
                "Loader recipe changed; resume requires identical data/settings"
            )
        if set(state["consumed"]) != set(self.names) or set(state["cursors"]) != set(
            self.names
        ):
            raise ValueError("Invalid saved source set")
        for name in self.names:
            used = state["consumed"][name]
            if type(used) is not int or not 0 <= used <= self.recipe["quotas"][name]:
                raise ValueError("Invalid saved consumption")
        self.close()
        self.state = copy.deepcopy({k: state[k] for k in ("consumed", "cursors")})
        self.rng.setstate(nested_tuple(state["rng_state"]))
        self._open_cursors()

    def __iter__(self):
        return self

    def __next__(self):
        rows, segments, sources = [], [], []
        for _ in range(self.batch_size):
            ids = np.full(self.sequence_length, self.pad_id, dtype=np.int64)
            seg = np.full(self.sequence_length, -1, dtype=np.int64)
            src = np.full(self.sequence_length, -1, dtype=np.int64)
            offset, segment = 0, 0
            while offset < self.sequence_length:
                remaining = [
                    self.recipe["quotas"][n] - self.state["consumed"][n]
                    for n in self.names
                ]
                phase_left = self.sequence_length
                if "continuation" in self.recipe:
                    total = sum(self.state["consumed"].values())
                    phase_left = max(0, self.introduced_start - total)
                    if phase_left:
                        remaining = [
                            0 if n in self.introduced_sources else v
                            for n, v in zip(self.names, remaining)
                        ]
                    else:
                        phase_left = self.sequence_length
                if not sum(remaining):
                    break
                point = self.rng.randrange(sum(remaining))
                for source_id, (name, weight) in enumerate(zip(self.names, remaining)):
                    if point < weight:
                        break
                    point -= weight
                values = self.cursors[name].take(
                    min(self.sequence_length - offset, weight, phase_left)
                )
                if int(values.max()) >= self.vocab_size:
                    raise ValueError("Token ID exceeds frozen vocabulary")
                end = offset + len(values)
                ids[offset:end], seg[offset:end], src[offset:end] = (
                    values,
                    segment,
                    source_id,
                )
                self.state["consumed"][name] += len(values)
                offset, segment = end, segment + 1
            if offset == 0:
                break
            rows.append(ids)
            segments.append(seg)
            sources.append(src)
        if not rows:
            raise StopIteration
        ids = torch.from_numpy(np.stack(rows))
        seg = torch.from_numpy(np.stack(segments))
        valid = seg >= 0
        return {
            "input_ids": ids,
            "segment_ids": seg,
            "attention_mask": valid,
            "labels": ids.clone().masked_fill(~valid, -100),
            "source_ids": torch.from_numpy(np.stack(sources)),
        }

    def close(self):
        for cursor in self.cursors.values():
            cursor.close()
        self.cursors.clear()


def target_count(batch):
    valid, seg, labels = (batch[k] for k in ("attention_mask", "segment_ids", "labels"))
    return int(
        (
            valid[:, :-1]
            & valid[:, 1:]
            & (seg[:, :-1] == seg[:, 1:])
            & (labels[:, 1:] != -100)
        ).sum()
    )


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items() if k != "source_ids"}
