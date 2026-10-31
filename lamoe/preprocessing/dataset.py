"""Read completed packed shards as tensors accepted by LaMoEForCausalLM."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .common import complete_manifest, file_hash


class PackedDataset(IterableDataset):
    def __init__(self, directory, shuffle=True, seed=122209, verify_hashes=True):
        self.directory = Path(directory)
        self.manifest = complete_manifest(self.directory / "manifest.json")
        self.shuffle, self.seed, self.verify_hashes = shuffle, seed, verify_hashes
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        shards = list(self.manifest["shards"])
        if self.shuffle:
            rng.shuffle(shards)
        worker = get_worker_info()
        if worker:
            shards = shards[worker.id :: worker.num_workers]
        for shard in shards:
            path = self.directory / shard["path"]
            if self.verify_hashes and file_hash(path) != shard["sha256"]:
                raise ValueError("Packed shard checksum mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                input_ids = arrays["input_ids"]
                segment_ids = arrays["segment_ids"]
                order = list(range(shard["sequences"]))
                if self.shuffle:
                    rng.shuffle(order)
                for i in order:
                    ids = torch.from_numpy(input_ids[i].astype(np.int64))
                    segments = torch.from_numpy(segment_ids[i].astype(np.int64))
                    valid = segments >= 0
                    labels = ids.clone().masked_fill(~valid, -100)
                    yield {
                        "input_ids": ids,
                        "segment_ids": segments,
                        "attention_mask": valid,
                        "labels": labels,
                    }
