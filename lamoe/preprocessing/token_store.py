"""LaMoE binary-v1: uint16 payloads and compact, memory-mappable document indexes.

This is a LaMoE format, not the Megatron .idx specification. Source and split
are stored once per shard in the manifest; document records have no Python objects.
"""

import hashlib
import os
from pathlib import Path

import numpy as np

from .common import file_hash

STORAGE_FORMAT = "lamoe-bin-v1"
TOKEN_DTYPE = np.dtype("<u2")
INDEX_DTYPE = np.dtype(
    [
        ("offset", "<u8"),
        ("length", "<u8"),
        ("raw_bytes", "<u8"),
        ("document_id", "V32"),
        ("fim", "u1"),
    ]
)


def require_binary(manifest):
    if manifest.get("storage_format") != STORAGE_FORMAT:
        raise ValueError(
            "Expected lamoe-bin-v1 token data. Legacy SQLite data is not modified; retokenize into a new directory."
        )


def local_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Token shard path escapes output directory")
    return path


def verify_shard(root, shard):
    """Check payload/index hashes, dimensions, and contiguous document boundaries."""
    data = local_path(root, shard["path"])
    index = local_path(root, shard["index_path"])
    if data.stat().st_size != shard["tokens"] * TOKEN_DTYPE.itemsize:
        raise ValueError("Token payload size disagrees with manifest")
    if file_hash(data) != shard["sha256"] or file_hash(index) != shard["index_sha256"]:
        raise ValueError("Token shard/index checksum mismatch")
    records = np.load(index, allow_pickle=False)
    if (
        records.dtype != INDEX_DTYPE
        or records.shape != (shard["documents"],)
        or not len(records)
    ):
        raise ValueError("Invalid document index schema or count")
    expected = 0
    raw_bytes = fim = 0
    for row in records:
        n = int(row["length"])
        if int(row["offset"]) != expected or n <= 0 or int(row["fim"]) not in (0, 1):
            raise ValueError("Invalid document index boundaries")
        expected += n
        raw_bytes += int(row["raw_bytes"])
        fim += int(row["fim"])
    if (
        expected != shard["tokens"]
        or raw_bytes != shard["raw_bytes"]
        or fim != shard["fim_documents"]
    ):
        raise ValueError("Index totals disagree with manifest")


class ShardWriter:
    """One bounded output pair; caller publishes the manifest after input completion."""

    def __init__(self, output, relative, source, split, domain, input_path):
        self.output = Path(output)
        self.relative = relative
        self.target = local_path(output, relative)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.temp = self.target.with_suffix(".bin.partial")
        self.stream = self.temp.open("wb")
        self.records = []
        self.tokens = self.raw_bytes = self.fim_documents = 0
        self.metadata = {
            "source": source,
            "split": split,
            "domain": domain,
            "input_path": input_path,
        }

    def append(self, ids, identity, raw_bytes, fim):
        values = np.asarray(ids, dtype=TOKEN_DTYPE)
        self.stream.write(values.tobytes())
        self.records.append(
            (self.tokens, len(values), raw_bytes, bytes.fromhex(identity), int(fim))
        )
        self.tokens += len(values)
        self.raw_bytes += raw_bytes
        self.fim_documents += int(fim)

    def finish(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        index_relative = self.relative.removesuffix(".bin") + ".idx.npy"
        index = local_path(self.output, index_relative)
        index_temp = index.with_suffix(".npy.partial")
        with index_temp.open("wb") as stream:
            np.save(
                stream, np.asarray(self.records, dtype=INDEX_DTYPE), allow_pickle=False
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(self.temp, self.target)
        os.replace(index_temp, index)
        return dict(
            **self.metadata,
            path=self.relative,
            index_path=index_relative,
            sha256=file_hash(self.target),
            index_sha256=file_hash(index),
            documents=len(self.records),
            tokens=self.tokens,
            raw_bytes=self.raw_bytes,
            fim_documents=self.fim_documents,
            token_bytes=self.target.stat().st_size,
            index_bytes=index.stat().st_size,
        )

    def close(self):
        self.stream.close()


class TokenCursor:
    """Deterministic two-level shuffle with a resumable document/fragment cursor.

    At most one token/index pair is mapped per source. Returned fragments are
    copied so callers cannot retain a view after the underlying mapping closes.
    """

    def __init__(
        self, root, manifest, split, source, state, seed, excluded=None, empty_shards=()
    ):
        require_binary(manifest)
        self.root, self.state, self.seed = Path(root), state, seed
        self.excluded = excluded or set()
        self.shards = [
            s
            for s in manifest["shards"]
            if s["split"] == split
            and s["source"] == source
            and s["path"] not in empty_shards
        ]
        self.shards.sort(
            key=lambda s: hashlib.sha256(f"{seed}:{s['path']}".encode()).digest()
        )
        if not self.shards:
            raise ValueError(f"Empty token source: {split}/{source}")
        self.index = self.tokens = self.order = None
        self.loaded = None
        self.verified = set()

    def _open(self):
        position = self.state["shard"]
        if not 0 <= position < len(self.shards):
            raise ValueError("Invalid token shard cursor")
        if self.loaded == position:
            return
        self.close()
        shard = self.shards[position]
        if position not in self.verified:
            verify_shard(self.root, shard)
            self.verified.add(position)
        self.index = np.load(
            local_path(self.root, shard["index_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.tokens = np.memmap(
            local_path(self.root, shard["path"]), dtype=TOKEN_DTYPE, mode="r"
        )
        key = int.from_bytes(
            hashlib.sha256(f"{self.seed}:documents:{shard['path']}".encode()).digest()[
                :8
            ],
            "big",
        )
        self.order = np.random.Generator(np.random.PCG64(key)).permutation(
            len(self.index)
        )
        if self.excluded:
            self.order = np.array(
                [
                    i
                    for i in self.order
                    if bytes(self.index[i]["document_id"]).hex() not in self.excluded
                ]
            )
            if not len(self.order):
                raise ValueError("Screening report omitted an empty shard")
        self.loaded = position

    def take(self, maximum):
        if maximum <= 0:
            raise ValueError("Token fragment length must be positive")
        self._open()
        state = self.state
        if not 0 <= state["document"] < len(self.order):
            raise ValueError("Invalid document cursor")
        record = self.index[self.order[state["document"]]]
        offset, n = state["offset"], int(record["length"])
        if not 0 <= offset < n:
            raise ValueError("Invalid document fragment cursor")
        end = min(n, offset + maximum)
        start = int(record["offset"])
        values = self.tokens[start + offset : start + end].copy()
        state["offset"] = end
        if end == n:
            state["offset"] = 0
            state["document"] += 1
            if state["document"] == len(self.order):
                state["document"] = 0
                state["shard"] += 1
                if state["shard"] == len(self.shards):
                    state["shard"] = 0
                    state["passes"] += 1
        return values

    def close(self):
        for name in ("index", "tokens"):
            value = getattr(self, name)
            if value is not None:
                value._mmap.close()
                setattr(self, name, None)
        self.order = None
        self.loaded = None
