"""Read version-pinned dataset files, without collection or publishing logic."""

import re
from pathlib import Path, PurePosixPath
from .common import VERSION, atomic_json, file_hash, read_json, stage_lock
from lamoe.hub_datasets import PRETRAINING_REPO, identity
import hashlib

DEFAULT_REPO = PRETRAINING_REPO


def hub_identity(repo_id, revision):
    identity(repo_id, DEFAULT_REPO, revision)


def parquet_name(value):
    if not isinstance(value, str):
        raise ValueError("Invalid dataset shard path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or path.as_posix() != value
        or path.suffix != ".parquet"
    ):
        raise ValueError("Only relative Parquet shard paths are supported")
    return value


def hub_client():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Hugging Face loading requires the preprocessing extra"
        ) from exc
    return HfApi, hf_hub_download


def prepare_hf(output, repo_id=DEFAULT_REPO, revision="main"):
    """Pin uploaded metadata; download checked Parquet files lazily when read."""
    hub_identity(repo_id, "0" * 40)
    if revision != "main":
        hub_identity(repo_id, revision)
    output = Path(output)
    with stage_lock(output):
        target = output / "manifest.json"
        if target.exists():
            old = read_json(target)
            if (
                old.get("local_only")
                or old.get("repo_id") != repo_id
                or revision not in ("main", old.get("revision"))
            ):
                raise ValueError(
                    "Output already pins different inputs; use a new directory"
                )
            hub_identity(repo_id, old.get("revision"))
            if not old.get("complete"):
                raise ValueError("Saved snapshot is incomplete")
            return old
        HfApi, download = hub_client()
        pinned = HfApi().dataset_info(repo_id, revision=revision).sha
        hub_identity(repo_id, pinned)
        state_path = download(
            repo_id=repo_id,
            filename="state/progress.json",
            repo_type="dataset",
            revision=pinned,
            cache_dir=output / "raw_cache",
        )
        state = read_json(state_path)
        sources = {row["name"]: row["domain"] for row in state["recipe"]["sources"]}
        if (
            len(sources) != len(state["recipe"]["sources"])
            or not sources
            or set(sources) != set(state["sources"])
            or any(
                not re.fullmatch(r"[a-z0-9_]+", name)
                or domain not in {"general", "code", "cyber", "math"}
                for name, domain in sources.items()
            )
        ):
            raise ValueError("Invalid dataset source/domain map")
        if any(
            row["status"] not in {"budget_reached", "exhausted"}
            for row in state["sources"].values()
        ):
            raise ValueError("Uploaded dataset metadata is incomplete")
        shards, seen = [], set()
        for row in state["shards"]:
            name = parquet_name(row["data"])
            if name in seen or row["source"] not in sources:
                raise ValueError("Duplicate shard path or unknown source")
            seen.add(name)
            if (
                type(row["docs"]) is not int
                or row["docs"] <= 0
                or type(row["raw_bytes"]) is not int
                or row["raw_bytes"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", row["parquet_sha256"])
            ):
                raise ValueError("Invalid shard counts or checksum")
            # Retain only file-loading metadata, never the upstream acquisition recipe.
            shards.append(
                {
                    k: row[k]
                    for k in ("data", "source", "docs", "raw_bytes", "parquet_sha256")
                }
            )
        if not shards:
            raise ValueError("No dataset shards")
        for name, totals in state["sources"].items():
            selected = [row for row in shards if row["source"] == name]
            if (
                sum(row["docs"] for row in selected) != totals["docs"]
                or sum(row["raw_bytes"] for row in selected) != totals["raw_bytes"]
            ):
                raise ValueError("Uploaded dataset totals disagree")
        manifest = dict(
            version=VERSION,
            complete=True,
            repo_id=repo_id,
            revision=pinned,
            raw_dir=None,
            local_only=False,
            state_sha256=file_hash(state_path),
            sources=sources,
            shards=shards,
        )
        atomic_json(target, manifest)
    return manifest


def local_file(root, relative):
    root = Path(root).resolve()
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative or ":" in relative:
        raise ValueError("Input paths must be relative to the supplied local directory")
    result = (root / relative).resolve()
    if result == root or not result.is_relative_to(root) or not result.is_file():
        raise ValueError("Input file is missing or outside the local directory")
    return result


def prepare(output, raw_dir):
    import pyarrow.parquet as pq

    root = Path(raw_dir).resolve()
    source_path = root / "manifest.json"
    spec = read_json(source_path)
    sources = spec.get("sources", {})
    if not sources or any(
        not re.fullmatch(r"[a-z0-9_]+", k)
        or v not in {"general", "code", "cyber", "math"}
        for k, v in sources.items()
    ):
        raise ValueError("Supply logical source names and supported domains")
    shards, seen = [], set()
    for item in spec.get("shards", []):
        relative = item.get("path", item.get("data", ""))
        if not relative or relative in seen or item.get("source") not in sources:
            raise ValueError("Duplicate input path or unknown source")
        seen.add(relative)
        path = local_file(root, relative)
        digest = file_hash(path)
        if item.get("sha256", digest) != digest:
            raise ValueError("Local input checksum mismatch")
        count = size = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if (
                    row.get("source") != item["source"]
                    or row.get("domain") != sources[item["source"]]
                ):
                    raise ValueError("Input source/domain disagrees with the manifest")
                if row.get("split") not in {"train", "validation"} or not isinstance(
                    row.get("text"), str
                ):
                    raise ValueError(
                        "Each row needs text and an explicit train/validation split"
                    )
                n = len(row["text"].encode("utf-8"))
                if row.get("utf8_bytes") != n:
                    raise ValueError("Each row needs a correct utf8_bytes field")
                if (
                    row.get("text_sha256")
                    != hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                ):
                    raise ValueError("Each row needs a correct text_sha256 field")
                count += 1
                size += n
        shards.append(
            {
                "data": relative,
                "source": item["source"],
                "docs": count,
                "raw_bytes": size,
                "parquet_sha256": digest,
            }
        )
    if not shards:
        raise ValueError("No local input shards")
    manifest = {
        "version": VERSION,
        "complete": True,
        "raw_dir": str(root),
        "sources": sources,
        "shards": shards,
        "local_only": True,
        "input_manifest_sha256": file_hash(source_path),
    }
    with stage_lock(output):
        target = Path(output) / "manifest.json"
        if target.exists() and read_json(target) != manifest:
            raise ValueError("Snapshot inputs changed; choose a new output directory")
        atomic_json(target, manifest)
    return manifest


def shard_path(snapshot_path, manifest, shard):
    if manifest.get("local_only") and manifest.get("raw_dir"):
        path = local_file(manifest["raw_dir"], shard["data"])
    else:
        hub_identity(manifest.get("repo_id"), manifest.get("revision"))
        name = parquet_name(shard["data"])
        _, download = hub_client()
        path = Path(
            download(
                repo_id=manifest["repo_id"],
                filename=name,
                repo_type="dataset",
                revision=manifest["revision"],
                cache_dir=Path(snapshot_path).parent / "raw_cache",
            )
        )
    if file_hash(path) != shard["parquet_sha256"]:
        raise ValueError("Input shard checksum changed")
    return path
