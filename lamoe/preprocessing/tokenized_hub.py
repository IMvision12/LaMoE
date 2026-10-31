"""Read the fixed tokenized repository into a verified native binary snapshot."""

from collections import defaultdict
import os
from pathlib import Path
import shutil

from lamoe.hub_datasets import (
    TOKENIZED_REPO,
    checksum,
    hub_client,
    identity,
    relative_name,
)
from .common import SPECIAL_TOKENS, atomic_json, file_hash, read_json, stage_lock
from .token_store import INDEX_DTYPE, require_binary, verify_shard


def artifacts(manifest):
    require_binary(manifest)
    if (
        manifest.get("complete") is not True
        or manifest.get("token_dtype") != "<u2"
        or manifest.get("index_record_bytes") != INDEX_DTYPE.itemsize
    ):
        raise ValueError("Incomplete or unsupported tokenized dataset")
    sources = manifest.get("sources", {})
    if not sources or any(
        domain not in {"general", "code", "cyber", "math"}
        for domain in sources.values()
    ):
        raise ValueError("Invalid tokenized source/domain map")
    tokenizer = manifest.get("tokenizer", {})
    checksum(tokenizer.get("sha256"))
    if (
        type(tokenizer.get("vocab_size")) is not int
        or not len(SPECIAL_TOKENS) < tokenizer["vocab_size"] <= 65536
    ):
        raise ValueError("Invalid tokenized vocabulary")
    if tokenizer.get("special_tokens") != {
        name: i for i, name in enumerate(SPECIAL_TOKENS)
    }:
        raise ValueError("Tokenized native special-token mapping differs")
    files, totals = (
        {},
        defaultdict(
            lambda: {"tokens": 0, "documents": 0, "raw_bytes": 0, "fim_documents": 0}
        ),
    )

    def add(name, digest, suffixes, size=None):
        relative_name(name, suffixes)
        checksum(digest)
        if name in files:
            if files[name] != {"sha256": digest, "bytes": size}:
                raise ValueError("Conflicting tokenized file metadata")
            return
        if size is not None and (type(size) is not int or size <= 0):
            raise ValueError("Invalid tokenized file size")
        files[name] = {"sha256": digest, "bytes": size}

    if not manifest.get("shards"):
        raise ValueError("Tokenized dataset has no shards")
    physical_paths = set()
    for row in manifest["shards"]:
        for key in ("path", "index_path"):
            if row.get(key) in physical_paths:
                raise ValueError("Duplicate tokenized shard/index path")
            physical_paths.add(row.get(key))
        if (
            row.get("split") not in {"train", "validation"}
            or row.get("source") not in sources
            or row.get("domain") != sources[row["source"]]
        ):
            raise ValueError("Tokenized shard source/split/domain mismatch")
        for key in ("tokens", "documents", "raw_bytes", "fim_documents"):
            if type(row.get(key)) is not int or row[key] < (
                1 if key in {"tokens", "documents"} else 0
            ):
                raise ValueError("Invalid tokenized shard counts")
            totals[(row["split"], row["source"])][key] += row[key]
        add(row["path"], row["sha256"], {".bin"}, row["tokens"] * 2)
        add(row["index_path"], row["index_sha256"], {".npy"}, row.get("index_bytes"))
    for split, sources_in_split in manifest.get("counts", {}).items():
        for source, counts in sources_in_split.items():
            if (split, source) not in totals or any(
                counts.get(key) != total
                for key, total in totals[(split, source)].items()
            ):
                raise ValueError("Tokenized dataset totals disagree")
    if set(totals) != {
        (split, source)
        for split, counts in manifest.get("counts", {}).items()
        for source in counts
    }:
        raise ValueError("Incomplete tokenized split totals")
    for row in manifest["shards"] + manifest.get("continuation", {}).get(
        "original_shards", []
    ):
        if row.get("selection"):
            add(row["selection"], row.get("selection_sha256"), {".npz"})
    return files


def local_target(root, relative):
    root = root.resolve()
    target = root / relative
    if not target.resolve().is_relative_to(root) or target.resolve() == root:
        raise ValueError("Tokenized output escapes its directory")
    return target


def copy_checked(source, target, digest, size=None):
    if file_hash(source) != digest or (
        size is not None and source.stat().st_size != size
    ):
        raise ValueError("Downloaded tokenized artifact checksum/size mismatch")
    if target.exists():
        if file_hash(target) != digest:
            raise ValueError("Local tokenized artifact changed; it was not overwritten")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".download.partial")
    if temp.exists():
        if file_hash(temp) != digest:
            raise ValueError(
                "Unverified partial download exists; inspect it before retrying"
            )
    else:
        try:
            os.link(source, temp)
        except OSError:
            shutil.copyfile(source, temp)
    if file_hash(temp) != digest:
        raise ValueError("Tokenized local copy checksum mismatch")
    os.replace(temp, target)


def prepare(output, repo_id=TOKENIZED_REPO, revision="main", tokenizer_path=None):
    """Download only declared token/index/selection files and the matching tokenizer.

    The original manifest bytes are published only after all payloads verify.
    Original continuation policies are retained, never reconstructed or bypassed.
    """
    identity(repo_id, TOKENIZED_REPO, revision, allow_main=True)
    output = Path(output)
    with stage_lock(output):
        progress_path = output / "hub-download.json"
        source_path = output / "source-manifest.json"
        Api, download = hub_client()
        if progress_path.exists():
            state = read_json(progress_path)
            identity(state.get("repo_id"), TOKENIZED_REPO, state.get("revision"))
            if (
                revision not in ("main", state["revision"])
                or file_hash(source_path) != state["manifest_sha256"]
            ):
                raise ValueError(
                    "Tokenized snapshot inputs changed; use a new directory"
                )
        else:
            if any(
                p.name not in {".writer.lock", "raw_cache"} for p in output.iterdir()
            ):
                raise ValueError("Use an empty tokenized snapshot directory")
            info = Api().dataset_info(repo_id, revision=revision)
            identity(repo_id, TOKENIZED_REPO, info.sha)
            path = Path(
                download(
                    repo_id=repo_id,
                    filename="manifest.json",
                    repo_type="dataset",
                    revision=info.sha,
                    cache_dir=output / "raw_cache",
                )
            )
            artifacts(read_json(path))
            checksum_value = file_hash(path)
            copy_checked(path, source_path, checksum_value)
            known_names = {item.rfilename for item in getattr(info, "siblings", [])}
            token_name = next(
                (
                    name
                    for name in ("tokenizer/tokenizer.json", "tokenizer.json")
                    if name in known_names
                ),
                None,
            )
            state = {
                "repo_id": repo_id,
                "revision": info.sha,
                "manifest_sha256": checksum_value,
                "tokenizer_file": token_name,
                "complete": False,
            }
            atomic_json(progress_path, state)
        manifest = read_json(source_path)
        selected = artifacts(manifest)
        token_digest = manifest["tokenizer"]["sha256"]
        destination = local_target(output, "tokenizer/tokenizer.json")
        if tokenizer_path:
            token_source = Path(tokenizer_path)
        elif destination.is_file():
            token_source = destination
        elif state["tokenizer_file"]:
            if state["tokenizer_file"] not in {
                "tokenizer/tokenizer.json",
                "tokenizer.json",
            }:
                raise ValueError("Unexpected tokenized tokenizer file")
            token_source = Path(
                download(
                    repo_id=repo_id,
                    filename=state["tokenizer_file"],
                    repo_type="dataset",
                    revision=state["revision"],
                    cache_dir=output / "raw_cache",
                )
            )
        else:
            raise ValueError(
                "The tokenized repository has no tokenizer file; supply its matching --tokenizer"
            )
        copy_checked(token_source, destination, token_digest)
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(destination))
        if tokenizer.get_vocab_size() != manifest["tokenizer"]["vocab_size"] or any(
            tokenizer.token_to_id(name) != i for i, name in enumerate(SPECIAL_TOKENS)
        ):
            raise ValueError("Frozen tokenizer metadata does not match its file")
        atomic_json(output / "tokenizer/manifest.json", manifest["tokenizer"])
        for number, (name, spec) in enumerate(selected.items(), 1):
            target = local_target(output, name)
            source = (
                target
                if target.is_file()
                else Path(
                    download(
                        repo_id=repo_id,
                        filename=name,
                        repo_type="dataset",
                        revision=state["revision"],
                        cache_dir=output / "raw_cache",
                    )
                )
            )
            copy_checked(source, target, spec["sha256"], spec["bytes"])
            if number % 32 == 0 or number == len(selected):
                print(f"Verified tokenized files: {number}/{len(selected)}", flush=True)
        for shard in manifest["shards"]:
            verify_shard(output, shard)
        # Keep the token-manifest identity used by existing plans/checkpoints.
        copy_checked(source_path, output / "manifest.json", state["manifest_sha256"])
        state["complete"] = True
        atomic_json(progress_path, state)
        return dict(
            repo_id=repo_id,
            revision=state["revision"],
            complete=True,
            files=len(selected),
            manifest_sha256=state["manifest_sha256"],
            train_tokens=sum(
                row["tokens"] for row in manifest["shards"] if row["split"] == "train"
            ),
        )
