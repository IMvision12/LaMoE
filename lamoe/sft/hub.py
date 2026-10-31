"""Load the fixed SFT repository and tokenize its separate main/identity splits."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from lamoe.hub_datasets import SFT_REPO, checksum, hub_client, identity
from lamoe.preprocessing.common import (
    SPECIAL_TOKENS,
    atomic_json,
    file_hash,
    read_json,
    stage_lock,
)
from .data import iter_records

SPLITS = ("train", "validation", "test")
CONFIGS = ("main", "identity")


def source_files(metadata, config):
    if metadata.get("dataset") != SFT_REPO or metadata.get("format_version") != 1:
        raise ValueError("Unsupported SFT upload manifest")
    selected, seen = {}, set()
    expected = {f"data/{group}/{split}.jsonl" for group in CONFIGS for split in SPLITS}
    raw = set()
    for row in metadata.get("files", []):
        name = row.get("path")
        if name in seen:
            raise ValueError("Duplicate SFT manifest path")
        seen.add(name)
        if name not in expected:
            # Optional pre-tokenized configurations are never used for re-tokenization.
            allowed_tokens = {
                f"data/{group}_tokens/{split}.parquet"
                for group in CONFIGS
                for split in SPLITS
            }
            if name not in allowed_tokens:
                raise ValueError("Unexpected SFT dataset file")
            continue
        raw.add(name)
        if (
            type(row.get("rows")) is not int
            or row["rows"] < 1
            or type(row.get("bytes")) is not int
            or row["bytes"] < 1
        ):
            raise ValueError("Invalid SFT split counts")
        checksum(row.get("sha256"))
        group, split = name.split("/")[1:]
        split = split.removesuffix(".jsonl")
        if row.get("config") != group or row.get("split") != split:
            raise ValueError("SFT configuration/split metadata disagree")
        if group == config:
            selected[split] = {k: row[k] for k in ("path", "sha256", "bytes", "rows")}
    if raw != expected or set(selected) != set(SPLITS):
        raise ValueError("Incomplete main/identity SFT manifest")
    return selected


def prepare(
    output,
    tokenizer_path,
    config="main",
    repo_id=SFT_REPO,
    revision="main",
    maximum_context=2048,
):
    identity(repo_id, SFT_REPO, revision, allow_main=True)
    if config not in CONFIGS:
        raise ValueError("SFT config must be main or identity")
    if type(maximum_context) is not int or not 2 <= maximum_context <= 2048:
        raise ValueError("SFT context must be between 2 and 2048; no truncation")
    from tokenizers import Tokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq

    output, tokenizer_path = Path(output), Path(tokenizer_path)
    if tokenizer_path.resolve().is_relative_to(output.resolve()):
        raise ValueError("Tokenizer must be separate from the prepared SFT output")
    tokenizer_hash = file_hash(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.encode_special_tokens = True
    if any(tokenizer.token_to_id(token) != i for i, token in enumerate(SPECIAL_TOKENS)):
        raise ValueError("Tokenizer must retain the native special-token mapping")
    vocab = tokenizer.get_vocab_size()
    recipe = {
        "repo_id": repo_id,
        "config": config,
        "tokenizer_sha256": tokenizer_hash,
        "maximum_context": maximum_context,
        "vocab_size": vocab,
        "label_policy": "assistant-only-unshifted-v1",
    }
    with stage_lock(output):
        target = output / "manifest.json"
        Api, download = hub_client()
        if target.exists():
            state = read_json(target)
            if state.get("recipe") != recipe or revision not in (
                "main",
                state.get("revision"),
            ):
                raise ValueError(
                    "SFT input/settings changed; use a new output directory"
                )
            identity(repo_id, SFT_REPO, state.get("revision"))
            if set(state.get("sources", {})) != set(SPLITS):
                raise ValueError("Incomplete saved SFT source metadata")
        else:
            if any(
                p.name not in {".writer.lock", "raw_cache"} for p in output.iterdir()
            ):
                raise ValueError("Use an empty SFT output directory")
            pinned = Api().dataset_info(repo_id, revision=revision).sha
            identity(repo_id, SFT_REPO, pinned)
            path = download(
                repo_id=repo_id,
                filename="upload-manifest.json",
                repo_type="dataset",
                revision=pinned,
                cache_dir=output / "raw_cache",
            )
            state = {
                "version": 1,
                "complete": False,
                "recipe": recipe,
                "revision": pinned,
                "source_manifest_sha256": file_hash(path),
                "sources": source_files(read_json(path), config),
                "outputs": {},
            }
            atomic_json(target, state)
        pinned = state["revision"]
        schema = pa.schema(
            [
                ("id", pa.string()),
                ("input_ids", pa.list_(pa.int32())),
                ("labels", pa.list_(pa.int32())),
            ]
        )
        # Temporary disk-backed hashes check held-out boundaries without retaining conversations.
        with sqlite3.connect("") as db:
            db.execute(
                "CREATE TABLE seen (kind TEXT, digest TEXT, split TEXT, PRIMARY KEY(kind,digest,split))"
            )

            def check_row(row, split):
                signatures = [
                    hashlib.sha256(row["id"].encode()).hexdigest(),
                    hashlib.sha256(
                        json.dumps(
                            [row["input_ids"], row["labels"]], separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                ]
                for kind, digest in zip(("id", "tokens"), signatures):
                    other = db.execute(
                        "SELECT split FROM seen WHERE kind=? AND digest=? AND split!=? LIMIT 1",
                        (kind, digest, split),
                    ).fetchone()
                    if other:
                        raise ValueError(
                            "SFT train/validation/test overlap by ID or tokenized content"
                        )
                    db.execute(
                        "INSERT OR IGNORE INTO seen VALUES (?,?,?)",
                        (kind, digest, split),
                    )

            for split in SPLITS:
                source = state["sources"][split]
                if source.get("path") != f"data/{config}/{split}.jsonl":
                    raise ValueError("Unexpected saved SFT source path")
                checksum(source.get("sha256"))
                path = output / "tokens" / f"{split}.parquet"
                temp = path.with_suffix(".parquet.partial")
                saved = state["outputs"].get(split)
                if saved:
                    if (
                        saved.get("path") != f"tokens/{split}.parquet"
                        or saved.get("rows") != source["rows"]
                    ):
                        raise ValueError("Invalid saved SFT output metadata")
                    candidate = path if path.exists() else temp
                    if (
                        not candidate.is_file()
                        or candidate.stat().st_size != saved["bytes"]
                        or file_hash(candidate) != saved["sha256"]
                    ):
                        raise ValueError("Prepared SFT output checksum changed")
                    count = 0
                    for row in iter_records(
                        candidate, maximum_context=maximum_context, vocab_size=vocab
                    ):
                        check_row(row, split)
                        count += 1
                    if count != source["rows"]:
                        raise ValueError("Prepared SFT row count changed")
                    if candidate == temp:
                        temp.replace(path)
                    saved["complete"] = True
                    atomic_json(target, state)
                    continue
                if path.exists():
                    raise ValueError(
                        "Unrecorded SFT token file exists; use a new output directory"
                    )
                raw = Path(
                    download(
                        repo_id=repo_id,
                        filename=source["path"],
                        repo_type="dataset",
                        revision=pinned,
                        cache_dir=output / "raw_cache",
                    )
                )
                if (
                    raw.stat().st_size != source["bytes"]
                    or file_hash(raw) != source["sha256"]
                ):
                    raise ValueError("SFT source checksum mismatch")
                path.parent.mkdir(exist_ok=True)
                count, batch = 0, []
                with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
                    for row in iter_records(raw, tokenizer, maximum_context, vocab):
                        check_row(row, split)
                        batch.append(row)
                        count += 1
                        if len(batch) == 256:
                            writer.write_table(
                                pa.Table.from_pylist(batch, schema=schema)
                            )
                            batch = []
                            db.commit()
                    if batch:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                if (
                    count != source["rows"]
                    or pq.ParquetFile(temp).metadata.num_rows != count
                ):
                    raise ValueError(
                        "SFT split row count disagrees with the Hub manifest"
                    )
                # Record the verified file before rename, so interruption is resumable.
                state["outputs"][split] = {
                    "path": f"tokens/{split}.parquet",
                    "rows": count,
                    "bytes": temp.stat().st_size,
                    "sha256": file_hash(temp),
                    "complete": False,
                }
                atomic_json(target, state)
                temp.replace(path)
                state["outputs"][split]["complete"] = True
                atomic_json(target, state)
                print(f"Prepared {config}/{split}: {count:,} rows", flush=True)
        state["complete"] = True
        atomic_json(target, state)
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", choices=[SFT_REPO], default=SFT_REPO)
    parser.add_argument("--config", choices=CONFIGS, default="main")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context", type=int, default=2048)
    args = parser.parse_args(argv)
    result = prepare(
        args.output,
        args.tokenizer,
        args.config,
        args.repo_id,
        args.revision,
        args.max_context,
    )
    print(
        json.dumps(
            {
                "repo_id": SFT_REPO,
                "config": args.config,
                "revision": result["revision"],
                "complete": result["complete"],
                "outputs": result["outputs"],
            },
            indent=2,
        )
    )
    return 0
