"""Validation-first cleaning with atomic per-shard commits and disk dedup indices."""

import json
import os
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .common import (
    VERSION,
    atomic_json,
    complete_manifest,
    database,
    digest,
    file_hash,
    parquet_rows,
    stage_lock,
)
from .filters import DuplicateIndex, evaluation_ngrams, normalized_hash, quality_reason
from .snapshot import shard_path


def clean(
    snapshot, output, near=True, evaluation_jsonl=None, min_chars=40, max_hours=None
):
    snapshot, output = Path(snapshot), Path(output)
    raw = complete_manifest(snapshot)
    recipe = {
        "version": VERSION,
        "snapshot_sha256": file_hash(snapshot),
        "near": near,
        "min_chars": min_chars,
        "evaluation_sha256": file_hash(evaluation_jsonl) if evaluation_jsonl else None,
    }
    deadline = time.monotonic() + max_hours * 3600 if max_hours else float("inf")
    with stage_lock(output):
        db = database(output / "index.sqlite", recipe)
        try:
            index = DuplicateIndex(db, near)
            db.execute(
                "CREATE TABLE IF NOT EXISTS completed (key TEXT PRIMARY KEY, result TEXT NOT NULL)"
            )
            db.commit()
            if not db.execute(
                "SELECT 1 FROM metadata WHERE key='evaluation_ready'"
            ).fetchone():
                with db:
                    if evaluation_jsonl:
                        with open(evaluation_jsonl, encoding="utf-8") as stream:
                            for line in stream:
                                row = json.loads(line)
                                text = row["text"]
                                # Exact normalized matches cover short evaluation examples too.
                                db.execute(
                                    "INSERT OR IGNORE INTO seen VALUES (?, 'evaluation')",
                                    (normalized_hash(text),),
                                )
                                db.executemany(
                                    "INSERT OR IGNORE INTO evaluation VALUES (?)",
                                    ((h,) for h in evaluation_ngrams(text)),
                                )
                    db.execute(
                        "INSERT INTO metadata VALUES ('evaluation_ready','true')"
                    )
            for split in ("validation", "train"):
                for shard in raw["shards"]:
                    key = digest([split, shard["data"]])
                    if db.execute(
                        "SELECT 1 FROM completed WHERE key=?", (key,)
                    ).fetchone():
                        continue
                    if time.monotonic() >= deadline:
                        return _report(db, output, recipe, raw, False)
                    path = shard_path(snapshot, raw, shard)
                    relative = f"{split}/{shard['source']}/{key}.parquet"
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(".partial")
                    writer, rows, buffered_bytes = None, [], 0
                    rejects, docs, byte_count = Counter(), 0, 0
                    schema = pq.ParquetFile(path).schema_arrow
                    db.execute("BEGIN")
                    try:
                        writer = pq.ParquetWriter(
                            temporary, schema, compression="zstd", compression_level=3
                        )
                        for row in parquet_rows(path):
                            if row["split"] not in {"train", "validation"}:
                                raise ValueError(
                                    "Unknown raw split; expected train or validation"
                                )
                            if row["split"] != split:
                                continue
                            if (
                                row["source"] != shard["source"]
                                or row["domain"] != raw["sources"][row["source"]]
                            ):
                                raise ValueError(
                                    "Row source/domain disagrees with raw manifest"
                                )
                            group = row.get("split_group")
                            if split == "validation" and group:
                                db.execute(
                                    "INSERT OR IGNORE INTO heldout_groups VALUES (?)",
                                    (group,),
                                )
                            text = row["text"]
                            reason = quality_reason(text, min_chars)
                            if not reason and evaluation_jsonl:
                                reason = (
                                    "evaluation_overlap"
                                    if any(
                                        db.execute(
                                            "SELECT 1 FROM evaluation WHERE hash=?",
                                            (h,),
                                        ).fetchone()
                                        for h in evaluation_ngrams(text)
                                    )
                                    else None
                                )
                            if not reason:
                                reason = index.check_and_add(text, split, group)
                            if reason:
                                rejects[reason] += 1
                                continue
                            size = len(text.encode("utf-8"))
                            if row["utf8_bytes"] != size:
                                raise ValueError("Raw byte count disagrees with text")
                            rows.append(row)
                            docs += 1
                            byte_count += size
                            buffered_bytes += size
                            if buffered_bytes >= 8_000_000:
                                writer.write_table(
                                    pa.Table.from_pylist(rows, schema=schema)
                                )
                                rows, buffered_bytes = [], 0
                        if rows:
                            writer.write_table(
                                pa.Table.from_pylist(rows, schema=schema)
                            )
                        writer.close()
                        writer = None
                        os.replace(temporary, target)
                        result = {
                            "path": relative,
                            "sha256": file_hash(target),
                            "source": shard["source"],
                            "domain": raw["sources"][shard["source"]],
                            "split": split,
                            "docs": docs,
                            "raw_bytes": byte_count,
                            "rejected": dict(rejects),
                        }
                        db.execute(
                            "INSERT INTO completed VALUES (?,?)",
                            (key, json.dumps(result)),
                        )
                        db.commit()
                        print(
                            f"{split}/{shard['source']}: kept {docs:,}; rejected {sum(rejects.values()):,}",
                            flush=True,
                        )
                    except BaseException:
                        if writer:
                            writer.close()
                        db.rollback()
                        raise
                    _report(db, output, recipe, raw, False)
            return _report(db, output, recipe, raw, True)
        finally:
            db.close()


def _report(db, output, recipe, raw, complete):
    shards = [
        json.loads(row[0])
        for row in db.execute("SELECT result FROM completed ORDER BY key")
    ]
    stats = {}
    rejected = Counter()
    for shard in shards:
        key = f"{shard['split']}/{shard['source']}"
        item = stats.setdefault(key, {"docs": 0, "raw_bytes": 0})
        item["docs"] += shard["docs"]
        item["raw_bytes"] += shard["raw_bytes"]
        rejected.update(shard["rejected"])
    manifest = {
        "version": VERSION,
        "complete": complete,
        "recipe": recipe,
        "sources": raw["sources"],
        "shards": shards,
        "statistics": stats,
        "rejected": dict(rejected),
        "evaluation_screened": recipe["evaluation_sha256"] is not None,
        "notes": [
            "SimHash is approximate and greedy; no exhaustive cluster guarantee.",
            "Credential checks match selected patterns; no complete privacy audit.",
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest
