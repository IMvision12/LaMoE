"""Screen existing cleaned text; exclude matching token documents without retokenizing.

CLI: python -m lamoe.preprocessing.screen --help
Exact whitespace-normalized matches and case-folded 13-word overlap only;
this is not semantic decontamination or a guarantee against all leakage.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .common import (
    atomic_json,
    complete_manifest,
    file_hash,
    parquet_rows,
    read_json,
    stage_lock,
)
from .filters import normalized_hash, evaluation_ngrams
from .token_store import local_path, verify_shard


def screen(clean_path, token_path, evaluation, output):
    clean, tokens = complete_manifest(clean_path), complete_manifest(token_path)
    if tokens["recipe"]["clean_sha256"] != file_hash(clean_path):
        raise ValueError("Token corpus does not belong to this cleaned manifest")
    exact, grams = set(), set()
    with open(evaluation, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            # Include prompts, solutions, suffixes and tests in the frozen file.
            texts = [
                row[k]
                for k in ("text", "prompt", "answer", "reference", "suffix", "tests")
                if isinstance(row.get(k), str) and row[k].strip()
            ]
            for text in texts:
                exact.add(normalized_hash(text))
                grams.update(evaluation_ngrams(text))
    if not exact:
        raise ValueError("Evaluation JSONL has no nonempty text fields")
    root = Path(output)
    recipe = dict(
        clean_sha256=file_hash(clean_path),
        token_manifest_sha256=file_hash(token_path),
        evaluation_sha256=file_hash(evaluation),
        method="exact-whitespace+13word-v1",
    )
    with stage_lock(root):
        progress_path = root / "progress.json"
        progress = (
            read_json(progress_path)
            if progress_path.exists()
            else dict(recipe=recipe, completed=[], excluded=[])
        )
        if progress["recipe"] != recipe:
            raise ValueError("Screening recipe changed; use a new output directory")
        excluded = set(progress["excluded"])
        for shard in clean["shards"]:
            if shard["path"] in progress["completed"]:
                continue
            path = local_path(Path(clean_path).parent, shard["path"])
            if file_hash(path) != shard["sha256"]:
                raise ValueError("Cleaned shard checksum mismatch")
            for row in parquet_rows(path):
                text = row["text"]
                if normalized_hash(text) in exact or any(
                    g in grams for g in evaluation_ngrams(text)
                ):
                    text_hash = hashlib.sha256(text.encode()).hexdigest()
                    excluded.add(
                        hashlib.sha256(
                            (row["source"] + ":" + text_hash).encode()
                        ).hexdigest()
                    )
            progress["completed"].append(shard["path"])
            progress["excluded"] = sorted(excluded)
            atomic_json(progress_path, progress)
            print(
                json.dumps(
                    dict(
                        screened_shards=len(progress["completed"]),
                        total_shards=len(clean["shards"]),
                        excluded_documents=len(excluded),
                    )
                ),
                flush=True,
            )
        removed, empty = {}, []
        for shard in tokens["shards"]:
            verify_shard(Path(token_path).parent, shard)
            idx = np.load(
                local_path(Path(token_path).parent, shard["index_path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            mask = np.fromiter(
                (bytes(r["document_id"]).hex() in excluded for r in idx), bool, len(idx)
            )
            key = shard["split"] + "/" + shard["source"]
            removed[key] = removed.get(key, 0) + int(idx["length"][mask].sum())
            if mask.all():
                empty.append(shard["path"])
            idx._mmap.close()
        report = dict(
            complete=True,
            **recipe,
            excluded_document_ids=sorted(excluded),
            removed_tokens=removed,
            empty_shards=empty,
            limitation="Only this frozen evaluation set and these matching rules were screened.",
        )
        atomic_json(root / "report.json", report)
        return report


def load_screening(path, token_manifest):
    report = read_json(path)
    if report.get("complete") is not True or report.get(
        "token_manifest_sha256"
    ) != file_hash(token_manifest):
        raise ValueError("Incomplete screening report or wrong token corpus")
    if report.get("method") != "exact-whitespace+13word-v1":
        raise ValueError("Unknown screening method")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-manifest", required=True)
    p.add_argument("--token-manifest", required=True)
    p.add_argument("--evaluation-jsonl", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    screen(a.clean_manifest, a.token_manifest, a.evaluation_jsonl, a.output)
