"""Resumable tokenization to uint16 .bin shards and compact document indexes."""

import hashlib
import random
import time
from pathlib import Path

from .common import (
    VERSION,
    atomic_json,
    complete_manifest,
    digest,
    file_hash,
    parquet_rows,
    read_json,
    stage_lock,
)
from .token_store import (
    INDEX_DTYPE,
    STORAGE_FORMAT,
    ShardWriter,
    require_binary,
    verify_shard,
)
from .tokenizer import load_tokenizer


def tokenize(
    clean_manifest,
    tokenizer_dir,
    output,
    fim_rate=0.2,
    seed=122209,
    max_hours=None,
    max_shard_tokens=64_000_000,
    max_shard_documents=100_000,
):
    clean = complete_manifest(clean_manifest)
    tokenizer, tok = load_tokenizer(tokenizer_dir)
    if not 0 <= fim_rate <= 1:
        raise ValueError("fim_rate must lie in [0,1]")
    if tok["vocab_size"] > 65536:
        raise ValueError("uint16 token store supports vocabularies up to 65536")
    if max_shard_tokens < 1 or max_shard_documents < 1:
        raise ValueError("Shard limits must be positive")
    recipe = {
        "version": VERSION,
        "storage_format": STORAGE_FORMAT,
        "clean_sha256": file_hash(clean_manifest),
        "tokenizer_sha256": tok["sha256"],
        "fim_rate": fim_rate,
        "seed": seed,
        "eos_per_document": True,
        "max_shard_tokens": max_shard_tokens,
        "max_shard_documents": max_shard_documents,
    }
    deadline = time.monotonic() + max_hours * 3600 if max_hours else float("inf")
    output = Path(output)
    with stage_lock(output):
        if (output / "tokens.sqlite").exists():
            raise ValueError(
                "Legacy tokens.sqlite exists; use a new output directory. It has not been modified."
            )
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            state = read_json(manifest_path)
            require_binary(state)
            if state["recipe"] != recipe:
                raise ValueError(
                    "Tokenization settings/input changed; use a new output directory"
                )
            for shard in state["shards"]:
                verify_shard(output, shard)
        else:
            if any(p.name != ".writer.lock" for p in output.iterdir()):
                raise ValueError(
                    "Token output is not empty and has no manifest; use a new directory"
                )
            state = {
                "version": VERSION,
                "complete": False,
                "recipe": recipe,
                "sources": clean["sources"],
                "tokenizer": tok,
                "storage_format": STORAGE_FORMAT,
                "token_dtype": "<u2",
                "index_record_bytes": INDEX_DTYPE.itemsize,
                "shards": [],
                "completed_inputs": [],
                "counts": {},
            }
            _report(output, state)
        if state["complete"]:
            return _report(output, state)
        completed = set(state["completed_inputs"])
        for shard in clean["shards"]:
            if shard["path"] in completed:
                continue
            if time.monotonic() >= deadline:
                return _report(output, state)
            path = Path(clean_manifest).parent / shard["path"]
            if file_hash(path) != shard["sha256"]:
                raise ValueError("Cleaned shard checksum mismatch")
            if (
                shard["split"] not in ("train", "validation")
                or shard["source"] not in clean["sources"]
                or shard["domain"] != clean["sources"][shard["source"]]
            ):
                raise ValueError("Invalid cleaned source, domain, or split")
            writer, published = None, []
            docs = byte_count = 0
            try:
                for row in parquet_rows(path):
                    if any(row[k] != shard[k] for k in ("source", "split", "domain")):
                        raise ValueError(
                            "Document source/split/domain differs from cleaned shard"
                        )
                    text = row["text"]
                    encoded_text = text.encode("utf-8")
                    text_hash = hashlib.sha256(encoded_text).hexdigest()
                    raw_bytes = len(encoded_text)
                    if (
                        text_hash != row["text_sha256"]
                        or raw_bytes != row["utf8_bytes"]
                    ):
                        raise ValueError("Document text hash/byte count mismatch")
                    identity = hashlib.sha256(
                        (row["source"] + ":" + text_hash).encode()
                    ).hexdigest()
                    rng = random.Random(
                        hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()
                    )
                    ids = tokenizer.encode(text, add_special_tokens=False).ids
                    fim = (
                        row["split"] == "train"
                        and row["domain"] == "code"
                        and len(ids) >= 3
                        and rng.random() < fim_rate
                    )
                    specials = tok["special_tokens"]
                    if fim:
                        a, b = sorted(rng.sample(range(len(ids) + 1), 2))
                        ids = (
                            [specials["<|fim_prefix|>"]]
                            + ids[:a]
                            + [specials["<|fim_suffix|>"]]
                            + ids[b:]
                            + [specials["<|fim_middle|>"]]
                            + ids[a:b]
                        )
                    ids.append(specials["<|eos|>"])
                    if min(ids) < 0 or max(ids) >= tok["vocab_size"]:
                        raise ValueError("Token ID outside vocabulary")
                    if writer and (
                        writer.tokens + len(ids) > max_shard_tokens
                        or len(writer.records) >= max_shard_documents
                    ):
                        published.append(writer.finish())
                        writer = None
                    if writer is None:
                        relative = f"{shard['split']}/{shard['source']}/{digest(shard['path'])}-{len(published):05d}.bin"
                        writer = ShardWriter(
                            output,
                            relative,
                            shard["source"],
                            shard["split"],
                            shard["domain"],
                            shard["path"],
                        )
                    writer.append(ids, identity, raw_bytes, fim)
                    docs += 1
                    byte_count += raw_bytes
                if docs != shard["docs"] or byte_count != shard["raw_bytes"]:
                    raise ValueError("Cleaned shard totals disagree with documents")
                if writer:
                    published.append(writer.finish())
                    writer = None
            finally:
                if writer:
                    writer.close()
            # Before this atomic commit, published files are orphans. Replay uses
            # identical names and recipe, overwriting only this input's outputs.
            state["shards"].extend(published)
            state["completed_inputs"].append(shard["path"])
            completed.add(shard["path"])
            _report(output, state)
            print(
                f"Tokenized {shard['path']}: {docs:,} documents, {len(published)} binary shards",
                flush=True,
            )
        state["complete"] = True
        return _report(output, state)


def _report(output, state):
    counts = {}
    for shard in state["shards"]:
        item = counts.setdefault(shard["split"], {}).setdefault(
            shard["source"],
            {"documents": 0, "tokens": 0, "raw_bytes": 0, "fim_documents": 0},
        )
        for key in ("documents", "tokens", "raw_bytes", "fim_documents"):
            item[key] += shard[key]
    for sources in counts.values():
        for item in sources.values():
            item["bytes_per_token"] = (
                item["raw_bytes"] / item["tokens"] if item["tokens"] else None
            )
    state["counts"] = counts
    state["token_bytes"] = sum(s["token_bytes"] for s in state["shards"])
    state["index_bytes"] = sum(s["index_bytes"] for s in state["shards"])
    atomic_json(output / "manifest.json", state)
    atomic_json(
        output / "train_token_counts.json",
        {s: v["tokens"] for s, v in counts.get("train", {}).items()},
    )
    return state
