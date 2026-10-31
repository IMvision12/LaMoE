"""Local SFT records, assistant-only labels and padded batches. No acquisition."""

import hashlib
import json
from pathlib import Path


def source_records(path):
    """Stream physical JSONL lines or bounded Parquet batches."""
    path = Path(path)
    if path.name.endswith((".parquet", ".parquet.partial")):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    else:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def iter_records(path, tokenizer=None, maximum_context=2048, vocab_size=32000):
    from lamoe.chat import render_record

    identities = set()
    for number, row in enumerate(source_records(path), 1):
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at row {number}")
        identity = (
            row.get("id")
            or hashlib.sha256(
                json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
        )
        if not isinstance(identity, str):
            raise ValueError(f"Record ID must be a string at row {number}")
        if identity in identities:
            raise ValueError(f"Duplicate SFT record at row {number}")
        identities.add(identity)
        if "input_ids" in row:
            ids, labels = row["input_ids"], row["labels"]
        else:
            if tokenizer is None:
                raise ValueError("Message records require a local tokenizer")
            messages = row.get("messages")
            if (
                not messages
                or not all(isinstance(m, dict) for m in messages)
                or not any(m.get("role") == "assistant" for m in messages)
            ):
                raise ValueError(f"No assistant target in row {number}")
            if messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"SFT record must end with an assistant turn at row {number}"
                )
            if any(
                m.get("role") not in {"user", "assistant", "system", "tool"}
                or not isinstance(m.get("content"), str)
                for m in messages
            ):
                raise ValueError(f"Invalid message schema at row {number}")
            ids, labels = render_record(row, tokenizer)
        if not 2 <= len(ids) <= maximum_context or len(ids) != len(labels):
            raise ValueError(
                f"Invalid/context-limited SFT row {number}; no truncation is applied"
            )
        if any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
            raise ValueError(f"Invalid token ID in row {number}")
        if labels[0] != -100 or not any(t != -100 for t in labels[1:]):
            raise ValueError(f"Invalid target mask in row {number}")
        if any(
            type(t) is not int or (t != -100 and t != i) for i, t in zip(ids, labels)
        ):
            raise ValueError(
                f"Labels must mask or equal the corresponding input token in row {number}"
            )
        yield {"id": identity, "input_ids": list(ids), "labels": list(labels)}
    if not identities:
        raise ValueError("SFT input is empty")


def read_records(path, tokenizer=None, maximum_context=2048, vocab_size=32000):
    return list(iter_records(path, tokenizer, maximum_context, vocab_size))


def collate(rows, pad, device="cpu"):
    import torch

    length = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), length), pad, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"], device=device)
        labels[i, :n] = torch.tensor(row["labels"], device=device)
        mask[i, :n] = True
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}
