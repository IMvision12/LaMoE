"""Train a byte-level BPE on a bounded, deterministic, training-only sample."""

import hashlib
import importlib.metadata
import random
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .common import (
    SPECIAL_TOKENS,
    VERSION,
    atomic_json,
    complete_manifest,
    file_hash,
    parquet_rows,
    stage_lock,
)
from .tokenizer_profiles import DEFAULT_PROFILE, make_pre_tokenizer

DOMAIN_SHARES = {"code": 0.35, "cyber": 0.25, "general": 0.30, "math": 0.10}


def sample_corpus(clean_manifest, split, sample_bytes, seed):
    """Bounded deterministic sample. Returns identical ordered texts for both profiles."""
    manifest = complete_manifest(clean_manifest)
    rng = random.Random(seed)
    selected, actual = [], {}
    for domain, share in DOMAIN_SHARES.items():
        budget = int(sample_bytes * share)
        candidates = [
            s
            for s in manifest["shards"]
            if s["split"] == split and s["domain"] == domain and s["docs"]
        ]
        rng.shuffle(candidates)
        total = 0
        for shard in candidates:
            path = Path(clean_manifest).parent / shard["path"]
            if file_hash(path) != shard["sha256"]:
                raise ValueError("Cleaned shard checksum mismatch")
            documents, size = [], 0
            for row in parquet_rows(path):
                if row["split"] != split or row["domain"] != domain:
                    raise ValueError("Sample row disagrees with shard split/domain")
                documents.append(row["text"])
                size += len(row["text"].encode("utf-8"))
                if size >= budget - total:
                    break
            rng.shuffle(documents)
            for text in documents:
                selected.append((domain, text))
                total += len(text.encode("utf-8"))
                if total >= budget:
                    break
            if total >= budget:
                break
        if not total:
            raise ValueError(f"No {split} text available for tokenizer domain {domain}")
        actual[domain] = total
    rng.shuffle(selected)
    return selected, actual


def sample_hash(records):
    h = hashlib.sha256()
    for domain, text in records:
        for value in (domain, text):
            value = value.encode("utf-8")
            h.update(len(value).to_bytes(8, "big"))
            h.update(value)
    return h.hexdigest()


def train_tokenizer(
    clean_manifest,
    output,
    vocab_size=32000,
    sample_bytes=200_000_000,
    seed=122209,
    profile=DEFAULT_PROFILE,
    *,
    sample=None,
):
    complete_manifest(clean_manifest)
    pre_tokenizer = make_pre_tokenizer(profile)
    if vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError("Vocabulary must include all 256 bytes plus special tokens")
    if sample_bytes < 1000:
        raise ValueError("Tokenizer sample must be at least 1,000 bytes")
    recipe = {
        "version": VERSION,
        "clean_sha256": file_hash(clean_manifest),
        "vocab_size": vocab_size,
        "sample_bytes": sample_bytes,
        "seed": seed,
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "profile": profile,
    }
    output = Path(output)
    with stage_lock(output):
        if (output / "manifest.json").exists():
            old = complete_manifest(output / "manifest.json")
            # Manifests predating profile selection always used the original splitter.
            # Never reinterpret them using the current default.
            previous = {"profile": "bytelevel", **old["recipe"]}
            if (
                previous != recipe
                or file_hash(output / "tokenizer.json") != old["sha256"]
            ):
                raise ValueError(
                    "Tokenizer output/settings changed; choose a new directory"
                )
            return old
        selected, actual = (
            sample
            if sample is not None
            else sample_corpus(clean_manifest, "train", sample_bytes, seed)
        )
        tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = pre_tokenizer
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train_from_iterator(
            (text for _, text in selected), trainer=trainer, length=len(selected)
        )
        if tokenizer.get_vocab_size() != vocab_size:
            raise ValueError(
                f"Sample only supports {tokenizer.get_vocab_size()} tokens; increase sample size or explicitly choose a smaller vocabulary"
            )
        tokenizer.encode_special_tokens = (
            True  # Literal corpus markers must not become control IDs.
        )
        for text in [
            "def f(x):\n\treturn x + 1  \n",
            "é 中文 🙂\r\n",
            "<|eos|> <|assistant|>",
        ]:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if tokenizer.decode(ids, skip_special_tokens=False) != text or set(
                ids
            ) & set(range(len(SPECIAL_TOKENS))):
                raise ValueError("Tokenizer round-trip/special-token isolation failed")
        tokenizer.save(str(output / "tokenizer.json"))
        report = {
            "complete": True,
            "recipe": recipe,
            "sha256": file_hash(output / "tokenizer.json"),
            "vocab_size": vocab_size,
            "special_tokens": {s: tokenizer.token_to_id(s) for s in SPECIAL_TOKENS},
            "sampled_bytes": actual,
            "sampled_documents": len(selected),
            "sample_sha256": sample_hash(selected),
            "notes": [
                "Training-only bounded shard sample; not a uniform corpus-wide sample.",
                "BPE training is restarted if interrupted; downstream tokenization is resumable.",
            ],
        }
        atomic_json(output / "manifest.json", report)
        atomic_json(output / "special_tokens_map.json", report["special_tokens"])
        return report


def load_tokenizer(directory):
    directory = Path(directory)
    manifest = complete_manifest(directory / "manifest.json")
    if file_hash(directory / "tokenizer.json") != manifest["sha256"]:
        raise ValueError("Tokenizer fingerprint changed")
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.encode_special_tokens = True
    return tokenizer, manifest
