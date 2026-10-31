"""Conservative document rejection, without changing code or prose bytes."""

import hashlib
import re

import numpy as np

WORDS = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SECRET = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bhf_[A-Za-z0-9]{30,}\b|\bgh[pousr]_[A-Za-z0-9]{30,}\b"
)


def quality_reason(text, min_chars=40):
    if not isinstance(text, str) or len(text.strip()) < min_chars:
        return "too_short"
    if "\x00" in text:
        return "nul"
    if sum(ord(c) < 32 and c not in "\n\r\t" for c in text) > len(text) * 0.005:
        return "control_characters"
    if SECRET.search(text):
        return "credential_pattern"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 30 and len(set(lines)) / len(lines) < 0.1:
        return "extreme_line_repetition"
    return None


def normalized_hash(text):
    # Exact after whitespace folding only; original retained text is never changed.
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def simhash(text):
    """128-bit, case-sensitive 5-token shingles; bounded deterministic sampling.

    Sampling makes this an approximate document filter, not exhaustive near-dedup.
    """
    tokens = WORDS.findall(text)
    if len(tokens) < 20:
        return None
    hashes = set()
    stride = max(1, (len(tokens) - 4) // 4096)
    for offset in range(0, len(tokens) - 4, stride):
        value = "\x1f".join(tokens[offset : offset + 5]).encode()
        hashes.add(hashlib.blake2b(value, digest_size=16).digest())
    bits = np.unpackbits(
        np.frombuffer(b"".join(sorted(hashes)), dtype=np.uint8)
    ).reshape(-1, 128)
    packed = np.packbits(bits.sum(0) * 2 >= len(hashes)).tobytes()
    return int.from_bytes(packed, "big")


def evaluation_ngrams(text, width=13):
    words = text.lower().split()
    for i in range(len(words) - width + 1):
        yield hashlib.blake2b(
            " ".join(words[i : i + width]).encode(), digest_size=16
        ).digest()


class DuplicateIndex:
    def __init__(self, db, near=True):
        self.db, self.near = db, near
        db.execute(
            "CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY, split TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS bands (band INTEGER, key INTEGER, fingerprint TEXT, size INTEGER, PRIMARY KEY(band,key,fingerprint)) WITHOUT ROWID"
        )
        db.execute("CREATE TABLE IF NOT EXISTS heldout_groups (name TEXT PRIMARY KEY)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS evaluation (hash BLOB PRIMARY KEY) WITHOUT ROWID"
        )
        db.commit()

    def check_and_add(self, text, split, group):
        if (
            split == "train"
            and group
            and self.db.execute(
                "SELECT 1 FROM heldout_groups WHERE name=?", (group,)
            ).fetchone()
        ):
            return "heldout_group"
        exact = normalized_hash(text)
        if self.db.execute("SELECT 1 FROM seen WHERE hash=?", (exact,)).fetchone():
            return "normalized_duplicate"
        fp = simhash(text) if self.near else None
        if fp is not None:
            for band in range(4):
                key = (fp >> (band * 32)) & 0xFFFFFFFF
                for previous, size in self.db.execute(
                    "SELECT fingerprint,size FROM bands WHERE band=? AND key=?",
                    (band, key),
                ):
                    if (
                        min(size, len(text)) / max(size, len(text)) >= 0.8
                        and (fp ^ int(previous, 16)).bit_count() <= 3
                    ):
                        return "simhash_near_duplicate"
        self.db.execute("INSERT INTO seen VALUES (?,?)", (exact, split))
        if fp is not None:
            self.db.executemany(
                "INSERT OR IGNORE INTO bands VALUES (?,?,?,?)",
                [
                    (i, (fp >> (i * 32)) & 0xFFFFFFFF, f"{fp:032x}", len(text))
                    for i in range(4)
                ],
            )
        return None
