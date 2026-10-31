"""Paired CPU tokenizer comparison; this measures encoding, not model quality."""

import os
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

from .common import atomic_json, complete_manifest, file_hash, read_json, stage_lock
from .filters import normalized_hash
from .tokenizer import (
    DOMAIN_SHARES,
    load_tokenizer,
    sample_corpus,
    sample_hash,
    train_tokenizer,
)
from .tokenizer_profiles import DEEPSEEK_SOURCE, PROFILES

STRESS_TEXTS = [
    'def f(x):\r\n\treturn "你好🙂"  \n',
    "1234567890 00001234 3.141592653589793 1e-10 0xDEADBEEF 192.168.1.1",
    "CVE-2024-12345 /var/log/auth.log https://example.org/a?q=1234&b=2",
    "e\u0301 é العربية हिन्दी 中文 日本語 한글 🧑🏽\u200d💻",
    "    \t\n\r\n\u00a0\u200b",
    "<|eos|><|assistant|><|fim_middle|><|unk|>",
]


def compare_tokenizers(
    clean_manifest,
    output,
    vocab_size=32000,
    sample_bytes=50_000_000,
    evaluation_bytes=5_000_000,
    seed=122209,
    freeze=None,
):
    if sample_bytes < 1000 or evaluation_bytes < 1000:
        raise ValueError(
            "Train and evaluation samples must each request at least 1,000 bytes"
        )
    clean_manifest, output = Path(clean_manifest), Path(output)
    corpus = complete_manifest(clean_manifest)
    recipe = {
        "version": 1,
        "clean_sha256": file_hash(clean_manifest),
        "vocab_size": vocab_size,
        "sample_bytes": sample_bytes,
        "evaluation_bytes": evaluation_bytes,
        "seed": seed,
        "profiles": list(PROFILES),
        "selection_policy": "compression-v1",
    }
    with stage_lock(output):
        recipe_path = output / "comparison_recipe.json"
        if recipe_path.exists() and read_json(recipe_path) != recipe:
            raise ValueError("Comparison settings changed; use a new output directory")
        atomic_json(recipe_path, recipe)
        report_path = output / "report.json"
        if report_path.exists():
            report = complete_manifest(report_path)
            if report["recipe"] != recipe:
                raise ValueError("Comparison report settings changed")
            for profile in PROFILES:
                _, existing = load_tokenizer(output / profile)
                if (
                    existing["sha256"]
                    != report["measurements"][profile]["tokenizer_sha256"]
                ):
                    raise ValueError("Compared tokenizer was modified")
            if freeze:
                if report["recommended_profile"] is None:
                    raise ValueError(
                        "Insufficient evidence/correctness failures prevent freezing"
                    )
                freeze_tokenizer(
                    output / report["recommended_profile"], freeze, report_path
                )
            return report
        training = sample_corpus(clean_manifest, "train", sample_bytes, seed)
        evaluation, eval_bytes = sample_corpus(
            clean_manifest, "validation", evaluation_bytes, seed + 1
        )
        train_hashes = {normalized_hash(text) for _, text in training[0]}
        if any(normalized_hash(text) in train_hashes for _, text in evaluation):
            raise ValueError(
                "Training/evaluation text overlap found; clean the corpus first"
            )
        expected_hash = sample_hash(training[0])
        evaluated_hash = sample_hash(evaluation)
        measured, encoders, counts = {}, {}, {}
        for profile in PROFILES:
            directory = output / profile
            cached = (directory / "manifest.json").exists()
            started = time.perf_counter()
            trained = train_tokenizer(
                clean_manifest,
                directory,
                vocab_size,
                sample_bytes,
                seed,
                profile,
                sample=training,
            )
            training_seconds = None if cached else time.perf_counter() - started
            if trained.get("sample_sha256") != expected_hash:
                raise ValueError(
                    "Candidate was not trained on the identical ordered sample"
                )
            tokenizer, _ = load_tokenizer(directory)
            encoders[profile] = tokenizer
            special_ids = set(trained["special_tokens"].values())
            counts[profile] = {domain: [] for domain in DOMAIN_SHARES}
            failures = 0
            for domain, text in evaluation:
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                failures += int(
                    tokenizer.decode(ids, skip_special_tokens=False) != text
                    or bool(special_ids.intersection(ids))
                )
                counts[profile][domain].append(len(ids))
            stress_counts = []
            for text in STRESS_TEXTS:
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                failures += int(
                    tokenizer.decode(ids, skip_special_tokens=False) != text
                    or bool(special_ids.intersection(ids))
                )
                stress_counts.append(len(ids))
            metrics = {}
            for domain, values in counts[profile].items():
                total = sum(values)
                metrics[domain] = {
                    "documents": len(values),
                    "utf8_bytes": eval_bytes[domain],
                    "tokens": total,
                    "bytes_per_token": eval_bytes[domain] / total,
                    "tokens_per_document": total / len(values),
                    "p95_document_tokens": float(np.percentile(values, 95)),
                }
            measured[profile] = {
                "tokenizer_sha256": trained["sha256"],
                "training_seconds": training_seconds,
                "domains": metrics,
                "roundtrip_or_control_failures": failures,
                "stress_token_counts": stress_counts,
                "encoding_seconds": [],
            }
        # Warm up both, then alternate order to reduce first-run/thermal timing bias.
        for tokenizer in encoders.values():
            for _, text in evaluation[:10]:
                tokenizer.encode(text, add_special_tokens=False)
        for repeat in range(3):
            for profile in PROFILES[:: 1 if repeat % 2 == 0 else -1]:
                started = time.perf_counter()
                for _, text in evaluation:
                    encoders[profile].encode(text, add_special_tokens=False)
                measured[profile]["encoding_seconds"].append(
                    time.perf_counter() - started
                )
        for metrics in measured.values():
            median = statistics.median(metrics["encoding_seconds"])
            metrics["median_encoding_mb_per_second"] = (
                sum(eval_bytes.values()) / median / 1_000_000
            )
        base, candidate = counts["bytelevel"], counts["deepseek_v3"]
        domain_changes = {
            d: 100 * (sum(candidate[d]) / sum(base[d]) - 1) for d in DOMAIN_SHARES
        }
        scores = {
            p: sum(
                DOMAIN_SHARES[d] * sum(counts[p][d]) / eval_bytes[d]
                for d in DOMAIN_SHARES
            )
            for p in PROFILES
        }
        gain = 100 * (1 - scores["deepseek_v3"] / scores["bytelevel"])
        # Paired document bootstrap; descriptive uncertainty, not independent shard coverage.
        lengths = {
            d: np.asarray(
                [len(t.encode("utf-8")) for domain, t in evaluation if domain == d]
            )
            for d in DOMAIN_SHARES
        }
        arrays = {
            p: {d: np.asarray(v) for d, v in c.items()} for p, c in counts.items()
        }
        rng, gains = np.random.default_rng(seed), []
        for _ in range(300):
            bootstrap = {p: 0.0 for p in PROFILES}
            for domain, share in DOMAIN_SHARES.items():
                indices = rng.integers(0, len(lengths[domain]), len(lengths[domain]))
                nbytes = lengths[domain][indices].sum()
                for profile in PROFILES:
                    bootstrap[profile] += (
                        share * arrays[profile][domain][indices].sum() / nbytes
                    )
            gains.append(100 * (1 - bootstrap["deepseek_v3"] / bootstrap["bytelevel"]))
        interval = [float(x) for x in np.percentile(gains, [2.5, 97.5])]
        enough = sum(training[1].values()) >= 20_000_000 and all(
            len(base[d]) >= 100 and eval_bytes[d] >= 100_000 for d in DOMAIN_SHARES
        )
        valid = all(m["roundtrip_or_control_failures"] == 0 for m in measured.values())
        decision = (
            "deepseek_v3"
            if gain >= 1 and interval[0] > 0 and max(domain_changes.values()) <= 2
            else "bytelevel"
        )
        if not enough or not valid:
            decision = None
        report = {
            "complete": True,
            "recipe": recipe,
            "training_sample_sha256": expected_hash,
            "evaluation_sample_sha256": evaluated_hash,
            "training_bytes": training[1],
            "evaluation_bytes": eval_bytes,
            "corpus_provenance": corpus.get("provenance"),
            "deepseek_reference": DEEPSEEK_SOURCE,
            "measurements": measured,
            "deepseek_token_change_percent_by_domain": domain_changes,
            "deepseek_weighted_token_reduction_percent": gain,
            "paired_document_bootstrap_95_percent": interval,
            "sufficient_sample": enough,
            "correctness_passed": valid,
            "recommended_profile": decision,
            "selection_rule": "Use DeepSeek only with >=1% weighted tokens/byte reduction, positive bootstrap lower bound, and <=2% token regression in every domain; otherwise retain baseline. Require >=20 MB train, >=100 evaluation docs and >=100 KB per domain, and zero correctness failures.",
            "limitations": [
                "Compression is not model accuracy or learning speed.",
                "Byte-weighted score is a proxy, not the final token-budget mixture.",
                "Bounded shard-prefix sampling; bootstrap does not measure corpus/shard bias.",
                "Validation used for tokenizer selection is development data, not an untouched final test.",
                "One training seed; encoding speed is descriptive and is not the selection criterion.",
            ],
        }
        atomic_json(output / "report.json", report)
        if freeze:
            if decision is None:
                raise ValueError(
                    "Report saved, but insufficient evidence/correctness failures prevent freezing"
                )
            freeze_tokenizer(output / decision, freeze, output / "report.json")
        return report


def freeze_tokenizer(candidate, destination, report_path):
    """Publish a selected tokenizer once. Never replace an incompatible frozen tokenizer."""
    candidate, destination = Path(candidate), Path(destination)
    _, manifest = load_tokenizer(candidate)
    report = complete_manifest(report_path)
    selected = report["recommended_profile"]
    if (
        selected is None
        or not report["correctness_passed"]
        or not report["sufficient_sample"]
        or report["measurements"][selected]["tokenizer_sha256"] != manifest["sha256"]
    ):
        raise ValueError("Candidate is not the eligible comparison winner")
    with stage_lock(destination):
        if (destination / "manifest.json").exists():
            _, existing = load_tokenizer(destination)
            if existing["sha256"] != manifest["sha256"]:
                raise ValueError("Frozen tokenizer differs; refusing to overwrite")
            return
        if any(p.name != ".writer.lock" for p in destination.iterdir()):
            raise ValueError(
                "Freeze destination is not empty; inspect it before retrying"
            )
        temp = destination / "tokenizer.json.partial"
        shutil.copyfile(candidate / "tokenizer.json", temp)
        if file_hash(temp) != manifest["sha256"]:
            raise ValueError("Tokenizer copy checksum mismatch")
        os.replace(temp, destination / "tokenizer.json")
        atomic_json(destination / "special_tokens_map.json", manifest["special_tokens"])
        manifest = {
            **manifest,
            "frozen": True,
            "selection_report_sha256": file_hash(report_path),
        }
        atomic_json(destination / "manifest.json", manifest)
