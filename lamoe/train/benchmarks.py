"""Frozen JSONL capability probes. Code execution is opt-in and Docker-only.

Rows: id, domain, kind (exact/fim/code), prompt, answer/reference; FIM also
suffix. Code rows include tests. Security factual accuracy still needs expert
review; exact match is NOT a security-quality benchmark.
"""

import argparse
import json
import math
import subprocess
import uuid
from pathlib import Path
import torch
from lamoe.model import LaMoEForCausalLM
from lamoe.preprocessing.common import atomic_json, file_hash
from lamoe.preprocessing.tokenizer import load_tokenizer


class CodeGraderInfrastructureError(RuntimeError):
    """The grader did not yield a valid candidate outcome; do not publish a zero."""


def sandbox_python(code, tests, image="python:3.12-slim", timeout=10):
    """No host mounts, credentials, network or unsandboxed fallback. Local image required."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Code grader timeout must be positive")
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodeGraderInfrastructureError(
            "Docker/local grading image is unavailable"
        ) from exc
    name = "lamoe-eval-" + uuid.uuid4().hex
    harness = "import sys,json,os; d=json.load(sys.stdin); sys.stdout=open(os.devnull,'w'); sys.stderr=sys.stdout; exec(compile(d['code']+chr(10)+d['tests'],'<evaluation>','exec'),{'__name__':'__main__'})"
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--name",
                name,
                "--network",
                "none",
                "--memory",
                "256m",
                "--cpus",
                "1",
                "--pids-limit",
                "32",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--user",
                "65534:65534",
                image,
                "python",
                "-I",
                "-c",
                harness,
            ],
            input=json.dumps(dict(code=code, tests=tests)),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        if result.returncode in (125, 126, 127) or result.returncode < 0:
            raise CodeGraderInfrastructureError(
                f"Docker grader failed with exit status {result.returncode}"
            )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CodeGraderInfrastructureError(
            "Docker launch/host grading timeout; no candidate score"
        ) from exc
    finally:
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CodeGraderInfrastructureError(
                "Owned grading container cleanup could not be confirmed"
            ) from exc


@torch.no_grad()
def run(
    checkpoint,
    tokenizer_dir,
    examples,
    output,
    device="cpu",
    max_new_tokens=256,
    execute_code=False,
):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    tok, metadata = load_tokenizer(tokenizer_dir)
    model = LaMoEForCausalLM.from_pretrained(checkpoint).to(device).eval()
    if model.config.vocab_size != metadata["vocab_size"]:
        raise ValueError("Tokenizer/model vocabulary mismatch")
    progress_path = Path(checkpoint) / "progress.json"
    if progress_path.exists():
        saved = json.loads(progress_path.read_text())["recipe"]["tokenizer"]
        if saved["sha256"] != metadata["sha256"]:
            raise ValueError("Frozen tokenizer hash differs from checkpoint")
    results = []
    with open(examples, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Nonempty benchmark and unique IDs required")
    specials = metadata["special_tokens"]
    for row in rows:
        kind = row["kind"]
        if kind not in ("exact", "fim", "code"):
            raise ValueError("Unknown benchmark kind")
        prefix = tok.encode(row["prompt"], add_special_tokens=False).ids
        if kind == "fim":
            prefix = (
                [specials["<|fim_prefix|>"]]
                + prefix
                + [specials["<|fim_suffix|>"]]
                + tok.encode(row["suffix"], add_special_tokens=False).ids
                + [specials["<|fim_middle|>"]]
            )
        if (
            not prefix
            or len(prefix) + max_new_tokens > model.config.max_position_embeddings
        ):
            raise ValueError(
                f"Prompt {row['id']} exceeds reserved context; do not truncate silently"
            )
        generated, cache = [], None
        ids = torch.tensor([prefix], device=device)
        from contextlib import nullcontext

        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        )
        with context:
            for _ in range(max_new_tokens):
                out = model(
                    ids, use_cache=True, past_key_values=cache, logits_to_keep=1
                )
                cache = out.past_key_values
                token = int(out.logits[0, -1].argmax())
                if token == specials["<|eos|>"]:
                    break
                generated.append(token)
                ids = torch.tensor([[token]], device=device)
        text = tok.decode(generated, skip_special_tokens=False)
        reference = row.get("answer", row.get("reference"))
        passed = (
            text.strip() == reference.strip() if isinstance(reference, str) else None
        )
        reference_ce = None
        if isinstance(reference, str):
            target = tok.encode(reference, add_special_tokens=False).ids + [
                specials["<|eos|>"]
            ]
            if len(prefix) + len(target) <= model.config.max_position_embeddings:
                all_ids = torch.tensor([prefix + target], device=device)
                labels = all_ids.clone()
                labels[:, : len(prefix)] = -100
                with (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda"
                    else nullcontext()
                ):
                    scored = model(
                        all_ids, labels=labels, loss_chunk_size=128, return_logits=False
                    )
                reference_ce = float(scored.ce_loss)
        if kind == "code":
            if execute_code and (
                not isinstance(row.get("tests"), str) or not row["tests"].strip()
            ):
                raise ValueError("Code evaluation requires nonempty trusted tests")
            passed = sandbox_python(text, row["tests"]) if execute_code else None
        results.append(
            dict(
                id=row["id"],
                domain=row["domain"],
                kind=kind,
                response=text,
                passed=passed,
                reference_target_ce=reference_ce,
                generated_tokens=len(generated),
                length_limit=len(generated) == max_new_tokens,
            )
        )
    summaries = {}
    for key in sorted({r["domain"] + "/" + r["kind"] for r in results}):
        selected = [
            r
            for r in results
            if r["domain"] + "/" + r["kind"] == key and r["passed"] is not None
        ]
        summaries[key] = dict(
            scored=len(selected),
            success_rate=sum(r["passed"] for r in selected) / len(selected)
            if selected
            else None,
        )
    report = dict(
        checkpoint=str(checkpoint),
        tokenizer_sha256=metadata["sha256"],
        evaluation_sha256=file_hash(examples),
        decoding="greedy/pass@1",
        max_new_tokens=max_new_tokens,
        results=results,
        summary=summaries,
    )
    atomic_json(output, report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer-dir", "examples", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--execute-code", action="store_true")
    run(**vars(p.parse_args()))
