"""Portable, local AdamW SFT baseline with epoch-boundary checkpoints.

This is a public entry point, not a claim to reproduce the private SFT curriculum.
No dataset acquisition, remote execution, tool calls or generated-code execution.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import uuid


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_checkpoint(path):
    path = Path(path)
    marker = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    names = {"model.safetensors", "config.json", "trainer.pt", "progress.json"}
    if marker.get("complete") is not True or set(marker.get("sha256", {})) != names:
        raise ValueError("Incomplete SFT checkpoint")
    if any(digest_file(path / name) != marker["sha256"][name] for name in names):
        raise ValueError("SFT checkpoint checksum mismatch")


def check_disjoint(train_rows, validation_rows):
    def signature(row):
        return hashlib.sha256(
            json.dumps([row["input_ids"], row["labels"]]).encode()
        ).hexdigest()

    if {r["id"] for r in train_rows} & {r["id"] for r in validation_rows} or {
        signature(r) for r in train_rows
    } & {signature(r) for r in validation_rows}:
        raise ValueError(
            "Training and validation rows overlap by ID or tokenized content"
        )


def weighted_loss(output, targets, inputs):
    """CE is target weighted; router losses are valid-input weighted."""
    return output.ce_loss_sum / targets + (output.loss - output.ce_loss) * (
        output.input_token_count / inputs
    )


def validation_ce(model, rows, pad, batch_size, device):
    import torch
    from .data import collate
    from lamoe.runtime import autocast

    model.eval()
    total, targets = 0.0, 0
    with torch.inference_mode(), autocast(device):
        for start in range(0, len(rows), batch_size):
            out = model(**collate(rows[start : start + batch_size], pad, device))
            total += float(out.ce_loss_sum)
            targets += int(out.target_token_count)
    return total / targets


def train(args):
    import torch
    from tokenizers import Tokenizer
    from .data import read_records, collate
    from lamoe.preprocessing.common import atomic_json, stage_lock
    from lamoe.runtime import load_local, autocast

    for name in ("epochs", "batch_size", "accumulation"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive and finite")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    tokenizer.encode_special_tokens = True
    source = Path(args.checkpoint)
    recipe = dict(
        version=1,
        optimizer="adamw",
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        gdn_backend=args.gdn_backend,
        train_sha256=digest_file(args.train_file),
        validation_sha256=digest_file(args.validation_file),
        tokenizer_sha256=digest_file(args.tokenizer),
        source_weights_sha256=digest_file(source / "model.safetensors"),
        source_config_sha256=digest_file(source / "config.json"),
    )
    output = Path(args.output)
    with stage_lock(output):
        recipe_path = output / "recipe.json"
        if recipe_path.exists():
            if not args.resume:
                raise ValueError(
                    "Existing SFT run requires an explicit epoch checkpoint resume"
                )
            if json.loads(recipe_path.read_text(encoding="utf-8")) != recipe:
                raise ValueError("SFT recipe or local input hashes changed")
        elif args.resume:
            raise ValueError("Resume requires the original output directory and recipe")
        elif any(p.name != ".writer.lock" for p in output.iterdir()):
            raise ValueError("Use a new empty output directory")
        if args.resume:
            if Path(args.resume).resolve().parent != output.resolve():
                raise ValueError(
                    "Resume checkpoint must belong to this output directory"
                )
            verify_checkpoint(args.resume)
        model = load_local(args.resume or source, args.device, args.gdn_backend)
        if tokenizer.get_vocab_size() != model.config.vocab_size:
            raise ValueError("Tokenizer/model vocabulary mismatch")
        pad = tokenizer.token_to_id("<|pad|>")
        if pad is None:
            raise ValueError("Tokenizer needs the frozen <|pad|> token")
        train_rows = read_records(
            args.train_file,
            tokenizer,
            model.config.max_position_embeddings,
            model.config.vocab_size,
        )
        valid_rows = read_records(
            args.validation_file,
            tokenizer,
            model.config.max_position_embeddings,
            model.config.vocab_size,
        )
        check_disjoint(train_rows, valid_rows)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        epoch, step = 0, 0
        if args.resume:
            resume = Path(args.resume).resolve()
            if resume.parent != output.resolve():
                raise ValueError(
                    "Resume checkpoint must belong to this output directory"
                )
            state = torch.load(
                resume / "trainer.pt", map_location="cpu", weights_only=True
            )
            if state["recipe"] != recipe:
                raise ValueError("Checkpoint recipe does not match")
            optimizer.load_state_dict(state["optimizer"])
            epoch, step = state["completed_epochs"], state["step"]
            torch.set_rng_state(state["cpu_rng"])
            if args.device == "cuda":
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        else:
            torch.manual_seed(args.seed)
        atomic_json(recipe_path, recipe)
        for epoch_index in range(epoch, args.epochs):
            model.train()
            generator = torch.Generator().manual_seed(args.seed + epoch_index)
            order = torch.randperm(len(train_rows), generator=generator).tolist()
            ordered = [train_rows[i] for i in order]
            group_size = args.batch_size * args.accumulation
            for start in range(0, len(ordered), group_size):
                group = ordered[start : start + group_size]
                targets = sum(sum(t != -100 for t in r["labels"][1:]) for r in group)
                inputs = sum(len(r["input_ids"]) for r in group)
                optimizer.zero_grad(set_to_none=True)
                model.reset_router_counts()
                for offset in range(0, len(group), args.batch_size):
                    batch = collate(
                        group[offset : offset + args.batch_size], pad, args.device
                    )
                    with autocast(args.device):
                        result = model(**batch)
                        loss = weighted_loss(result, targets, inputs)
                    if not torch.isfinite(loss):
                        raise ValueError(
                            "Non-finite SFT loss; no epoch checkpoint published"
                        )
                    loss.backward()
                    model.accumulate_router_counts(result)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                model.reset_router_counts()
                step += 1
            ce = validation_ce(model, valid_rows, pad, args.batch_size, args.device)
            if not math.isfinite(ce):
                raise ValueError("Non-finite validation CE")
            target = output / f"epoch-{epoch_index + 1:04d}"
            if target.exists():
                raise ValueError("Checkpoint exists; it will not be overwritten")
            partial = output / f".partial-{uuid.uuid4().hex}"
            partial.mkdir()
            model.save_pretrained(partial)
            state = dict(
                recipe=recipe,
                optimizer=optimizer.state_dict(),
                completed_epochs=epoch_index + 1,
                step=step,
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all()
                if args.device == "cuda"
                else [],
            )
            torch.save(state, partial / "trainer.pt")
            atomic_json(
                partial / "progress.json",
                dict(
                    completed_epochs=epoch_index + 1,
                    step=step,
                    validation_ce=ce,
                    train_records=len(train_rows),
                    validation_records=len(valid_rows),
                ),
            )
            atomic_json(
                partial / "checkpoint.json",
                dict(
                    complete=True,
                    sha256={
                        name: digest_file(partial / name)
                        for name in (
                            "model.safetensors",
                            "config.json",
                            "trainer.pt",
                            "progress.json",
                        )
                    },
                ),
            )
            partial.rename(target)
            print(
                json.dumps(dict(checkpoint=str(target), validation_ce=ce, step=step)),
                flush=True,
            )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer", "train-file", "validation-file", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--resume", help="Complete epoch checkpoint in the same output directory"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=122209)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--gdn-backend", choices=("reference", "fla"), default="reference"
    )
    train(parser.parse_args(argv))


if __name__ == "__main__":
    main()
