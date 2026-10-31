"""Fixed-domain token-weighted probes, kept separate from capability benchmarks."""

import math
import torch
from .data import TokenBatchLoader, to_device

DOMAIN_WEIGHTS = {"code": 0.35, "cyber": 0.25, "general": 0.30, "math": 0.10}


@torch.no_grad()
def evaluate_domains(model, config, autocast_context, budget=None):
    was_training = model.training
    model.eval()
    results = {}
    try:
        for domain in DOMAIN_WEIGHTS:
            loader = TokenBatchLoader(
                config.token_manifest,
                split="validation",
                sequence_length=config.sequence_length,
                batch_size=config.batch_size,
                seed=config.seed + 1,
                domain=domain,
                screening_report=config.screening_report,
            )
            loss = targets = inputs = 0
            try:
                for batch in loader:
                    if (
                        budget is not None
                        and budget.remaining_seconds < config.save_margin_seconds
                    ):
                        return {"validation_deferred": True}
                    with autocast_context(config):
                        out = model(
                            **to_device(batch, config.device),
                            loss_chunk_size=config.loss_chunk_size,
                            return_logits=False,
                        )
                    loss += float(out.ce_loss_sum)
                    targets += int(out.target_token_count)
                    inputs += int(out.input_token_count)
                    if targets >= config.eval_tokens_per_domain:
                        break
            finally:
                loader.close()
            if not targets or not math.isfinite(loss):
                raise ValueError(f"Invalid validation probe for {domain}")
            ce = loss / targets
            results[domain] = dict(
                ce=ce, perplexity=math.exp(min(ce, 700)), targets=targets, inputs=inputs
            )
    finally:
        model.train(was_training)
    weighted = sum(DOMAIN_WEIGHTS[d] * v["ce"] for d, v in results.items())
    return dict(
        validation_ce=weighted,
        validation_perplexity=math.exp(min(weighted, 700)),
        validation_domains=results,
        validation_weights=DOMAIN_WEIGHTS,
    )


def evaluate_checkpoint(
    checkpoint,
    token_manifest,
    screening_report,
    output,
    device="cuda",
    tokens_per_domain=1_000_000,
    batch_size=1,
):
    """Larger final holdout sweep, independently runnable after a daily pause."""
    from pathlib import Path
    from lamoe.model import LaMoEForCausalLM
    from lamoe.preprocessing.common import read_json, file_hash, atomic_json
    from .checkpoint import verify_checkpoint
    from .trainer import TrainConfig, autocast_context

    if tokens_per_domain < 1 or batch_size < 1:
        raise ValueError("Positive probe size and batch required")
    verify_checkpoint(checkpoint)
    progress = read_json(Path(checkpoint) / "progress.json")
    recipe = progress["recipe"]
    if recipe["data"]["manifest_sha256"] != file_hash(token_manifest):
        raise ValueError("Evaluation token corpus differs from checkpoint")
    if recipe["data"].get("screening_sha256") != file_hash(screening_report):
        raise ValueError("Evaluation exclusions differ from checkpoint")
    cfg = TrainConfig(
        token_manifest,
        "",
        screening_report=screening_report,
        device=device,
        precision="bf16" if device == "cuda" else "fp32",
        sequence_length=recipe["settings"]["sequence_length"],
        seed=recipe["settings"]["seed"],
        batch_size=batch_size,
        eval_tokens_per_domain=tokens_per_domain,
    )
    model = LaMoEForCausalLM.from_pretrained(checkpoint).to(device).eval()
    result = evaluate_domains(model, cfg, autocast_context)
    result.update(
        checkpoint=str(checkpoint),
        checkpoint_sha256=file_hash(Path(checkpoint) / "checkpoint.json"),
        screening_sha256=file_hash(screening_report),
        token_manifest_sha256=file_hash(token_manifest),
    )
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Final fixed-domain language-model evaluation; does not advance training"
    )
    for name in ("checkpoint", "token-manifest", "screening-report", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--tokens-per-domain", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=1)
    print(evaluate_checkpoint(**vars(p.parse_args())))
