"""On-host kernel parity and full-update profiling, not a convergence benchmark.

Run on the intended A100 before selecting the token horizon. No main training
checkpoint is touched. Output includes environment, tested dimensions, errors,
peak memory and optimizer-inclusive throughput. Does not certify other versions.
"""

import argparse
import time
import torch
from lamoe.model import LaMoEConfig, LaMoEForCausalLM
from lamoe.preprocessing.common import atomic_json
from .launch import environment
from .optim import make_optimizer


def run(
    output,
    mla_backend="flash_varlen",
    moe_backend="batched",
    steps=5,
    sequence_length=128,
    batch_size=1,
    full=False,
    gradient_checkpointing=False,
):
    if not torch.cuda.is_available():
        raise ValueError("Kernel acceptance requires the intended CUDA host")
    if steps < 2 or sequence_length < 8 or batch_size < 1:
        raise ValueError("Use >=2 steps, length>=8 and batch>=1")
    device = "cuda"
    torch.manual_seed(123)
    cfg = LaMoEConfig.tiny(
        hidden_size=64,
        mla_heads=2,
        qk_content_dim=48,
        qk_rope_dim=16,
        value_head_dim=64,
        gdn_key_dim=64,
        gdn_value_dim=64,
        max_position_embeddings=max(64, sequence_length),
    )
    ref = LaMoEForCausalLM(cfg).to(device)
    fast = LaMoEForCausalLM(
        LaMoEConfig(
            **{
                **cfg.to_dict(),
                "mla_backend": mla_backend,
                "gdn_backend": "fla",
                "moe_backend": moe_backend,
            }
        )
    ).to(device)
    fast.load_state_dict(ref.state_dict())
    ids = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    seg = torch.zeros_like(ids)
    seg[:, 7:] = 1
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[1, 13:] = False
    for m in (ref, fast):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(ids, labels=ids, attention_mask=mask, segment_ids=seg)
        out.loss.backward()
        if m is ref:
            reference = out.logits.detach()
        else:
            torch.testing.assert_close(out.logits, reference, rtol=0.03, atol=0.015)
    worst = 0.0
    for (name, p), (_, q) in zip(ref.named_parameters(), fast.named_parameters()):
        if p.grad is None or q.grad is None:
            if (p.grad is None) != (q.grad is None):
                raise AssertionError("Missing gradient: " + name)
            continue
        if not torch.isfinite(q.grad).all():
            raise AssertionError("Nonfinite gradients: " + name)
        torch.testing.assert_close(p.grad, q.grad, rtol=0.08, atol=0.003)
        worst = max(worst, float((p.grad - q.grad).abs().max()))
    # Packed-document isolation and future causality for optimized kernels.
    fast.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline = fast(ids, attention_mask=mask, segment_ids=seg).logits
        altered = ids.clone()
        altered[:, :7] = (altered[:, :7] + 1) % cfg.vocab_size
        changed = fast(altered, attention_mask=mask, segment_ids=seg).logits
        torch.testing.assert_close(baseline[:, 7:], changed[:, 7:], rtol=0, atol=0)
        altered = ids.clone()
        altered[:, 12:] = (altered[:, 12:] + 1) % cfg.vocab_size
        changed = fast(altered, attention_mask=mask, segment_ids=seg).logits
        torch.testing.assert_close(baseline[:, :12], changed[:, :12], rtol=0, atol=0)
        decode_ids = ids[:1, :10]
        expected = fast(decode_ids).logits
        cache = None
        pieces = []
        for t in range(decode_ids.shape[1]):
            out = fast(decode_ids[:, t : t + 1], use_cache=True, past_key_values=cache)
            cache = out.past_key_values
            pieces.append(out.logits)
        torch.testing.assert_close(
            torch.cat(pieces, dim=1), expected, rtol=0.03, atol=0.015
        )
    del ref, fast, out
    torch.cuda.empty_cache()
    config = (
        LaMoEConfig(mla_backend=mla_backend, moe_backend=moe_backend, gdn_backend="fla")
        if full
        else LaMoEConfig(
            **{
                **cfg.to_dict(),
                "mla_backend": mla_backend,
                "moe_backend": moe_backend,
                "gdn_backend": "fla",
            }
        )
    )
    model = LaMoEForCausalLM(config).to(device)
    model.gradient_checkpointing = gradient_checkpointing
    opt = make_optimizer(model, 3e-4, 0.1)
    ids = torch.randint(
        0, config.vocab_size, (batch_size, sequence_length), device=device
    )
    segment = torch.arange(sequence_length, device=device)[None, :].expand_as(
        ids
    ) // max(1, sequence_length // 3)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(steps + 1):
        torch.cuda.synchronize()
        start = time.monotonic()
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                ids,
                labels=ids,
                segment_ids=segment,
                loss_chunk_size=128,
                return_logits=False,
            )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        if i:
            times.append(time.monotonic() - start)
    report = dict(
        passed=True,
        environment=environment(),
        gpu=torch.cuda.get_device_name(),
        kernel_gradient_max_abs_error=worst,
        mla_backend=mla_backend,
        moe_backend=moe_backend,
        gdn_backend="fla",
        full_model_profile=full,
        gradient_checkpointing=gradient_checkpointing,
        sequence_length=sequence_length,
        batch_size=batch_size,
        optimizer="rms-muon-v04",
        model=config.to_dict(),
        step_seconds=sum(times) / len(times),
        tokens_per_second=ids.numel() * len(times) / sum(times),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        limitation="Synthetic profile excludes real-data I/O, evaluation and checkpoint cost; no learning claim.",
    )
    atomic_json(output, report)
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--mla-backend", choices=("sdpa", "flash_varlen"), default="flash_varlen"
    )
    p.add_argument("--moe-backend", choices=("reference", "batched"), default="batched")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--full", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    run(**vars(p.parse_args()))
