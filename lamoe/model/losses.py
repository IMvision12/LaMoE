"""Next-token CE and router auxiliary losses with explicit reduction counts."""

import torch
from torch.nn import functional as F


def chunked_linear_loss(hidden, weight, labels, layout, chunk_size):
    """Exact CE with checkpointed token chunks: never retain B*T*V logits."""
    from torch.utils.checkpoint import checkpoint

    if (
        labels.shape != hidden.shape[:2]
        or labels.dtype != torch.long
        or labels.device != hidden.device
    ):
        raise ValueError("labels must match hidden states and use int64")
    eligible = layout.valid[:, :-1] & layout.valid[:, 1:]
    eligible &= layout.document_ids[:, :-1] == layout.document_ids[:, 1:]
    eligible &= labels[:, 1:] != -100
    targets = labels[:, 1:].masked_fill(~eligible, -100).reshape(-1)
    x = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    total = hidden.float().sum() * 0

    def chunk(a, w, y):
        return F.cross_entropy(
            F.linear(a, w).float(), y, ignore_index=-100, reduction="sum"
        )

    for start in range(0, len(targets), chunk_size):
        args = (
            x[start : start + chunk_size],
            weight,
            targets[start : start + chunk_size],
        )
        total = total + (
            checkpoint(chunk, *args, use_reentrant=False)
            if torch.is_grad_enabled()
            else chunk(*args)
        )
    count = eligible.sum()
    return total / count.clamp_min(1), total, count


def causal_language_model_loss(logits, labels, layout):
    """Labels are unshifted; -100 masks prompts or other unsupervised targets."""
    if (
        labels.shape != logits.shape[:2]
        or labels.dtype != torch.long
        or labels.device != logits.device
    ):
        raise ValueError("labels must be int64 and match input_ids shape and device")
    targets = labels[:, 1:].clone()
    eligible = layout.valid[:, :-1] & layout.valid[:, 1:]
    eligible &= layout.document_ids[:, :-1] == layout.document_ids[:, 1:]
    eligible &= targets != -100
    targets.masked_fill_(~eligible, -100)
    count = eligible.sum()
    # reduction='sum' remains finite even if every target is masked or T == 1.
    loss_sum = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum / count.clamp_min(1), loss_sum, count


def router_losses(statistics, previous_counts, zero):
    balances, z_losses = [], []
    for index, stats in enumerate(statistics):
        # Clone so an explicit accumulation update before backward cannot mutate saved tensors.
        counts = (
            previous_counts[index].detach().float().clone()
            + stats.assignment_counts.detach()
        )
        fractions = counts / counts.sum().clamp_min(1)
        probabilities = stats.probability_sum / max(stats.token_count, 1)
        balances.append(probabilities.numel() * (fractions * probabilities).sum())
        z_losses.append(stats.z_loss_sum / max(stats.token_count, 1))
    if not balances:
        return zero, zero
    return torch.stack(balances).mean(), torch.stack(z_losses).mean()
