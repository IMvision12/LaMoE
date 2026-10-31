"""One boundary contract for attention, recurrent memory, convolution, and loss."""

from dataclasses import dataclass

import torch

from .cache import ModelCache


@dataclass(frozen=True)
class SequenceLayout:
    valid: torch.Tensor
    segment_ids: torch.Tensor
    document_ids: torch.Tensor
    position_ids: torch.Tensor
    reset: torch.Tensor
    attention_bias: torch.Tensor | None
    is_causal: bool


def build_layout(
    input_ids,
    attention_mask=None,
    segment_ids=None,
    past: ModelCache | None = None,
    materialize_attention=True,
):
    if input_ids.ndim != 2 or input_ids.dtype != torch.long or not all(input_ids.shape):
        raise ValueError("input_ids must be a nonempty int64 [batch, length] tensor")
    batch, length = input_ids.shape
    device = input_ids.device
    if attention_mask is None:
        valid = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if attention_mask.shape != input_ids.shape or attention_mask.device != device:
            raise ValueError("attention_mask must match input_ids shape and device")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only zero and one")
        valid = attention_mask.bool()
    if past is not None and not valid.all():
        raise ValueError("Cached inference currently requires unpadded inputs")
    if segment_ids is None:
        segment_ids = (
            torch.zeros_like(input_ids)
            if past is None
            else past.last_segment_ids[:, None].expand_as(input_ids)
        )
    if (
        segment_ids.shape != input_ids.shape
        or segment_ids.device != device
        or segment_ids.dtype != torch.long
    ):
        raise ValueError(
            "segment_ids must be int64 and match input_ids shape and device"
        )
    if (segment_ids[valid] < 0).any():
        raise ValueError("Valid tokens require nonnegative segment IDs")

    reset = valid.clone()
    reset[:, 1:] &= (~valid[:, :-1]) | (segment_ids[:, 1:] != segment_ids[:, :-1])
    base_doc = torch.full((batch,), -1, device=device, dtype=torch.long)
    previous_position = torch.full_like(base_doc, -1)
    if past is not None:
        reset[:, 0] &= segment_ids[:, 0] != past.last_segment_ids
        base_doc = past.document_ids[:, -1]
        previous_position = past.last_position_ids
    docs = base_doc[:, None] + reset.long().cumsum(dim=1)
    docs = docs.masked_fill(~valid, -1)
    indices = torch.arange(length, device=device).expand(batch, -1)
    starts = torch.where(reset, indices, -previous_position[:, None] - 1)
    positions = (indices - starts.cummax(dim=1).values).masked_fill(~valid, 0)

    # Use SDPA's causal fast path only for square, unpadded, single-document input.
    simple = past is None and bool(valid.all()) and not bool(reset[:, 1:].any())
    mask = None
    if not simple and materialize_attention:
        past_len = 0 if past is None else past.length
        all_docs = docs if past is None else torch.cat((past.document_ids, docs), dim=1)
        keys = torch.arange(past_len + length, device=device)
        causal = keys[None, :] <= (past_len + indices[0])[:, None]
        mask = valid[:, :, None] & (docs[:, :, None] == all_docs[:, None, :]) & causal
        # SDPA True means a permitted key. Explicit offset avoids decode's upper-left trap.
        mask = mask[:, None, :, :]
    return SequenceLayout(valid, segment_ids, docs, positions, reset, mask, simple)
