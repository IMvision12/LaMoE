"""Local teacher-forced continuation scoring, without truncation."""


def encode_pair(tokenizer, context, continuation):
    # Match the harness convention: move trailing context spaces to the answer.
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation, context = context[-spaces:] + continuation, context[:-spaces]
    if not continuation:
        raise ValueError("An empty continuation has no scored target")
    if not context:
        bos = tokenizer.token_to_id("<|bos|>")
        if bos is None:
            raise ValueError("Tokenizer requires the native BOS token")
        return [bos], tokenizer.encode(continuation, add_special_tokens=False).ids
    prefix = tokenizer.encode(context, add_special_tokens=False).ids
    whole = tokenizer.encode(context + continuation, add_special_tokens=False).ids
    bos = tokenizer.token_to_id("<|bos|>")
    if bos is None:
        raise ValueError("Tokenizer requires the native BOS token")
    return [bos] + prefix, whole[len(prefix) :]


def continuation_score(model, context_ids, continuation_ids):
    import torch
    from lamoe.runtime import autocast

    if not context_ids or not continuation_ids:
        raise ValueError("Nonempty context and continuation token sequences required")
    ids = context_ids + continuation_ids
    if len(ids) > model.config.max_position_embeddings:
        raise ValueError(
            "Scoring request exceeds the context limit; no truncation is applied"
        )
    if any(type(t) is not int or not 0 <= t < model.config.vocab_size for t in ids):
        raise ValueError("Invalid token ID")
    device = next(model.parameters()).device
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode(), autocast(device):
        logits = model(tensor).logits[0, len(context_ids) - 1 : -1].float()
        targets = tensor[0, len(context_ids) :]
        logprob = logits.log_softmax(-1).gather(-1, targets[:, None]).sum().item()
        greedy = bool(torch.equal(logits.argmax(-1), targets))
    return logprob, greedy
