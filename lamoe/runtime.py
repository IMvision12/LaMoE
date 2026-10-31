"""Local checkpoint loading and greedy decoding; no network or tool execution."""

from contextlib import nullcontext
from dataclasses import replace
import hashlib
from pathlib import Path


def load_local(checkpoint, device="cpu", gdn_backend="reference", expected_sha256=None):
    from safetensors.torch import load_model
    from .model import LaMoEConfig, LaMoEForCausalLM

    checkpoint = Path(checkpoint)
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise ValueError("A local model.safetensors checkpoint is required")
    if expected_sha256:
        with weights.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected_sha256:
                raise ValueError("Model weights do not match the expected SHA256")
    config = replace(
        LaMoEConfig.load(checkpoint / "config.json"),
        gdn_backend=gdn_backend,
        mla_backend="sdpa",
    )
    if str(device).startswith("cpu") and gdn_backend != "reference":
        raise ValueError("CPU inference requires the reference recurrence")
    model = LaMoEForCausalLM(config)
    load_model(model, str(weights), strict=True)
    return model.to(device).eval()


def autocast(device):
    import torch

    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else nullcontext()
    )


def greedy_ids(model, prefix, max_new_tokens, stop_ids=()):
    import torch

    if not prefix or type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("A nonempty prefix and positive token allowance are required")
    limit = model.config.max_position_embeddings
    if len(prefix) >= limit:
        raise ValueError(
            "Prompt exceeds the context allowance; it will not be truncated"
        )
    if any(type(t) is not int or not 0 <= t < model.config.vocab_size for t in prefix):
        raise ValueError("Prefix contains invalid token IDs")
    device = next(model.parameters()).device
    allowance = min(max_new_tokens, limit - len(prefix))
    current = torch.tensor([prefix], dtype=torch.long, device=device)
    cache, generated, stop = None, [], "token_limit"
    with torch.inference_mode(), autocast(device):
        for _ in range(allowance):
            out = model(
                current, past_key_values=cache, use_cache=True, logits_to_keep=1
            )
            token = int(out.logits[0, -1].argmax())
            generated.append(token)
            cache = out.past_key_values
            if token in stop_ids:
                stop = "stop_token"
                break
            current = torch.tensor([[token]], dtype=torch.long, device=device)
    if stop == "token_limit" and allowance < max_new_tokens:
        stop = "context_limit"
    return generated, stop
