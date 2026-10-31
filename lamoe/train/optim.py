"""Dossier RMS-Muon + AdamW with explicit, checkpointed parameter ownership."""

import math

import torch


class RMSMuon(torch.optim.Optimizer):
    """v0.4 unnormalized momentum and RMS-matched NS5; FP32 momentum.

    Unlike native Muon's EMA momentum, this follows the dossier's M=mu*M+g.
    Each expert/projection remains a separate matrix. grad=None skips all state.
    """

    def __init__(self, params, lr=3e-4, weight_decay=0.1, momentum=0.95, ns_steps=5):
        super().__init__(
            params,
            dict(
                lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("RMSMuon does not accept closures")
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2 or p.dtype != torch.float32 or p.grad.is_sparse:
                    raise ValueError("RMSMuon requires dense FP32 master matrices")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                m = state["momentum_buffer"]
                m.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(m, alpha=group["momentum"])
                transposed = g.shape[0] > g.shape[1]
                x = g.T if transposed else g
                x = (x / x.norm().clamp_min(1e-7)).to(torch.bfloat16)
                for _ in range(group["ns_steps"]):
                    a = x @ x.T
                    x = 3.4445 * x + (-4.775 * a + 2.0315 * (a @ a)) @ x
                if transposed:
                    x = x.T
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(x.float(), alpha=-group["lr"] * 0.2 * math.sqrt(max(p.shape)))


class OptimizerBundle:
    """Two disjoint optimizers, stepped once per accumulated update.

    Not a torch.optim.Optimizer subclass: the trainer uses a stateless LR
    multiplier instead of attaching a PyTorch scheduler. RMSMuon and AdamW
    both skip parameters whose grad is None, including momentum and decay.
    """

    def __init__(self, optimizers, layout):
        self.optimizers = optimizers
        self.layout = layout

    @property
    def param_groups(self):
        return [group for opt in self.optimizers.values() for group in opt.param_groups]

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def set_lr_multiplier(self, multiplier):
        if not math.isfinite(multiplier) or not 0 <= multiplier <= 1:
            raise ValueError("LR multiplier must be finite and between zero and one")
        for group in self.param_groups:
            group["lr"] = group["peak_lr"] * multiplier
        return {
            name: opt.param_groups[0]["lr"] for name, opt in self.optimizers.items()
        }

    def step(self):
        # Caller checks/clips all gradients before either optimizer updates.
        for opt in self.optimizers.values():
            opt.step()

    def state_dict(self):
        return {
            "format": "lamoe-optimizer-bundle-v1",
            "layout": self.layout,
            "optimizers": {
                name: opt.state_dict() for name, opt in self.optimizers.items()
            },
        }

    def load_state_dict(self, state):
        if state.get("format") != "lamoe-optimizer-bundle-v1":
            raise ValueError(
                "Legacy optimizer checkpoint; cannot silently convert to Muon + AdamW"
            )
        if state["layout"] != self.layout or set(state["optimizers"]) != set(
            self.optimizers
        ):
            raise ValueError("Optimizer parameter ownership or configuration changed")
        # Validate ordered names/shapes before native loaders map states by index.
        for name, opt in self.optimizers.items():
            opt.load_state_dict(state["optimizers"][name])

    def summary(self):
        return [
            {
                "group": group["group"],
                "optimizer": group["optimizer"],
                "tensors": len(group["parameters"]),
                "parameters": sum(math.prod(p["shape"]) for p in group["parameters"]),
                "peak_lr": group["peak_lr"],
                "weight_decay": group["weight_decay"],
            }
            for group in self.layout["groups"]
        ]


def make_optimizer(
    model,
    learning_rate,
    weight_decay,
    beta2=0.95,
    *,
    optimizer="muon_adamw",
    muon_learning_rate=3e-4,
    muon_weight_decay=0.1,
    muon_momentum=0.95,
    muon_ns_steps=5,
):
    """Use the dossier's RMS-matched Muon scaling and momentum convention.

    AdamW retains sensitive routing/dynamics gates even when they are 2D.
    Muon handles hidden nn.Linear weights only; convolutions are not flattened.
    The tied embedding/output matrix is optimized once, by AdamW.
    """
    if optimizer not in ("muon_adamw", "adamw"):
        raise ValueError("optimizer must be muon_adamw or adamw")
    for value in (learning_rate, muon_learning_rate):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Optimizer learning rates must be positive and finite")
    for value in (weight_decay, muon_weight_decay):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Optimizer weight decay must be nonnegative and finite")
    if not 0 < beta2 < 1 or not 0 < muon_momentum < 1:
        raise ValueError("Invalid beta2 or Muon momentum")
    if type(muon_ns_steps) is not int or not 1 <= muon_ns_steps < 100:
        raise ValueError("muon_ns_steps must be an integer in [1, 99]")
    modules = dict(model.named_modules())
    embedding_ids = {
        id(p)
        for m in modules.values()
        if isinstance(m, torch.nn.Embedding)
        for p in m.parameters(recurse=False)
    }
    buckets = {"muon_hidden": [], "adamw_decay": [], "adamw_no_decay": []}
    for name, param in model.named_parameters():  # Deduplicates tied weights.
        if not param.requires_grad:
            continue
        parent, _, leaf = name.rpartition(".")
        gate = (
            "router" in name
            or "head_gate" in name
            or parent.endswith((".gdn.a_proj", ".gdn.beta_proj"))
        )
        hidden = (
            optimizer == "muon_adamw"
            and param.ndim == 2
            and leaf == "weight"
            and isinstance(modules.get(parent), torch.nn.Linear)
            and id(param) not in embedding_ids
            and parent != "lm_head"
            and not gate
        )
        bucket = (
            "muon_hidden"
            if hidden
            else ("adamw_no_decay" if param.ndim < 2 or gate else "adamw_decay")
        )
        buckets[bucket].append((name, param))
    assigned = [id(p) for values in buckets.values() for _, p in values]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError(
            "Each trainable parameter must belong to exactly one optimizer"
        )
    if not assigned:
        raise ValueError("Model has no trainable parameters")
    optimizers, layout, adamw_groups = {}, [], []
    for bucket, pairs in buckets.items():
        if not pairs:
            continue
        muon = bucket == "muon_hidden"
        peak = muon_learning_rate if muon else learning_rate
        decay = (
            muon_weight_decay
            if muon
            else (weight_decay if bucket == "adamw_decay" else 0.0)
        )
        group = {
            "params": [p for _, p in pairs],
            "param_names": [n for n, _ in pairs],
            "lr": peak,
            "peak_lr": peak,
            "weight_decay": decay,
        }
        layout.append(
            {
                "group": bucket,
                "optimizer": "muon" if muon else "adamw",
                "peak_lr": peak,
                "weight_decay": decay,
                "parameters": [{"name": n, "shape": list(p.shape)} for n, p in pairs],
            }
        )
        if muon:
            optimizers["muon"] = RMSMuon(
                [group],
                lr=peak,
                weight_decay=decay,
                momentum=muon_momentum,
                ns_steps=muon_ns_steps,
            )
        else:
            adamw_groups.append(group)
    if optimizer == "muon_adamw" and "muon" not in optimizers:
        raise ValueError("No eligible hidden matrices found for Muon")
    if adamw_groups:
        optimizers["adamw"] = torch.optim.AdamW(
            adamw_groups,
            lr=learning_rate,
            betas=(0.9, beta2),
            eps=1e-8,
            fused=next(model.parameters()).is_cuda,
        )
    return OptimizerBundle(
        optimizers,
        {
            "selection_policy": "lamoe-v04-rms-muon-v2",
            "mode": optimizer,
            "groups": layout,
            "adamw_betas": [0.9, beta2],
            "adamw_eps": 1e-8,
            "muon_momentum": muon_momentum,
            "muon_ns_steps": muon_ns_steps,
            "muon_nesterov": True,
            "muon_adjust_lr_fn": "match_rms_adamw",
            "muon_momentum_convention": "unnormalized-nesterov",
            "muon_eps": 1e-7,
            "muon_ns_coefficients": [3.4445, -4.775, 2.0315],
        },
    )


def learning_rate_at(step, *, peak, minimum_ratio, warmup_steps, max_steps):
    """step is zero-based: warmup's first update uses peak/warmup_steps."""
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    span = max_steps - warmup_steps - 1
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, span)))
    return peak * (
        minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    )
