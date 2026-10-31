"""Bounded, opt-in sampled update diagnostics; no extra model-size snapshot."""

import math
import torch


def distribution(value):
    x = value.detach().float().flatten()
    if not x.numel():
        return {}
    return dict(
        mean=float(x.mean()),
        minimum=float(x.min()),
        maximum=float(x.max()),
        low_fraction=float((x < 0.01).float().mean()),
        high_fraction=float((x > 0.99).float().mean()),
        histogram=torch.histc(x, bins=10, min=0, max=1).cpu().tolist(),
    )


class Diagnostics:
    def __init__(self, model):
        self.model, self.handles, self.values = model, [], {}
        self.enabled = False
        self.domains = {}
        for i, layer in enumerate(model.layers):
            for name, branch in (("mla", layer.mla), ("gdn", layer.gdn)):
                if branch is not None:
                    self.handles.append(
                        branch.register_forward_hook(self.hook(f"layer_{i + 1}/{name}"))
                    )
            if hasattr(layer.ffn, "experts"):
                self.handles.append(
                    layer.ffn.register_forward_hook(
                        self.expert_hook(f"layer_{i + 1}/experts")
                    )
                )

    def set_domains(self, source_ids, names, mapping):
        if self.enabled:
            self.domains = {
                d: sum(
                    (source_ids == i).long()
                    for i, n in enumerate(names)
                    if mapping[n] == d
                ).bool()
                for d in set(mapping[n] for n in names)
            }

    def expert_hook(self, name):
        def capture(module, inputs, output):
            if not self.enabled:
                return
            x, valid = inputs
            with (
                torch.no_grad(),
                torch.autocast(device_type=x.device.type, enabled=False),
            ):
                choices = (
                    module.router(x.float())
                    .topk(module.config.experts_per_token, dim=-1)
                    .indices
                )
                self.values[name + "/by_domain"] = {
                    d: torch.bincount(
                        choices[mask.to(x.device) & valid].flatten(),
                        minlength=module.config.num_experts,
                    )
                    .cpu()
                    .tolist()
                    for d, mask in self.domains.items()
                }

        return capture

    def hook(self, name):
        def capture(module, inputs, output):
            if not self.enabled:
                return
            value = output[0]
            valid = inputs[1].valid
            selected = value.detach().float()[valid]
            self.values[name + "/output_rms"] = (
                float(selected.square().mean().sqrt()) if selected.numel() else 0
            )
            self.values[name + "/domain_rms"] = {
                d: float(
                    value.detach()
                    .float()[mask.to(value.device) & valid]
                    .square()
                    .mean()
                    .sqrt()
                )
                for d, mask in self.domains.items()
                if bool((mask.to(value.device) & valid).any())
            }
            if value.requires_grad:

                def gradient(g):
                    self.values[name + "/output_gradient_rms"] = (
                        float(g.detach().float()[valid].square().mean().sqrt())
                        if bool(valid.any())
                        else 0
                    )

                value.register_hook(gradient)
            if name.endswith("/mla") and module.head_gate is not None:
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=value.device.type, enabled=False),
                ):
                    gate = module.head_gate(inputs[0].float()).sigmoid()[valid]
                self.values[name + "/head_gate"] = distribution(gate)

        return capture

    def start(self, enabled):
        self.enabled = enabled
        self.values = {}
        for layer in self.model.layers:
            layer.collect_diagnostics = enabled

    def sample_weights(self, optimizer):
        # First 1024 entries per tensor: explicitly sampled, not exact full RMS.
        return [
            [p.detach().flatten()[:1024].clone() for p in g["params"]]
            for g in optimizer.param_groups
        ]

    def finish(self, optimizer, before):
        groups = []
        for group, old in zip(optimizer.param_groups, before):
            delta = weight = n = 0
            for p, previous in zip(group["params"], old):
                now = p.detach().flatten()[:1024]
                delta += float((now - previous).square().sum())
                weight += float(previous.square().sum())
                n += now.numel()
            groups.append(
                dict(
                    sampled_elements=n,
                    update_rms=math.sqrt(delta / max(1, n)),
                    weight_rms=math.sqrt(weight / max(1, n)),
                    update_weight_ratio=math.sqrt(delta / max(weight, 1e-30)),
                )
            )
        for layer in self.model.layers:
            if getattr(layer, "diagnostics", None):
                self.values[f"layer_{layer.layer_number}/fusion"] = layer.diagnostics
        return dict(
            last_microbatch_branches=self.values, sampled_optimizer_groups=groups
        )

    def close(self):
        for handle in self.handles:
            handle.remove()
