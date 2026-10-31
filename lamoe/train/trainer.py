"""Single-device pretraining, bounded evaluation, and update-boundary resume."""

import json
import math
import random
import signal
import time
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import numpy as np

from lamoe.model import LaMoEConfig, LaMoEForCausalLM
from lamoe.preprocessing.common import atomic_json, stage_lock

from .checkpoint import (
    load_checkpoint,
    save_checkpoint,
    retain_checkpoints,
    verify_checkpoint,
)
from .data import TokenBatchLoader, target_count, to_device
from .optim import make_optimizer
from .schedule import wsd_multiplier
from .final_decay import (
    final_decay_multiplier,
    decay_spec,
    bind_decay_state,
    decay_done,
    midpoint_due,
    speed_stop_reason,
)
from .budget import SessionBudget
from .diagnostics import Diagnostics
from .evaluation import evaluate_domains
from .launch import prepare_plan, environment


@dataclass
class TrainConfig:
    token_manifest: str
    output: str
    max_steps: int = 1_000_000_000
    model_config: str | None = None
    tiny: bool = False
    plan: str | None = None
    resume: str | None = None
    sequence_length: int = 2048
    batch_size: int = 1
    accumulation_steps: int = 8
    optimizer: str = "muon_adamw"
    learning_rate: float = 3e-4
    muon_learning_rate: float = 3e-4
    muon_weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    device: str = "cuda"
    precision: str = "bf16"
    gdn_backend: str | None = None
    gradient_checkpointing: bool = False
    seed: int = 122209
    eval_every: int = 100
    eval_batches: int = 20
    save_every: int = 100
    log_every: int = 10
    max_runtime_hours: float | None = 6.0
    training_tokens: int | None = None
    screening_report: str | None = None
    kernel_report: str | None = None
    cu_per_hour: float = 7.0
    pretraining_cu: float = 225.0
    save_margin_seconds: float = 600.0
    save_every_minutes: float = 30.0
    min_free_gb: float = 10.0
    loss_chunk_size: int = 128
    eval_tokens_per_domain: int = 100_000
    eval_every_tokens: int = 50_000_000
    eval_every_hours: float = 1.0
    mla_backend: str | None = None
    moe_backend: str | None = None
    extend_completed_plan: bool = False
    final_decay_tokens: int = 0
    final_decay_start_tokens: int | None = None
    training_deadline_unix: float | None = None

    def validate(self):
        if type(self.final_decay_tokens) is not int or self.final_decay_tokens < 0:
            raise ValueError("Invalid final decay token length")
        if self.final_decay_tokens:
            if not self.resume or self.extend_completed_plan:
                raise ValueError(
                    "Final decay requires an original resume, not plan extension"
                )
            decay_spec(
                self.final_decay_start_tokens,
                self.final_decay_tokens,
                self.min_lr_ratio,
            )
            if self.training_deadline_unix is None or not math.isfinite(
                self.training_deadline_unix
            ):
                raise ValueError(
                    "Final decay requires the fixed allocation-relative training deadline"
                )
        elif (
            self.final_decay_start_tokens is not None
            or self.training_deadline_unix is not None
        ):
            raise ValueError("Decay controls supplied without enabling final decay")
        if self.extend_completed_plan and not self.resume:
            raise ValueError("Plan extension requires an existing completed checkpoint")
        for name in (
            "max_steps",
            "batch_size",
            "accumulation_steps",
            "eval_every",
            "eval_batches",
            "save_every",
            "log_every",
            "muon_ns_steps",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")
        if self.training_tokens is not None and (
            type(self.training_tokens) is not int or self.training_tokens < 2
        ):
            raise ValueError("training_tokens must be an integer >= 2")
        for name in (
            "cu_per_hour",
            "pretraining_cu",
            "save_margin_seconds",
            "save_every_minutes",
            "min_free_gb",
            "eval_every_hours",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        # Public release: the caller selects their own local accounting budget.
        for name in ("loss_chunk_size", "eval_tokens_per_domain", "eval_every_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.max_runtime_hours is not None
            and self.max_runtime_hours * 3600 <= self.save_margin_seconds
        ):
            raise ValueError("Session must be longer than its checkpoint reserve")
        for name in ("learning_rate", "muon_learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.muon_weight_decay) or self.muon_weight_decay < 0:
            raise ValueError("muon_weight_decay must be finite and nonnegative")
        if not 0 < self.muon_momentum < 1 or self.muon_ns_steps >= 100:
            raise ValueError("Muon requires 0 < momentum < 1 and 1 <= ns_steps < 100")
        if self.optimizer not in ("muon_adamw", "adamw"):
            raise ValueError("optimizer must be muon_adamw or adamw")
        if not 0 <= self.min_lr_ratio <= 1 or not 0 < self.beta2 < 1:
            raise ValueError("Invalid LR ratio or beta2")
        if self.max_runtime_hours is not None and (
            not math.isfinite(self.max_runtime_hours) or self.max_runtime_hours <= 0
        ):
            raise ValueError("max_runtime_hours must be positive and finite")
        if self.device not in ("cpu", "cuda") or self.precision not in ("fp32", "bf16"):
            raise ValueError("Use cpu/cuda and fp32/bf16")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU verification currently requires --precision fp32")
        if self.tiny and self.model_config:
            raise ValueError("Choose --tiny or --model-config, not both")


def autocast_context(config):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if config.precision == "bf16"
        else nullcontext()
    )


@torch.no_grad()
def evaluate(model, config):
    """Same deterministic held-out subset each evaluation; never consumes train data."""
    if not config.tiny:
        return evaluate_domains(model, config, autocast_context)
    loader = TokenBatchLoader(
        config.token_manifest,
        split="validation",
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        seed=config.seed + 1,
    )
    was_training = model.training
    model.eval()
    loss_sum, targets, inputs = 0.0, 0, 0
    try:
        for _, batch in zip(range(config.eval_batches), loader):
            with autocast_context(config):
                out = model(
                    **to_device(batch, config.device),
                    loss_chunk_size=config.loss_chunk_size,
                    return_logits=False,
                )
            loss_sum += float(out.ce_loss_sum)
            targets += int(out.target_token_count)
            inputs += int(out.input_token_count)
    finally:
        loader.close()
        model.train(was_training)
    if targets == 0 or not math.isfinite(loss_sum):
        raise ValueError("Validation has no supervised targets or nonfinite loss")
    ce = loss_sum / targets
    return {
        "validation_ce": ce,
        "validation_perplexity": math.exp(ce) if ce < 700 else None,
        "validation_targets": targets,
        "validation_inputs": inputs,
    }


def _append_metrics(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def train(config: TrainConfig):
    config.validate()
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA unavailable; use --device cpu --precision fp32 for checks"
            )
        if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    # Exact replay is tested on CPU. CUDA reductions/kernels can still vary.
    torch.backends.cudnn.benchmark = False
    root = Path(config.output)
    with stage_lock(root):
        if config.resume == "auto":
            from .checkpoint import latest_checkpoint

            config.resume = latest_checkpoint(root)
        prepare_plan(config, root)
        if (root / "run.json").exists() and not config.resume:
            raise ValueError(
                "Output already contains a run; use --resume or a new folder"
            )
        if list(root.glob("step-*")) and not config.resume:
            raise ValueError(
                "Output contains checkpoints; use --resume or a new folder"
            )
        if config.resume and (root / "latest.json").exists():
            latest = json.loads((root / "latest.json").read_text())["checkpoint"]
            if (root / latest).resolve() != Path(config.resume).resolve():
                raise ValueError(
                    "Use the latest checkpoint here, or a new output for a fork"
                )
        loader = TokenBatchLoader(
            config.token_manifest,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            seed=config.seed,
            plan_path=config.plan,
            screening_report=config.screening_report,
        )
        try:
            budget = SessionBudget(
                root,
                hours=config.max_runtime_hours or 6.0,
                cu_per_hour=config.cu_per_hour,
                cap=config.pretraining_cu,
                save_margin_seconds=config.save_margin_seconds,
            )
            try:
                if budget.remaining_seconds <= 0:
                    return dict(
                        status="paused",
                        reason="pretraining_budget_exhausted",
                        checkpoint=config.resume,
                        estimated_cu=budget.spent,
                    )
                return _run(config, loader, root, budget)
            finally:
                budget.close()
        finally:
            loader.close()


def _run(config, loader, root, budget):
    if config.model_config:
        model_config = LaMoEConfig.load(config.model_config)
    elif config.tiny:
        model_config = LaMoEConfig.tiny(
            vocab_size=loader.vocab_size,
            max_position_embeddings=max(64, config.sequence_length),
        )
    else:
        model_config = LaMoEConfig()
    if config.gdn_backend:
        model_config = LaMoEConfig(
            **{**model_config.to_dict(), "gdn_backend": config.gdn_backend}
        )
    overrides = {
        n: getattr(config, n)
        for n in ("mla_backend", "moe_backend")
        if getattr(config, n)
    }
    if overrides:
        model_config = LaMoEConfig(**{**model_config.to_dict(), **overrides})
    if model_config.vocab_size != loader.vocab_size:
        raise ValueError("Model vocabulary differs from token manifest")
    if config.sequence_length > model_config.max_position_embeddings:
        raise ValueError("Sequence length exceeds model context limit")
    if not loader.manifest["counts"].get("validation"):
        raise ValueError("Held-out validation split required")
    planned_tokens = sum(loader.recipe["quotas"].values())
    if (
        config.training_tokens is not None
        and config.training_tokens != planned_tokens
        and config.plan
    ):
        raise ValueError("training_tokens differs from the frozen mixture plan")
    planned_tokens = config.training_tokens or planned_tokens
    settings = asdict(config)
    for name in (
        "output",
        "resume",
        "token_manifest",
        "plan",
        "model_config",
        "tiny",
        "gdn_backend",
        "max_runtime_hours",
        "screening_report",
        "kernel_report",
        "cu_per_hour",
        "pretraining_cu",
        "save_margin_seconds",
        "save_every_minutes",
        "min_free_gb",
        "extend_completed_plan",
        "final_decay_tokens",
        "final_decay_start_tokens",
        "training_deadline_unix",
    ):
        settings.pop(name)
    recipe = json.loads(
        json.dumps(
            {
                "settings": settings,
                "model": model_config.to_dict(),
                "data": loader.recipe,
                "planned_tokens": planned_tokens,
                "schedule": "token-wsd-2-78-20-budget-cooldown-v1",
                "tokenizer": loader.manifest["tokenizer"],
            }
        )
    )
    if config.final_decay_tokens:
        recipe["schedule"] = decay_spec(
            config.final_decay_start_tokens,
            config.final_decay_tokens,
            config.min_lr_ratio,
        )
    model = LaMoEForCausalLM(model_config).to(config.device)
    if not config.tiny:
        from .launch import check_kernel_report

        recipe["kernel_acceptance_sha256"] = check_kernel_report(config, model_config)
    model.gradient_checkpointing = config.gradient_checkpointing
    optimizer = make_optimizer(
        model,
        config.learning_rate,
        config.weight_decay,
        config.beta2,
        optimizer=config.optimizer,
        muon_learning_rate=config.muon_learning_rate,
        muon_weight_decay=config.muon_weight_decay,
        muon_momentum=config.muon_momentum,
        muon_ns_steps=config.muon_ns_steps,
    )
    recipe["optimizer_layout"] = optimizer.layout
    state = {
        "step": 0,
        "input_tokens": 0,
        "target_tokens": 0,
        "best_validation_ce": None,
        "lr_progress": 0.0,
        "last_eval_tokens": 0,
        "last_eval_wall_seconds": 0.0,
        "wall_seconds": 0.0,
        "clipped_updates": 0,
    }
    if config.resume:
        state = load_checkpoint(
            config.resume,
            model,
            optimizer,
            loader,
            recipe,
            extend_completed_plan=config.extend_completed_plan,
            allow_final_decay=bool(config.final_decay_tokens),
        )
        if config.extend_completed_plan:
            prior = json.loads((Path(config.resume) / "progress.json").read_text())
            if prior["recipe"] != recipe:
                atomic_json(
                    root / "continuation-verified.json",
                    {
                        "passed": True,
                        "checkpoint": Path(config.resume).name,
                        "initial_state": state,
                        "previous_planned_tokens": prior["recipe"]["planned_tokens"],
                        "planned_tokens": planned_tokens,
                        "schedule": "preserve-completed-wsd-final-learning-rate",
                        "learning_rates": {
                            "muon": config.muon_learning_rate * config.min_lr_ratio,
                            "adamw": config.learning_rate * config.min_lr_ratio,
                        },
                        "loader_consumed": loader.state_dict()["consumed"],
                        "optimizer_rng_and_cursors_restored": True,
                    },
                )
    if config.final_decay_tokens:
        bind_decay_state(state, recipe["schedule"])
        multiplier = final_decay_multiplier(
            state["input_tokens"] - state["final_decay_start_tokens"],
            state["final_decay_tokens"],
            state["final_decay_start_multiplier"],
        )
        initial_rates = optimizer.set_lr_multiplier(multiplier)
        atomic_json(
            root / "continuation-verified.json",
            dict(
                passed=True,
                checkpoint=Path(config.resume).name,
                state=state,
                schedule=recipe["schedule"],
                learning_rates=initial_rates,
                loader_consumed=loader.state_dict()["consumed"],
                source_checkpoint_files_unchanged=True,
            ),
        )
    if state["step"] > config.max_steps:
        raise ValueError("Checkpoint step exceeds max_steps")
    atomic_json(
        root / "run.json",
        {
            "recipe": recipe,
            "parameters": model.parameter_summary(),
            "optimizer_groups": optimizer.summary(),
            "tokenizer": loader.manifest["tokenizer"],
            "torch_version": str(torch.__version__),
            "environment": environment(),
            "device_name": torch.cuda.get_device_name()
            if config.device == "cuda"
            else "cpu",
        },
    )
    print(
        json.dumps(dict(event="start", **state, parameters=model.parameter_summary())),
        flush=True,
    )
    started = time.monotonic()
    prior_wall = state["wall_seconds"]
    starting_tokens = state["input_tokens"]
    diagnostics = Diagnostics(model)
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    previous_handlers = {
        s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)
    }
    last_saved = Path(config.resume).resolve() if config.resume else None
    last_save_time = time.monotonic()
    checkpoint_seconds = 0.0

    def save():
        nonlocal last_saved, last_save_time, checkpoint_seconds
        expected = (root / f"step-{state['step']:08d}").resolve()
        if last_saved != expected:
            save_start = time.monotonic()
            last_saved = save_checkpoint(
                root, model, optimizer, loader, state, recipe
            ).resolve()
            verify_checkpoint(last_saved)
            retain_checkpoints(
                root, keep=3, protected_extra=[config.resume] if config.resume else []
            )
            checkpoint_seconds = time.monotonic() - save_start
            last_save_time = time.monotonic()
            _append_metrics(
                root / "metrics.jsonl",
                dict(
                    event="checkpoint", step=state["step"], seconds=checkpoint_seconds
                ),
            )

    def report(status, **extra):
        atomic_json(
            root / "status.json",
            dict(
                status=status,
                **state,
                checkpoint=str(last_saved) if last_saved else None,
                updated_unix=time.time(),
                estimated_cu=budget.spent,
                estimated_cu_remaining=max(0, budget.cap - budget.spent),
                **extra,
            ),
        )

    reason = "max_steps"
    try:
        model.train()
        report("running")
        last_update_seconds = 0.0
        while (
            state["step"] < config.max_steps
            and state["input_tokens"] < planned_tokens
            and (not config.final_decay_tokens or not decay_done(state))
        ):
            if (
                stop["requested"]
                or (root / "STOP").exists()
                or budget.remaining_seconds
                <= max(last_update_seconds * 1.5, checkpoint_seconds)
            ):
                reason = (
                    "pretraining_budget_exhausted"
                    if budget.progress >= 0.999
                    else "stop_requested_or_runtime_limit"
                )
                break
            if config.final_decay_tokens:
                remaining_tokens = (
                    state["final_decay_start_tokens"]
                    + state["final_decay_tokens"]
                    - state["input_tokens"]
                )
                remaining_time = min(
                    config.training_deadline_unix - time.time(),
                    budget.remaining_seconds,
                )
                reason_to_stop = speed_stop_reason(
                    time.monotonic() - started,
                    state["input_tokens"] - starting_tokens,
                    remaining_tokens,
                    remaining_time,
                )
                if reason_to_stop or remaining_time <= max(
                    last_update_seconds * 1.5, 1.0
                ):
                    reason = reason_to_stop or "session_training_deadline"
                    atomic_json(
                        root / "session-stop.json",
                        {
                            "reason": reason,
                            "input_tokens": state["input_tokens"],
                            "schedule_not_shortened": True,
                        },
                    )
                    break
            if shutil.disk_usage(root).free < config.min_free_gb * 1e9:
                reason = "low_disk"
                break
            window_start = time.monotonic()
            # Stage only this update's CPU batches, so exact denominator counts
            # are known before backward even for padding/partial final batches.
            window = []
            for _ in range(config.accumulation_steps):
                try:
                    window.append(next(loader))
                except StopIteration:
                    break
            if not window:
                reason = "data_exhausted"
                break
            total_targets = sum(target_count(b) for b in window)
            total_inputs = sum(int(b["attention_mask"].sum()) for b in window)
            if total_inputs == 0:
                raise ValueError("Accumulation window has no valid input tokens")
            if config.final_decay_tokens:
                into = (
                    state["input_tokens"]
                    + total_inputs
                    - state["final_decay_start_tokens"]
                )
                lr_multiplier = final_decay_multiplier(
                    into,
                    state["final_decay_tokens"],
                    state["final_decay_start_multiplier"],
                )
            else:
                # Original scheduling remains unchanged when final decay is disabled.
                progress = (state["input_tokens"] + total_inputs) / planned_tokens
                if budget.progress >= 0.8:
                    progress = max(progress, budget.progress)
                state["lr_progress"] = max(state["lr_progress"], min(1.0, progress))
                lr_multiplier = wsd_multiplier(
                    state["lr_progress"], config.min_lr_ratio
                )
            learning_rates = optimizer.set_lr_multiplier(lr_multiplier)
            optimizer.zero_grad(set_to_none=True)
            model.reset_router_counts()
            ce_sum, balance, router_z = 0.0, 0.0, 0.0
            gates = [0.0] * len(model_config.hybrid_layers)
            detailed = (state["step"] + 1) % config.log_every == 0 or state["step"] == 0
            diagnostics.start(detailed)
            data_wait_seconds = time.monotonic() - window_start
            for batch in window:
                diagnostics.set_domains(
                    batch["source_ids"], loader.names, loader.manifest["sources"]
                )
                with autocast_context(config):
                    out = model(
                        **to_device(batch, config.device),
                        loss_chunk_size=config.loss_chunk_size,
                        return_logits=False,
                    )
                    weight = int(out.input_token_count) / total_inputs
                    loss = out.ce_loss_sum / max(1, total_targets) + weight * (
                        model_config.load_balance_coef * out.balance_loss
                        + model_config.router_z_coef * out.router_z_loss
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Nonfinite loss; latest committed checkpoint remains valid"
                    )
                loss.backward()
                model.accumulate_router_counts(out)
                ce_sum += float(out.ce_loss_sum.detach())
                balance += float(out.balance_loss.detach()) * weight
                router_z += float(out.router_z_loss.detach()) * weight
                for i, gate in enumerate(out.fusion_gate_means):
                    gates[i] += float(gate.detach()) * weight
                del out, loss
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm, error_if_nonfinite=True
                )
            )
            usage = model.balance_counts.detach().cpu()
            usage = usage / usage.sum(dim=-1, keepdim=True).clamp_min(1)
            # Both optimizers skip grad=None parameters,
            # including their momentum and weight decay for that update.
            before = diagnostics.sample_weights(optimizer) if detailed else None
            if config.device == "cuda":
                torch.cuda.synchronize()
            opt_start = time.monotonic()
            optimizer.step()
            if config.device == "cuda":
                torch.cuda.synchronize()
            optimizer_seconds = time.monotonic() - opt_start
            detail = diagnostics.finish(optimizer, before) if detailed else {}
            # Validation can use a different batch size and domain layout. Do not
            # reuse the last training microbatch's masks in its forward hooks.
            diagnostics.start(False)
            model.reset_router_counts()
            state["step"] += 1
            state["input_tokens"] += total_inputs
            state["target_tokens"] += total_targets
            state["clipped_updates"] += int(grad_norm > config.max_grad_norm)
            state["wall_seconds"] = prior_wall + time.monotonic() - started
            elapsed = time.monotonic() - window_start
            last_update_seconds = elapsed
            metric = dict(
                event="train",
                **state,
                ce=ce_sum / total_targets if total_targets else None,
                balance_loss=balance,
                router_z_loss=router_z,
                gradient_norm=grad_norm,
                learning_rate=learning_rates.get("adamw"),
                learning_rates=learning_rates,
                input_tokens_per_second=total_inputs / elapsed,
                target_tokens_per_second=total_targets / elapsed,
                update_seconds=elapsed,
                fusion_gate_means=gates,
                expert_assignment_fractions=usage.tolist(),
                expert_max_mean_load=(
                    usage.max(-1).values * model_config.num_experts
                ).tolist(),
                expert_empty_counts=(usage == 0).sum(-1).tolist(),
                expert_entropy=(
                    -(usage * usage.clamp_min(1e-30).log()).sum(-1)
                ).tolist(),
                dropped_tokens=0,
                clipping_fraction=state["clipped_updates"] / state["step"],
                optimizer_seconds=optimizer_seconds,
                data_wait_seconds=data_wait_seconds,
                session_end_to_end_tokens_per_second=(
                    state["input_tokens"] - starting_tokens
                )
                / max(time.monotonic() - started, 1e-9),
                estimated_cu=budget.spent,
                peak_allocated_bytes=torch.cuda.max_memory_allocated()
                if config.device == "cuda"
                else 0,
                peak_reserved_bytes=torch.cuda.max_memory_reserved()
                if config.device == "cuda"
                else 0,
                diagnostics=detail,
            )
            if state["step"] % config.log_every == 0 or state["step"] == 1:
                _append_metrics(root / "metrics.jsonl", metric)
                print(json.dumps(metric), flush=True)
            if config.final_decay_tokens and midpoint_due(state):
                state["final_decay_midpoint_saved"] = True
                save()
                atomic_json(
                    root / "decay-midpoint.json",
                    {
                        "checkpoint": last_saved.name,
                        "input_tokens": state["input_tokens"],
                    },
                )
            if config.final_decay_tokens and decay_done(state):
                reason = "final_decay_complete"
                break  # Save immediately; enlarged comparison runs separately.
            final = loader.exhausted or state["input_tokens"] >= planned_tokens
            due = (
                state["step"] % config.eval_every == 0
                if config.tiny
                else state["input_tokens"] - state["last_eval_tokens"]
                >= config.eval_every_tokens
                and state["wall_seconds"] - state["last_eval_wall_seconds"]
                >= config.eval_every_hours * 3600
            )
            if (due or final) and budget.remaining_seconds > config.save_margin_seconds:
                eval_start = time.monotonic()
                validation = (
                    evaluate(model, config)
                    if config.tiny
                    else evaluate_domains(model, config, autocast_context, budget)
                )
                if validation.get("validation_deferred"):
                    save()
                    reason = "validation_deferred_session_end"
                    break
                ce = validation["validation_ce"]
                best = state["best_validation_ce"]
                state["best_validation_ce"] = ce if best is None else min(best, ce)
                state["last_eval_tokens"] = state["input_tokens"]
                state["last_eval_wall_seconds"] = (
                    prior_wall + time.monotonic() - started
                )
                metric = dict(
                    event="validation",
                    step=state["step"],
                    seconds=time.monotonic() - eval_start,
                    **validation,
                )
                _append_metrics(root / "metrics.jsonl", metric)
                print(json.dumps(metric), flush=True)
                if best is None or ce < best:
                    save()
                    atomic_json(
                        root / "best.json",
                        dict(checkpoint=last_saved.name, validation_ce=ce),
                    )
            if (
                state["step"] % config.save_every == 0
                or time.monotonic() - last_save_time >= config.save_every_minutes * 60
            ):
                save()
            report("running")
        save()
        complete = loader.exhausted or state["input_tokens"] >= planned_tokens
        if complete:
            reason = "token_plan_complete"
            atomic_json(
                root / "base-final.json",
                dict(
                    checkpoint=last_saved.name,
                    final_evaluation_done=state["last_eval_tokens"]
                    == state["input_tokens"],
                ),
            )
        if config.final_decay_tokens and decay_done(state):
            complete = True
            reason = "final_decay_complete"
            atomic_json(
                root / "decay-final.json",
                {
                    "checkpoint": last_saved.name,
                    "input_tokens": state["input_tokens"],
                    "full_pretraining_goal_complete": False,
                },
            )
        report("complete" if complete else "paused", reason=reason)
        return dict(**state, status="complete" if complete else "paused", reason=reason)
    except Exception as exc:
        # Never persist a partially applied optimizer update or advanced cursor
        # as resumable state. Resume the previous complete checkpoint instead.
        model.reset_router_counts()
        report("failed", error=type(exc).__name__)
        raise
    finally:
        diagnostics.close()
        _append_metrics(
            root / "metrics.jsonl",
            dict(
                event="session_end",
                seconds=budget.elapsed,
                session_input_tokens=state["input_tokens"] - starting_tokens,
                end_to_end_tokens_per_second=(state["input_tokens"] - starting_tokens)
                / max(budget.elapsed, 1e-9),
                estimated_cu=budget.spent,
                estimated_cu_per_hour=config.cu_per_hour,
            ),
        )
        for s, handler in previous_handlers.items():
            signal.signal(s, handler)
