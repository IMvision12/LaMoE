"""Opt-in, token-driven final decay and fail-closed resume/speed guards."""

import math

LEGACY_SCHEDULE = "token-wsd-2-78-20-budget-cooldown-v1"


def final_decay_multiplier(tokens_into_decay, decay_tokens, start=0.1):
    if type(decay_tokens) is not int or decay_tokens <= 0:
        raise ValueError("A final decay needs a positive integer token length")
    if not math.isfinite(start) or not 0 < start <= 1:
        raise ValueError("Invalid starting multiplier")
    p = min(1.0, max(0.0, tokens_into_decay / decay_tokens))
    return start * (1.0 - p)


def decay_spec(start, length, multiplier):
    if type(start) is not int or start < 0:
        raise ValueError("Pin the initial checkpoint's exact input token count")
    final_decay_multiplier(0, length, multiplier)
    return {
        "kind": "final-linear-token-decay-v1",
        "start_input_tokens": start,
        "end_input_tokens": start + length,
        "decay_input_tokens": length,
        "start_multiplier": multiplier,
        "rounding": "exact_input_token_endpoint",
    }


def validate_recipe_transition(progress, recipe):
    """Allow only a one-time legacy-schedule -> explicit decay transition."""
    from .continuation import validate_transition
    from lamoe.preprocessing.common import read_json
    from pathlib import Path

    contract_path = Path(__file__).with_name("continuation_contract.json")
    if not contract_path.is_file():
        raise ValueError(
            "Historical recipe transitions require the original private continuation contract; fresh public training does not use this path"
        )
    contract = read_json(contract_path)
    return validate_transition(progress, recipe, contract)


def bind_decay_state(state, spec):
    expected = {
        "final_decay_start_tokens": spec["start_input_tokens"],
        "final_decay_end_tokens": spec["end_input_tokens"],
        "final_decay_tokens": spec["decay_input_tokens"],
        "final_decay_start_multiplier": spec["start_multiplier"],
    }
    present = [key in state for key in expected]
    if any(present):
        if not all(present) or any(
            state[key] != value for key, value in expected.items()
        ):
            raise ValueError("Saved decay window changed or is incomplete")
    elif state["input_tokens"] != expected["final_decay_start_tokens"]:
        raise ValueError("Cannot initialize decay after its pinned starting checkpoint")
    state.update(expected)
    state.setdefault("final_decay_midpoint_saved", False)
    if state["input_tokens"] < spec["start_input_tokens"]:
        raise ValueError("Checkpoint precedes the decay start")


def decay_done(state):
    return (
        state["input_tokens"]
        >= state["final_decay_start_tokens"] + state["final_decay_tokens"]
    )


def midpoint_due(state):
    return (
        not state["final_decay_midpoint_saved"]
        and 2 * (state["input_tokens"] - state["final_decay_start_tokens"])
        >= state["final_decay_tokens"]
    )


def speed_stop_reason(
    elapsed,
    added_tokens,
    remaining_tokens,
    remaining_seconds,
    check_after=900.0,
    minimum_tps=38000.0,
):
    if remaining_tokens <= 0:
        return None
    if remaining_seconds <= 0:
        return "session_training_deadline"
    # The global decay spans sessions; remaining global tokens need not fit this allocation.
    return None
