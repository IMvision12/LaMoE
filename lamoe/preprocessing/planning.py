"""Local, source-agnostic token allocation. No collection registry is imported."""


def plan_tokens(counts, total):
    """Allocate a single pass proportionally to supplied training token counts.

    This public default is not the private historical training recipe. Use an
    explicit reviewed plan when reproducing a particular corpus distribution.
    """
    if not isinstance(counts, dict) or not counts:
        raise ValueError("Supply named training token counts")
    if any(not isinstance(k, str) or not k for k in counts):
        raise ValueError("Sources must have nonempty names")
    if any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("Counts must be nonnegative integers")
    if type(total) is not int or total <= 0:
        raise ValueError("Training token budget must be a positive integer")
    available = sum(counts.values())
    if total > available:
        raise ValueError(
            "Budget exceeds unique training tokens; supply an explicit reviewed repetition plan"
        )
    allocation = {name: total * n // available for name, n in counts.items()}
    residual = total - sum(allocation.values())
    order = sorted(counts, key=lambda k: (-(total * counts[k] % available), k))
    for name in order[:residual]:
        allocation[name] += 1
    assert sum(allocation.values()) == total
    return {
        "training_tokens": total,
        "counts_are": "post-processing training-split tokens",
        "allocation": "public proportional single-pass v1",
        "source_plan": {
            name: {
                "available_unique_tokens": counts[name],
                "sampled_tokens": n,
                "training_weight": n / total,
                "effective_passes": n / counts[name] if counts[name] else 0,
            }
            for name, n in sorted(allocation.items())
        },
        "notes": [
            "No private source registry or acquisition recipe is distributed.",
            "Validation tokens never contribute to the training budget.",
        ],
    }
