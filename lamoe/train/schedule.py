"""Token-based warmup/stable/cosine cooldown, retained across daily sessions."""

import math


def wsd_multiplier(progress, minimum=0.1):
    p = min(1.0, max(0.0, progress))
    if p < 0.02:
        return p / 0.02
    if p < 0.8:
        return 1.0
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * (p - 0.8) / 0.2)) / 2
