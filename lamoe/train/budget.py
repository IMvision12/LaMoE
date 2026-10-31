"""Conservative estimated CU ledger. Never claims to read Colab's live balance."""

import time
import math
from pathlib import Path
from lamoe.preprocessing.common import atomic_json, read_json


class SessionBudget:
    def __init__(self, root, *, hours, cu_per_hour, cap, save_margin_seconds):
        self.path = Path(root) / "budget.json"
        self.rate, self.cap = cu_per_hour, cap
        previous = read_json(self.path) if self.path.exists() else {}
        # An interrupted session's whole lease remains charged. This may
        # overestimate consumption, but cannot silently refund lost GPU time.
        self.prior = previous.get("charged_cu", 0.0)
        if not math.isfinite(self.prior) or self.prior < 0:
            raise ValueError("Invalid saved credit ledger")
        self.hours = min(hours, max(0, cap - self.prior) / cu_per_hour)
        self.started = time.monotonic()
        self.margin = save_margin_seconds
        self.write(self.prior + self.hours * self.rate, active=True)

    def write(self, charged, active):
        atomic_json(
            self.path,
            dict(
                charged_cu=charged,
                active_lease=active,
                cu_per_hour=self.rate,
                cap_cu=self.cap,
                updated_unix=time.time(),
                accounting="estimated; crashed leases charged in full",
            ),
        )

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    @property
    def spent(self):
        return self.prior + self.elapsed * self.rate / 3600

    @property
    def remaining_seconds(self):
        return max(0, self.hours * 3600 - self.elapsed - self.margin)

    @property
    def progress(self):
        usable = max(1e-9, self.cap - self.margin * self.rate / 3600)
        return min(1, self.spent / usable)

    def close(self):
        self.write(self.spent, active=False)
