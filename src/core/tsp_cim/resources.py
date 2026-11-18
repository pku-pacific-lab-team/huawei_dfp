"""Deterministic fluid reservations for the analytical hemisphere schedule.

Rates are exact rational bytes/cycle. This is not a bank or stream-slot simulator.
"""
from fractions import Fraction
from math import ceil


class ResourceCalendar:
    def __init__(self, capacities):
        self.capacities = {key: Fraction(str(value)) for key, value in capacities.items()}
        self.intervals = []

    def available(self, at):
        available = dict(self.capacities)
        for start, end, rates, _ in self.intervals:
            if start <= at < end:
                for key, rate in rates.items():
                    available[key] -= rate
        return available

    def next_boundary(self, at):
        return min((t for start, end, _, _ in self.intervals for t in (start, end) if t > at), default=None)

    def reserve(self, start, end, rates, label):
        if end <= start:
            return
        rates = {k: Fraction(v) for k, v in rates.items() if v}
        boundaries = {start, *(t for a, b, _, _ in self.intervals for t in (a, b) if start < t < end)}
        for at in boundaries:
            available = self.available(at)
            if any(rate < 0 or rate > available[key] for key, rate in rates.items()):
                raise ValueError("analytical resource reservation exceeds capacity")
        if self.intervals and self.intervals[-1][1:] == (start, rates, label):
            self.intervals[-1] = (self.intervals[-1][0], end, rates, label)
        else:
            self.intervals.append((start, end, rates, label))

    def transfer(self, size, earliest, resources, rate_limit, label):
        """Consume residual bandwidth, pausing at existing reservation boundaries."""
        remaining, at = Fraction(size), earliest
        while remaining:
            available = self.available(at)
            rate = min(Fraction(rate_limit), *(available[key] for key in resources))
            boundary = self.next_boundary(at)
            if rate <= 0:
                if boundary is None:
                    raise ValueError("No available transfer service")
                at = boundary
                continue
            duration = ceil(remaining / rate)
            if boundary is not None:
                duration = min(duration, boundary - at)
            amount = min(remaining, rate * duration)
            self.reserve(at, at + duration, {key: amount / duration for key in resources}, label)
            remaining -= amount
            at += duration
        return at

    def summary(self):
        peaks = dict.fromkeys(self.capacities, Fraction(0))
        # Sweep endpoints instead of sampling every simulated cycle.
        changes = {}
        for start, end, rates, _ in self.intervals:
            for at, sign in ((start, 1), (end, -1)):
                change = changes.setdefault(at, {})
                for key, rate in rates.items():
                    change[key] = change.get(key, Fraction(0)) + sign * rate
        used = dict.fromkeys(peaks, Fraction(0))
        for at in sorted(changes):
            for key, rate in changes[at].items():
                used[key] += rate
                peaks[key] = max(peaks[key], used[key])
        return {key: {"capacity_bpc": float(value), "peak_reserved_bpc": float(peaks[key])}
                for key, value in self.capacities.items()}
