"""Uncertainty arithmetic: every quantity in soif is a (low, mid, high) triple.

Published per-prompt water figures disagree by two orders of magnitude
(Google reports 0.26 mL per median Gemini prompt; Mistral's lifecycle
analysis reports 45 mL per 400-token Large 2 response). Propagating a
low/mid/high scenario through every factor keeps that uncertainty visible
instead of pretending to a precision nobody has.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Triple:
    low: float
    mid: float
    high: float

    def __add__(self, other: Triple | float) -> Triple:
        if isinstance(other, Triple):
            return Triple(self.low + other.low, self.mid + other.mid, self.high + other.high)
        return Triple(self.low + other, self.mid + other, self.high + other)

    __radd__ = __add__

    def __mul__(self, other: Triple | float) -> Triple:
        if isinstance(other, Triple):
            return Triple(self.low * other.low, self.mid * other.mid, self.high * other.high)
        return Triple(self.low * other, self.mid * other, self.high * other)

    __rmul__ = __mul__

    def scale(self, k: float) -> Triple:
        return self * k

    def to_dict(self) -> dict[str, float]:
        return {"low": self.low, "mid": self.mid, "high": self.high}

    @staticmethod
    def zero() -> Triple:
        return Triple(0.0, 0.0, 0.0)

    @staticmethod
    def of(value: Triple | tuple[float, float, float] | float) -> Triple:
        if isinstance(value, Triple):
            return value
        if isinstance(value, tuple):
            return Triple(*value)
        return Triple(value, value, value)
