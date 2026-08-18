"""Session/graph-level water accounting.

A :class:`Meter` accumulates estimates across many calls — the nodes of an
agent graph, the turns of a chat session, a batch job — and can enforce a
soft water budget, which is the hook you need to *minimise* water across a
chain of agent prompts rather than merely observe it.

Example (any agent framework — LangGraph, a hand-rolled DAG, ...)::

    meter = soif.Meter(budget_ml=50)

    def node(state):
        response = client.chat.completions.create(...)
        meter.record(soif.from_response(response))
        if meter.over_budget:
            state["degrade"] = True   # e.g. route later nodes to a smaller model
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field

from soif.estimator import WaterEstimate


@dataclass
class Meter:
    """Accumulates :class:`WaterEstimate` objects; optionally budget-aware.

    ``budget_ml`` is compared against the *mid* scenario of the running
    total. It is advisory — the meter never raises — so callers decide how
    to react (degrade model tier, truncate context, stop early, ...).
    """

    budget_ml: float | None = None
    records: list[WaterEstimate] = field(default_factory=list)

    def record(self, est: WaterEstimate) -> WaterEstimate:
        """Add an estimate and return it (convenient for pass-through)."""
        self.records.append(est)
        return est

    @property
    def total(self) -> WaterEstimate | None:
        if not self.records:
            return None
        total = self.records[0]
        for est in self.records[1:]:
            total = total + est
        return total

    @property
    def total_ml(self) -> float:
        total = self.total
        return total.total_ml.mid if total else 0.0

    @property
    def over_budget(self) -> bool:
        return self.budget_ml is not None and self.total_ml > self.budget_ml

    @property
    def remaining_ml(self) -> float | None:
        if self.budget_ml is None:
            return None
        return max(0.0, self.budget_ml - self.total_ml)

    def summary(self) -> str:
        total = self.total
        if total is None:
            return "no calls recorded"
        line = f"{len(self.records)} call(s): {total.humanize()}"
        if self.budget_ml is not None:
            state = "OVER" if self.over_budget else "within"
            line += f" — {state} budget ({self.total_ml:.1f}/{self.budget_ml:.1f} mL)"
        return line
