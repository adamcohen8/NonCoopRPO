from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeadlineDecision:
    runtime_ms: float
    overrun: bool


def evaluate_controller_runtime(runtime_ms: float, budget_ms: float) -> DeadlineDecision:
    return DeadlineDecision(runtime_ms=runtime_ms, overrun=runtime_ms > budget_ms)
