from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .base import StrategyContext

Condition = Callable[[StrategyContext], bool]
Action = Callable[[StrategyContext], np.ndarray]


@dataclass(frozen=True)
class EventRule:
    condition: Condition
    action: Action


class EventBasedStrategy:
    """Evaluates rules in order and applies the first matching action."""

    def __init__(self, rules: list[EventRule], default_action: Optional[Action] = None):
        self.rules = list(rules)
        self.default_action = default_action

    def command_accel_ric(self, context: StrategyContext) -> np.ndarray:
        for rule in self.rules:
            if rule.condition(context):
                u = np.asarray(rule.action(context), dtype=float)
                if u.shape != (3,):
                    raise ValueError("Event rule action must return a 3-vector in RIC.")
                return u
        if self.default_action is not None:
            u = np.asarray(self.default_action(context), dtype=float)
            if u.shape != (3,):
                raise ValueError("default_action must return a 3-vector in RIC.")
            return u
        return np.zeros(3, dtype=float)
