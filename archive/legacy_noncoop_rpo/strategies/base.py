from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class StrategyContext:
    t_s: float
    dt_s: float
    x_self_eci: np.ndarray
    x_other_ric_curv: Optional[np.ndarray]
    state: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Strategy(Protocol):
    def command_accel_ric(self, context: StrategyContext) -> np.ndarray:
        ...


PolicyFn = Callable[[float, np.ndarray, np.ndarray], np.ndarray]
StrategyLike = Union[Strategy, PolicyFn]


class CallableStrategy:
    """Adapter for legacy policy callables."""

    def __init__(self, fn: PolicyFn):
        self.fn = fn

    def command_accel_ric(self, context: StrategyContext) -> np.ndarray:
        if context.x_other_ric_curv is None:
            return np.zeros(3, dtype=float)
        return np.asarray(self.fn(context.t_s, context.x_other_ric_curv, context.x_self_eci), dtype=float)


def as_strategy(strategy_like: StrategyLike) -> Strategy:
    if isinstance(strategy_like, Strategy):
        return strategy_like
    return CallableStrategy(strategy_like)
