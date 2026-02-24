from __future__ import annotations

from typing import Callable

import numpy as np

from .base import StrategyContext


class ModelStrategy:
    """Generic model-backed strategy (NN or other regressors)."""

    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray]):
        self.predict_fn = predict_fn

    def command_accel_ric(self, context: StrategyContext) -> np.ndarray:
        if context.x_other_ric_curv is None:
            return np.zeros(3, dtype=float)
        features = np.hstack((context.x_other_ric_curv, context.x_self_eci))
        u = np.asarray(self.predict_fn(features), dtype=float)
        if u.shape != (3,):
            raise ValueError("Model strategy predict_fn must return a 3-vector in RIC.")
        return u
