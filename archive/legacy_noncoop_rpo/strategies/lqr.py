from __future__ import annotations

import numpy as np

from .base import StrategyContext


class HCWLQRStrategy:
    """LQR-based strategy over HCW relative state."""

    def __init__(self, n_rad_s: float, dt_s: float, a_max_km_s2: float, position_only: bool = False):
        try:
            from ..hcw_control import build_hcw_lqr, build_hcw_lqr_position_only
        except Exception as exc:  # pragma: no cover - environment/dependency specific
            raise RuntimeError(
                "HCWLQRStrategy requires a working SciPy installation compatible with the active NumPy version."
            ) from exc

        if position_only:
            self._controller = build_hcw_lqr_position_only(n_rad_s, dt_s, a_max_km_s2)
        else:
            self._controller = build_hcw_lqr(n_rad_s, dt_s, a_max_km_s2)

    def command_accel_ric(self, context: StrategyContext) -> np.ndarray:
        if context.x_other_ric_curv is None:
            return np.zeros(3, dtype=float)
        return np.asarray(self._controller(context.x_other_ric_curv), dtype=float)
