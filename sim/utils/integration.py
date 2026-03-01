from __future__ import annotations

import numpy as np


def rk4_step(deriv_fn, x: np.ndarray, dt_s: float, *args):
    k1 = deriv_fn(x, *args)
    k2 = deriv_fn(x + 0.5 * dt_s * k1, *args)
    k3 = deriv_fn(x + 0.5 * dt_s * k2, *args)
    k4 = deriv_fn(x + dt_s * k3, *args)
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
