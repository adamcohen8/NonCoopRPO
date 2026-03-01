from __future__ import annotations

import numpy as np


def rk4_step_state(deriv_fn, t_s: float, x: np.ndarray, dt_s: float) -> np.ndarray:
    k1 = deriv_fn(t_s, x)
    k2 = deriv_fn(t_s + 0.5 * dt_s, x + 0.5 * dt_s * k1)
    k3 = deriv_fn(t_s + 0.5 * dt_s, x + 0.5 * dt_s * k2)
    k4 = deriv_fn(t_s + dt_s, x + dt_s * k3)
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def dopri45_step(deriv_fn, t_s: float, x: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    k1 = deriv_fn(t_s, x)
    k2 = deriv_fn(t_s + dt_s * 1 / 5, x + dt_s * (1 / 5) * k1)
    k3 = deriv_fn(t_s + dt_s * 3 / 10, x + dt_s * (3 / 40 * k1 + 9 / 40 * k2))
    k4 = deriv_fn(t_s + dt_s * 4 / 5, x + dt_s * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3))
    k5 = deriv_fn(
        t_s + dt_s * 8 / 9,
        x + dt_s * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4),
    )
    k6 = deriv_fn(
        t_s + dt_s,
        x + dt_s * (9017 / 3168 * k1 - 355 / 33 * k2 + 46732 / 5247 * k3 + 49 / 176 * k4 - 5103 / 18656 * k5),
    )
    k7 = deriv_fn(
        t_s + dt_s,
        x + dt_s * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6),
    )

    x5 = x + dt_s * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6)
    x4 = x + dt_s * (
        5179 / 57600 * k1
        + 7571 / 16695 * k3
        + 393 / 640 * k4
        - 92097 / 339200 * k5
        + 187 / 2100 * k6
        + 1 / 40 * k7
    )
    err = x5 - x4
    return x5, err


def integrate_adaptive(
    deriv_fn,
    t_s: float,
    x: np.ndarray,
    dt_s: float,
    atol: float = 1e-9,
    rtol: float = 1e-7,
    max_substeps: int = 32,
) -> np.ndarray:
    t = t_s
    xk = x
    remain = dt_s
    h = min(dt_s, 1.0)
    steps = 0

    while remain > 0.0 and steps < max_substeps:
        h = min(h, remain)
        x_next, err = dopri45_step(deriv_fn, t, xk, h)
        scale = atol + rtol * np.maximum(np.abs(xk), np.abs(x_next))
        err_ratio = float(np.max(np.abs(err) / np.maximum(scale, 1e-14)))

        if err_ratio <= 1.0:
            t += h
            xk = x_next
            remain -= h
            if err_ratio < 1e-10:
                h *= 2.0
            else:
                h *= min(2.0, max(0.5, 0.9 * err_ratio ** (-0.2)))
        else:
            h *= max(0.1, 0.9 * err_ratio ** (-0.25))
        steps += 1

    return xk
