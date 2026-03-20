from __future__ import annotations

import numpy as np


def rk4_step_state(deriv_fn, t_s: float, x: np.ndarray, dt_s: float) -> np.ndarray:
    k1 = deriv_fn(t_s, x)
    k2 = deriv_fn(t_s + 0.5 * dt_s, x + 0.5 * dt_s * k1)
    k3 = deriv_fn(t_s + 0.5 * dt_s, x + 0.5 * dt_s * k2)
    k4 = deriv_fn(t_s + dt_s, x + dt_s * k3)
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rkf78_step(deriv_fn, t_s: float, x: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Fehlberg embedded Runge-Kutta 7(8) step.

    Returns the propagated state and the embedded error estimate used for
    adaptive step-size control.
    """
    k1 = deriv_fn(t_s, x)
    k2 = deriv_fn(t_s + dt_s * (2.0 / 27.0), x + dt_s * ((2.0 / 27.0) * k1))
    k3 = deriv_fn(t_s + dt_s * (1.0 / 9.0), x + dt_s * ((1.0 / 36.0) * k1 + (1.0 / 12.0) * k2))
    k4 = deriv_fn(t_s + dt_s * (1.0 / 6.0), x + dt_s * ((1.0 / 24.0) * k1 + (1.0 / 8.0) * k3))
    k5 = deriv_fn(
        t_s + dt_s * (5.0 / 12.0),
        x + dt_s * ((5.0 / 12.0) * k1 - (25.0 / 16.0) * k3 + (25.0 / 16.0) * k4),
    )
    k6 = deriv_fn(
        t_s + dt_s * 0.5,
        x + dt_s * ((1.0 / 20.0) * k1 + 0.25 * k4 + 0.2 * k5),
    )
    k7 = deriv_fn(
        t_s + dt_s * (5.0 / 6.0),
        x + dt_s * (-(25.0 / 108.0) * k1 + (125.0 / 108.0) * k4 - (65.0 / 27.0) * k5 + (125.0 / 54.0) * k6),
    )
    k8 = deriv_fn(
        t_s + dt_s * (1.0 / 6.0),
        x + dt_s * ((31.0 / 300.0) * k1 + (61.0 / 225.0) * k5 - (2.0 / 9.0) * k6 + (13.0 / 900.0) * k7),
    )
    k9 = deriv_fn(
        t_s + dt_s * (2.0 / 3.0),
        x + dt_s * (2.0 * k1 - (53.0 / 6.0) * k4 + (704.0 / 45.0) * k5 - (107.0 / 9.0) * k6 + (67.0 / 90.0) * k7 + 3.0 * k8),
    )
    k10 = deriv_fn(
        t_s + dt_s * (1.0 / 3.0),
        x
        + dt_s
        * (
            -(91.0 / 108.0) * k1
            + (23.0 / 108.0) * k4
            - (976.0 / 135.0) * k5
            + (311.0 / 54.0) * k6
            - (19.0 / 60.0) * k7
            + (17.0 / 6.0) * k8
            - (1.0 / 12.0) * k9
        ),
    )
    k11 = deriv_fn(
        t_s + dt_s,
        x
        + dt_s
        * (
            (2383.0 / 4100.0) * k1
            - (341.0 / 164.0) * k4
            + (4496.0 / 1025.0) * k5
            - (301.0 / 82.0) * k6
            + (2133.0 / 4100.0) * k7
            + (45.0 / 82.0) * k8
            + (45.0 / 164.0) * k9
            + (18.0 / 41.0) * k10
        ),
    )
    k12 = deriv_fn(
        t_s,
        x
        + dt_s
        * (
            (3.0 / 205.0) * k1
            - (6.0 / 41.0) * k6
            - (3.0 / 205.0) * k7
            - (3.0 / 41.0) * k8
            + (3.0 / 41.0) * k9
            + (6.0 / 41.0) * k10
        ),
    )
    k13 = deriv_fn(
        t_s + dt_s,
        x
        + dt_s
        * (
            -(1777.0 / 4100.0) * k1
            - (341.0 / 164.0) * k4
            + (4496.0 / 1025.0) * k5
            - (289.0 / 82.0) * k6
            + (2193.0 / 4100.0) * k7
            + (51.0 / 82.0) * k8
            + (33.0 / 164.0) * k9
            + (12.0 / 41.0) * k10
            + k12
        ),
    )

    x_next = x + dt_s * (
        (41.0 / 840.0) * k1
        + (34.0 / 105.0) * k6
        + (9.0 / 35.0) * k7
        + (9.0 / 35.0) * k8
        + (9.0 / 280.0) * k9
        + (9.0 / 280.0) * k10
        + (41.0 / 840.0) * k11
    )
    err = dt_s * (41.0 / 840.0) * (k1 + k11 - k12 - k13)
    return x_next, err


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
    max_substeps: int = 4096,
    method: str = "rkf78",
) -> np.ndarray:
    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative.")
    if dt_s == 0.0:
        return np.array(x, dtype=float, copy=True)

    method_name = str(method).strip().lower()
    if method_name == "rkf78":
        step_fn = rkf78_step
        growth_exponent = -1.0 / 8.0
    elif method_name in ("dopri5", "dopri45"):
        step_fn = dopri45_step
        growth_exponent = -1.0 / 5.0
    else:
        raise ValueError(f"Unknown adaptive integrator method '{method}'.")

    t = t_s
    xk = x
    remain = dt_s
    h = min(dt_s, 1.0)
    min_h = max(1e-12, 1e-12 * max(1.0, abs(dt_s)))
    steps = 0

    while remain > 0.0 and steps < max_substeps:
        h = min(h, remain)
        x_next, err = step_fn(deriv_fn, t, xk, h)
        scale = atol + rtol * np.maximum(np.abs(xk), np.abs(x_next))
        err_ratio = float(np.max(np.abs(err) / np.maximum(scale, 1e-14)))

        if err_ratio <= 1.0:
            t += h
            xk = x_next
            remain -= h
            if err_ratio < 1e-10:
                h *= 2.0
            else:
                h *= min(2.0, max(0.5, 0.9 * err_ratio**growth_exponent))
        else:
            h *= max(0.1, 0.9 * err_ratio**growth_exponent)
            if h < min_h:
                raise RuntimeError(
                    f"Adaptive integrator step size underflow at t={t:.9f}s while trying to cover dt={dt_s:.9f}s."
                )
        steps += 1

    if remain > max(min_h, 1e-9 * max(1.0, abs(dt_s))):
        raise RuntimeError(
            f"Adaptive integrator exhausted {max_substeps} internal substeps with {remain:.9e}s remaining."
        )
    return xk
