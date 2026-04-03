from __future__ import annotations

import numpy as np


def rk4_step_state(deriv_fn, t_s: float, x: np.ndarray, dt_s: float) -> np.ndarray:
    k1 = deriv_fn(t_s, x)
    k2 = deriv_fn(t_s + 0.5 * dt_s, x + 0.5 * dt_s * k1)
    k3 = deriv_fn(t_s + 0.5 * dt_s, x + 0.5 * dt_s * k2)
    k4 = deriv_fn(t_s + dt_s, x + dt_s * k3)
    return x + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rkf78_stage_trace(deriv_fn, t_s: float, x: np.ndarray, dt_s: float) -> list[dict[str, np.ndarray | float | str]]:
    x = np.array(x, dtype=float, copy=True)
    stages: list[dict[str, np.ndarray | float | str]] = []

    def record(name: str, stage_t: float, stage_x: np.ndarray) -> np.ndarray:
        k = deriv_fn(stage_t, stage_x)
        stages.append(
            {
                "name": name,
                "t": float(stage_t),
                "x": np.array(stage_x, dtype=float, copy=True),
                "k": np.array(k, dtype=float, copy=True),
            }
        )
        return k

    k1 = record("k1", t_s, x)
    k2 = record("k2", t_s + dt_s * (2.0 / 27.0), x + dt_s * ((2.0 / 27.0) * k1))
    k3 = record("k3", t_s + dt_s * (1.0 / 9.0), x + dt_s * ((1.0 / 36.0) * k1 + (1.0 / 12.0) * k2))
    k4 = record("k4", t_s + dt_s * (1.0 / 6.0), x + dt_s * ((1.0 / 24.0) * k1 + (1.0 / 8.0) * k3))
    k5 = record(
        "k5",
        t_s + dt_s * (5.0 / 12.0),
        x + dt_s * ((5.0 / 12.0) * k1 - (25.0 / 16.0) * k3 + (25.0 / 16.0) * k4),
    )
    k6 = record(
        "k6",
        t_s + dt_s * 0.5,
        x + dt_s * ((1.0 / 20.0) * k1 + 0.25 * k4 + 0.2 * k5),
    )
    k7 = record(
        "k7",
        t_s + dt_s * (5.0 / 6.0),
        x + dt_s * (-(25.0 / 108.0) * k1 + (125.0 / 108.0) * k4 - (65.0 / 27.0) * k5 + (125.0 / 54.0) * k6),
    )
    k8 = record(
        "k8",
        t_s + dt_s * (1.0 / 6.0),
        x + dt_s * ((31.0 / 300.0) * k1 + (61.0 / 225.0) * k5 - (2.0 / 9.0) * k6 + (13.0 / 900.0) * k7),
    )
    k9 = record(
        "k9",
        t_s + dt_s * (2.0 / 3.0),
        x + dt_s * (2.0 * k1 - (53.0 / 6.0) * k4 + (704.0 / 45.0) * k5 - (107.0 / 9.0) * k6 + (67.0 / 90.0) * k7 + 3.0 * k8),
    )
    k10 = record(
        "k10",
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
    k11 = record(
        "k11",
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
    k12 = record(
        "k12",
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
    record(
        "k13",
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
            + stages[-1]["k"]
        ),
    )
    return stages


def rkf78_step(deriv_fn, t_s: float, x: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Fehlberg embedded Runge-Kutta 7(8) step.

    Returns the propagated state and the embedded error estimate used for
    adaptive step-size control.
    """
    stages = rkf78_stage_trace(deriv_fn, t_s, x, dt_s)
    k1 = stages[0]["k"]
    k6 = stages[5]["k"]
    k7 = stages[6]["k"]
    k8 = stages[7]["k"]
    k9 = stages[8]["k"]
    k10 = stages[9]["k"]
    k11 = stages[10]["k"]
    k12 = stages[11]["k"]
    k13 = stages[12]["k"]

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


def integrate_rkf78_hpop(
    deriv_fn,
    t_s: float,
    x: np.ndarray,
    dt_s: float,
    *,
    tolerance: float = 1e-10,
    h_init: float | None = None,
    max_attempts: int = 12,
) -> tuple[np.ndarray, float]:
    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative.")
    if dt_s == 0.0:
        return np.array(x, dtype=float, copy=True), float(h_init if h_init is not None else 0.01)

    min_scale = 0.125
    max_scale = 4.0
    err_exponent = 1.0 / 7.0

    x_now = float(t_s)
    x_end = float(t_s + dt_s)
    h = float(h_init if h_init is not None else 0.01)
    if h <= 0.0:
        h = 0.01
    y = np.array(x, dtype=float, copy=True)
    last_interval = False
    if h > (x_end - x_now):
        h = x_end - x_now
        last_interval = True

    tol_per_unit = float(tolerance) / (x_end - x_now)

    while x_now < x_end:
        scale = 1.0
        for attempt in range(max_attempts):
            y_trial, err_vec = rkf78_step(deriv_fn, x_now, y, h)
            err = float(np.linalg.norm(err_vec))
            if err == 0.0:
                scale = max_scale
                break
            y_norm = float(np.linalg.norm(y))
            yy = tol_per_unit if y_norm == 0.0 else y_norm
            scale = 0.8 * (tol_per_unit * yy / err) ** err_exponent
            scale = min(max(scale, min_scale), max_scale)
            if err < (tol_per_unit * yy):
                break
            h *= scale
            if x_now + h > x_end:
                h = x_end - x_now
            elif x_now + h + 0.5 * h > x_end:
                h = 0.5 * h
        else:
            raise RuntimeError(
                f"HPOP-style RKF78 failed to converge within {max_attempts} attempts at t={x_now:.9f}s."
            )

        y = y_trial
        x_now += h
        h *= scale
        h_next = h
        if last_interval:
            return y, h_next
        if x_now + h > x_end:
            last_interval = True
            h = x_end - x_now
        elif x_now + h + 0.5 * h > x_end:
            h = 0.5 * h

    return y, float(h)


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
        x_next, _ = integrate_rkf78_hpop(
            deriv_fn=deriv_fn,
            t_s=t_s,
            x=x,
            dt_s=dt_s,
            tolerance=rtol,
            h_init=min(dt_s, 0.01),
        )
        return x_next
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
