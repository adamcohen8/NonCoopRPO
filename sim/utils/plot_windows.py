from __future__ import annotations

import numpy as np

RIC_FOLLOW_MARGIN = 1.38


def attitude_axis_limits(frame: str, lim: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    axis_lim = float(max(lim, 1e-9))
    if str(frame).strip().lower() == "ric":
        return (axis_lim, -axis_lim), (-axis_lim, axis_lim), (-axis_lim, axis_lim)
    return (-axis_lim, axis_lim), (-axis_lim, axis_lim), (-axis_lim, axis_lim)


def fuel_fraction_from_remaining_series(values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=float).reshape(-1)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return out
    initial = float(finite[0])
    if initial <= 0.0:
        out[np.isfinite(arr)] = 0.0
        return out
    out[np.isfinite(arr)] = np.clip(arr[np.isfinite(arr)] / initial, 0.0, 1.0)
    return out


def axis_window_from_values(
    values: list[np.ndarray],
    *,
    min_span: float = 1.0,
    margin: float = RIC_FOLLOW_MARGIN,
) -> tuple[float, float]:
    finite_parts: list[np.ndarray] = []
    for arr in values:
        a = np.array(arr, dtype=float).reshape(-1)
        finite = a[np.isfinite(a)]
        if finite.size > 0:
            finite_parts.append(finite)
    if not finite_parts:
        half_span = 0.5 * float(min_span)
        return -half_span, half_span
    merged = np.concatenate(finite_parts)
    lower = float(np.min(merged))
    upper = float(np.max(merged))
    center = 0.5 * (lower + upper)
    span = max(float(min_span), margin * max(upper - lower, 0.0))
    half_span = 0.5 * span
    return center - half_span, center + half_span


def windows_from_points(
    points: list[np.ndarray],
    *,
    axis_indices: tuple[int, ...],
    min_span: float = 1.0,
    margin: float = RIC_FOLLOW_MARGIN,
) -> list[tuple[float, float]]:
    axis_values: list[list[np.ndarray]] = [[] for _ in axis_indices]
    for point in points:
        arr = np.array(point, dtype=float).reshape(-1)
        if arr.size <= max(axis_indices, default=0):
            continue
        for out_idx, axis_idx in enumerate(axis_indices):
            axis_values[out_idx].append(np.array([arr[axis_idx]], dtype=float))
    return [
        axis_window_from_values(values, min_span=min_span, margin=margin)
        for values in axis_values
    ]
