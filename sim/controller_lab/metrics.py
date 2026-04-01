from __future__ import annotations

from typing import Any

import numpy as np

from sim.controller_lab.models import ControllerBenchMetric


def _deep_get(root: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = root
    for token in str(path).split("."):
        if not token:
            continue
        if not isinstance(cur, dict) or token not in cur:
            return default
        cur = cur[token]
    return cur


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    arr = np.array(q, dtype=float).reshape(-1)
    if arr.size != 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    nrm = float(np.linalg.norm(arr))
    if nrm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return arr / nrm


def _quat_error_deg_series(truth_hist: np.ndarray, desired_q: np.ndarray) -> np.ndarray:
    q_des = _normalize_quaternion(desired_q)
    q_hist = np.array(truth_hist[:, 6:10], dtype=float)
    out = np.full(q_hist.shape[0], np.nan)
    for i in range(q_hist.shape[0]):
        q_cur = _normalize_quaternion(q_hist[i])
        dot = float(np.clip(np.dot(q_des, q_cur), -1.0, 1.0))
        out[i] = float(np.degrees(2.0 * np.arccos(abs(dot))))
    return out


def evaluate_metric(metric: ControllerBenchMetric, payload: dict[str, Any]) -> Any:
    if metric.source_path:
        return _deep_get(payload, metric.source_path)

    kind = str(metric.kind or "").strip().lower()
    object_id = str(metric.object_id or "").strip()
    ref_id = str(metric.reference_object_id or "").strip()

    truth_map = dict(payload.get("truth_by_object", {}) or {})
    debug_map = dict(payload.get("controller_debug_by_object", {}) or {})
    summary = dict(payload.get("summary", {}) or {})

    if kind in {"total_dv_m_s", "burn_samples", "max_accel_km_s2"}:
        thrust_stats = dict(summary.get("thrust_stats", {}) or {})
        return _deep_get({"thrust_stats": thrust_stats}, f"thrust_stats.{object_id}.{kind}")

    if kind in {"mean_controller_runtime_ms", "max_controller_runtime_ms", "controller_trace_samples"}:
        debug_hist = list(debug_map.get(object_id, []) or [])
        runtimes = np.array([_safe_float(d.get("controller_runtime_ms")) for d in debug_hist], dtype=float)
        runtimes = runtimes[np.isfinite(runtimes)]
        if kind == "mean_controller_runtime_ms":
            return float(np.mean(runtimes)) if runtimes.size else float("nan")
        if kind == "max_controller_runtime_ms":
            return float(np.max(runtimes)) if runtimes.size else float("nan")
        return int(len(debug_hist))

    if kind in {"final_body_rate_norm_rad_s", "mean_body_rate_norm_rad_s"}:
        truth = np.array(truth_map.get(object_id, []), dtype=float)
        if truth.ndim != 2 or truth.shape[1] < 13 or truth.shape[0] == 0:
            return float("nan")
        rates = np.linalg.norm(truth[:, 10:13], axis=1)
        if kind == "final_body_rate_norm_rad_s":
            return float(rates[-1])
        return float(np.mean(rates))

    if kind in {"final_attitude_error_deg", "rms_attitude_error_deg", "max_attitude_error_deg"}:
        truth = np.array(truth_map.get(object_id, []), dtype=float)
        if truth.ndim != 2 or truth.shape[1] < 10 or truth.shape[0] == 0 or metric.desired_quat_bn is None:
            return float("nan")
        err = _quat_error_deg_series(truth, np.array(metric.desired_quat_bn, dtype=float))
        finite = err[np.isfinite(err)]
        if finite.size == 0:
            return float("nan")
        if kind == "final_attitude_error_deg":
            return float(finite[-1])
        if kind == "rms_attitude_error_deg":
            return float(np.sqrt(np.mean(finite**2)))
        return float(np.max(finite))

    if kind in {"final_relative_distance_km", "min_relative_distance_km"}:
        truth = np.array(truth_map.get(object_id, []), dtype=float)
        ref = np.array(truth_map.get(ref_id, []), dtype=float)
        if truth.ndim != 2 or ref.ndim != 2 or truth.shape[0] == 0 or ref.shape[0] == 0:
            return float("nan")
        n = int(min(truth.shape[0], ref.shape[0]))
        dist = np.linalg.norm(truth[:n, :3] - ref[:n, :3], axis=1)
        if kind == "final_relative_distance_km":
            return float(dist[-1])
        return float(np.min(dist))

    return None
