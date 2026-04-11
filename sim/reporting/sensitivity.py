from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.utils.io import write_json


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _deep_get(root: dict[str, Any], path: str, default: Any = None) -> Any:
    parts = path.split(".")
    cur: Any = root
    for tok in parts:
        if "[" in tok and tok.endswith("]"):
            key, idx_txt = tok[:-1].split("[", 1)
            idx = int(idx_txt)
            if key:
                if not isinstance(cur, dict) or key not in cur:
                    return default
                cur = cur[key]
            if not isinstance(cur, list) or idx < 0 or idx >= len(cur):
                return default
            cur = cur[idx]
            continue
        if not isinstance(cur, dict) or tok not in cur:
            return default
        cur = cur[tok]
    return cur


def analysis_metrics(cfg: SimulationScenarioConfig) -> list[str]:
    metrics = [str(x).strip() for x in list(cfg.analysis.metrics or []) if str(x).strip()]
    if metrics:
        return metrics
    return [
        "summary.duration_s",
        "summary.terminated_early",
        "summary.termination_reason",
        "derived.closest_approach_km",
    ]


def extract_analysis_metric(run_payload: dict[str, Any], metric_path: str) -> Any:
    from sim.master_simulator import _closest_approach_from_run_payload

    path = str(metric_path or "").strip()
    if not path:
        return None
    if path == "derived.closest_approach_km":
        value = _closest_approach_from_run_payload(run_payload)
        return float(value) if np.isfinite(value) else None
    if path.startswith("summary."):
        return _deep_get(dict(run_payload.get("summary", {}) or {}), path[len("summary."):], default=None)
    if path.startswith("payload."):
        return _deep_get(run_payload, path[len("payload."):], default=None)
    return _deep_get(run_payload, path, default=None)


def extract_analysis_metrics(run_payload: dict[str, Any], metric_paths: list[str]) -> dict[str, Any]:
    return {path: extract_analysis_metric(run_payload, path) for path in metric_paths}


def _serialize_analysis_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (int, bool)):
        return str(value)
    if value is None:
        return "null"
    return json.dumps(value, sort_keys=True)


def build_lhs_parameter_summaries(
    *,
    cfg: SimulationScenarioConfig,
    metric_paths: list[str],
    run_details: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not run_details:
        return [], []
    summaries: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    for param in cfg.analysis.sensitivity.parameters:
        parameter_path = str(param.parameter_path)
        x_vals: list[float] = []
        for detail in run_details:
            value = dict(detail.get("sampled_parameters", {}) or {}).get(parameter_path)
            if isinstance(value, bool):
                x_vals.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                x_vals.append(float(value))
            else:
                x_vals.append(float("nan"))
        x = np.array(x_vals, dtype=float)
        finite_x = np.isfinite(x)
        metric_correlations: dict[str, dict[str, float]] = {}
        max_abs_corr = 0.0
        for metric_path in metric_paths:
            y_vals: list[float] = []
            for detail in run_details:
                value = dict(detail.get("metrics", {}) or {}).get(metric_path)
                if isinstance(value, bool):
                    y_vals.append(1.0 if value else 0.0)
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    y_vals.append(float(value))
                else:
                    y_vals.append(float("nan"))
            y = np.array(y_vals, dtype=float)
            finite = finite_x & np.isfinite(y)
            corr = float("nan")
            if int(np.sum(finite)) >= 3:
                x_ok = x[finite]
                y_ok = y[finite]
                if not np.allclose(np.std(x_ok), 0.0) and not np.allclose(np.std(y_ok), 0.0):
                    corr = float(np.corrcoef(x_ok, y_ok)[0, 1])
            abs_corr = abs(corr) if np.isfinite(corr) else float("nan")
            if np.isfinite(abs_corr):
                max_abs_corr = max(max_abs_corr, abs_corr)
            metric_correlations[metric_path] = {
                "correlation": corr,
                "abs_correlation": abs_corr,
            }
        finite_param = x[finite_x]
        summaries.append(
            {
                "parameter_path": parameter_path,
                "distribution": str(param.distribution),
                "sample_count": int(finite_param.size),
                "value_stats": {
                    "mean": float(np.mean(finite_param)) if finite_param.size else float("nan"),
                    "min": float(np.min(finite_param)) if finite_param.size else float("nan"),
                    "max": float(np.max(finite_param)) if finite_param.size else float("nan"),
                    "std": float(np.std(finite_param)) if finite_param.size else float("nan"),
                },
                "metric_correlations": metric_correlations,
            }
        )
        rankings.append(
            {
                "parameter_path": parameter_path,
                "distribution": str(param.distribution),
                "sample_count": int(finite_param.size),
                "max_abs_correlation": float(max_abs_corr),
            }
        )
    rankings.sort(key=lambda row: float(row.get("max_abs_correlation", 0.0)), reverse=True)
    return summaries, rankings


def aggregate_sensitivity_parameter_runs(
    *,
    parameter_path: str,
    metric_paths: list[str],
    run_details: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for detail in run_details:
        if str(detail.get("parameter_path", "")) != parameter_path:
            continue
        value = detail.get("parameter_value")
        value_key = _serialize_analysis_value(value)
        bucket = grouped.setdefault(
            value_key,
            {
                "parameter_value": value,
                "runs": [],
                "metrics": {},
            },
        )
        bucket["runs"].append(detail)

    for bucket in grouped.values():
        metric_summary: dict[str, Any] = {}
        for metric_path in metric_paths:
            values = [run["metrics"].get(metric_path) for run in bucket["runs"]]
            numeric = np.array(
                [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(float(v))],
                dtype=float,
            )
            entry: dict[str, Any] = {"sample_count": int(len(values))}
            if numeric.size:
                entry["mean"] = float(np.mean(numeric))
                entry["min"] = float(np.min(numeric))
                entry["max"] = float(np.max(numeric))
                baseline_value = baseline_metrics.get(metric_path)
                if isinstance(baseline_value, (int, float)) and not isinstance(baseline_value, bool) and np.isfinite(float(baseline_value)):
                    entry["delta_from_baseline"] = float(np.mean(numeric) - float(baseline_value))
            else:
                non_null = [v for v in values if v is not None]
                if non_null:
                    entry["values"] = non_null
                    baseline_value = baseline_metrics.get(metric_path)
                    if baseline_value is not None:
                        entry["matches_baseline"] = bool(all(v == baseline_value for v in non_null))
            metric_summary[metric_path] = entry
        bucket["metrics"] = metric_summary

    response_curve = [grouped[key] for key in sorted(grouped.keys())]
    max_abs_delta = 0.0
    for bucket in response_curve:
        for metric_entry in dict(bucket.get("metrics", {}) or {}).values():
            delta = metric_entry.get("delta_from_baseline")
            if isinstance(delta, (int, float)) and np.isfinite(float(delta)):
                max_abs_delta = max(max_abs_delta, abs(float(delta)))
    return {
        "parameter_path": parameter_path,
        "value_count": int(len(response_curve)),
        "max_abs_delta_from_baseline": float(max_abs_delta),
        "response_curve": response_curve,
    }


def build_sensitivity_report_payload(
    *,
    cfg: SimulationScenarioConfig,
    config_path: str | Path,
    prepared: list[dict[str, Any]],
    completed: dict[int, dict[str, Any]],
    baseline: dict[str, Any] | None,
    metric_paths: list[str],
    sensitivity_method: str,
    parallel_enabled: bool,
    parallel_active: bool,
    parallel_workers: int,
    parallel_fallback_reason: str | None,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for prepared_run in sorted(prepared, key=lambda entry: int(entry["iteration"])):
        idx = int(prepared_run["iteration"])
        completed_run = dict(completed.get(idx, {}) or {})
        payload = dict(completed_run.get("payload", {}) or {})
        if not payload:
            payload = {
                "summary": dict(completed_run.get("summary", {}) or {}),
            }
        metrics = extract_analysis_metrics(payload, metric_paths)
        runs.append(
            {
                "iteration": idx,
                "parameter_path": prepared_run.get("parameter_path"),
                "parameter_value": prepared_run.get("parameter_value"),
                "value_index": int(prepared_run["value_index"]),
                "sampled_parameters": dict(prepared_run.get("sampled_parameters", {}) or {}),
                "summary": dict(completed_run.get("summary", payload.get("summary", {})) or {}),
                "closest_approach_km": _safe_float(completed_run.get("closest_approach_km")),
                "metrics": metrics,
            }
        )

    baseline_metrics = dict(baseline.get("metrics", {}) or {}) if baseline is not None else {}
    if sensitivity_method == "lhs":
        parameter_summaries, parameter_rankings = build_lhs_parameter_summaries(
            cfg=cfg,
            metric_paths=metric_paths,
            run_details=runs,
        )
    else:
        parameter_summaries = [
            aggregate_sensitivity_parameter_runs(
                parameter_path=str(param.parameter_path),
                metric_paths=metric_paths,
                run_details=runs,
                baseline_metrics=baseline_metrics,
            )
            for param in cfg.analysis.sensitivity.parameters
        ]
        parameter_rankings = sorted(
            (
                {
                    "parameter_path": str(summary.get("parameter_path", "")),
                    "max_abs_delta_from_baseline": float(summary.get("max_abs_delta_from_baseline", 0.0)),
                    "value_count": int(summary.get("value_count", 0)),
                }
                for summary in parameter_summaries
            ),
            key=lambda entry: float(entry["max_abs_delta_from_baseline"]),
            reverse=True,
        )

    agg = {
        "config_path": str(Path(config_path).resolve()),
        "scenario_name": cfg.scenario_name,
        "scenario_description": cfg.scenario_description,
        "analysis": {
            "enabled": True,
            "study_type": "sensitivity",
            "method": sensitivity_method,
            "parallel_enabled": bool(parallel_active),
            "parallel_requested": bool(parallel_enabled and len(prepared) > 1),
            "parallel_workers": int(parallel_workers if parallel_active else 1),
            "metrics": metric_paths,
            "parameter_count": int(len(cfg.analysis.sensitivity.parameters)),
            "run_count": int(len(runs)),
            "samples": int(cfg.analysis.sensitivity.samples if sensitivity_method == "lhs" else len(runs)),
            "seed": int(cfg.analysis.sensitivity.seed),
        },
        "monte_carlo": {"enabled": False},
        "baseline": baseline,
        "parameter_summaries": parameter_summaries,
        "parameter_rankings": parameter_rankings,
        "runs": runs,
        "artifacts": {},
    }
    if parallel_fallback_reason is not None:
        agg["analysis"]["parallel_fallback_reason"] = str(parallel_fallback_reason)
    return agg


def write_sensitivity_summary_artifact(
    *,
    outdir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    summary_path = outdir / "master_analysis_sensitivity_summary.json"
    write_json(str(summary_path), payload)
    artifacts = dict(payload.get("artifacts", {}) or {})
    artifacts["summary_json"] = str(summary_path)
    payload["artifacts"] = artifacts
    return payload
