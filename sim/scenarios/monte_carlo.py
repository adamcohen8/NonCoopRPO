from __future__ import annotations

import csv
import hashlib
import inspect
import math
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from sim.core.models import SimLog
from sim.metrics.engagement import compute_engagement_metrics
from sim.utils.figure_size import cap_figsize
from sim.utils.io import write_json

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_THRESHOLD_KM = 0.1
DEFAULT_OUTCOME_THRESHOLD_BREACH = "threshold_breach"
DEFAULT_OUTCOME_NO_BREACH = "no_breach"


@dataclass(frozen=True)
class MonteCarloConfig:
    """Campaign configuration for repeated scenario execution with uncertainty sampling."""

    runs: int
    base_seed: int = 0
    pos_sigma_km: float = 0.01
    vel_sigma_km_s: float = 1e-4
    threshold_km: float | list[float] = DEFAULT_THRESHOLD_KM
    object_ids: tuple[str, ...] = field(default_factory=tuple)
    initial_position_sigma_km_by_object: dict[str, float] = field(default_factory=dict)
    initial_velocity_sigma_km_s_by_object: dict[str, float] = field(default_factory=dict)
    deployment_timing_jitter_s: float | dict[str, float] = 0.0
    actuator_thrust_magnitude_error_fraction: float | dict[str, float] = 0.0
    actuator_pointing_error_deg: float | dict[str, float] = 0.0
    sensor_noise_scale_multiplier_sigma: float | dict[str, float] = 0.0
    update_cadence_jitter_s: float | dict[str, float] = 0.0
    dropout_probability: float | dict[str, float] | None = None


@dataclass(frozen=True)
class ThresholdProbabilitySummary:
    """Threshold probability summary with a Wilson 95% confidence interval."""

    threshold_km: float
    success_count: int
    failure_count: int
    probability: float
    confidence_interval_95: dict[str, float]


def run_monte_carlo(
    config: MonteCarloConfig,
    scenario_fn,
    output_dir: str,
    plot_mode: Literal["interactive", "save", "both"] = "interactive",
) -> dict[str, str]:
    """Run a Monte Carlo campaign and save campaign-level analytics."""
    if config.runs <= 0:
        raise ValueError("runs must be positive.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = _normalize_thresholds(config.threshold_km)
    config_dict = _to_jsonable(asdict(config))
    metadata = _build_campaign_metadata(config=config, scenario_fn=scenario_fn, thresholds=thresholds)

    run_summaries: list[dict[str, Any]] = []
    min_separations: list[float] = []
    tca_values: list[float] = []
    total_overruns: list[int] = []
    per_object_overruns: dict[str, int] = {}

    for run_idx in tqdm(range(config.runs), desc="Monte Carlo Runs", unit="run", dynamic_ncols=True):
        seed = int(config.base_seed + run_idx)
        rng = np.random.default_rng(seed)
        sampled_parameters = sample_campaign_parameters(config=config, rng=rng)
        scenario_result = _invoke_scenario(
            scenario_fn=scenario_fn,
            seed=seed,
            config=config,
            sampled_parameters=sampled_parameters,
        )
        metrics = compute_engagement_metrics(
            scenario_result["log"],
            keepout_radius_km=scenario_result.get("keepout_radius_km"),
        )
        closest = _compute_closest_approach_summary(scenario_result["log"])
        threshold_breach = any(float(metrics.min_separation_km) < thr for thr in thresholds)
        outcome = classify_run_outcome(
            scenario_result=scenario_result,
            metrics=metrics,
            threshold_breach=threshold_breach,
        )

        run_record = _build_run_record(
            seed=seed,
            metrics=metrics,
            sampled_parameters=sampled_parameters,
            scenario_result=scenario_result,
            closest=closest,
            outcome=outcome,
        )
        run_summaries.append(run_record)
        min_separations.append(float(metrics.min_separation_km))
        total_overruns.append(int(run_record["output_metrics"]["total_overruns"]))
        for object_id, count in run_record["output_metrics"]["compute_overruns_by_object"].items():
            per_object_overruns[object_id] = per_object_overruns.get(object_id, 0) + int(count)
        if closest["time_of_closest_approach_s"] is not None:
            tca_values.append(float(closest["time_of_closest_approach_s"]))

    threshold_probability_table = [
        asdict(summary) for summary in compute_threshold_probabilities(min_separations=min_separations, thresholds_km=thresholds)
    ]
    aggregate = _build_aggregate_summary(
        min_separations=min_separations,
        threshold_probability_table=threshold_probability_table,
        total_overruns=total_overruns,
        per_object_overruns=per_object_overruns,
        runs=run_summaries,
    )
    flat_rows = _build_flat_rows(run_summaries)
    sensitivity = _build_sensitivity_summary(flat_rows)

    summary = {
        "metadata": metadata,
        "config": config_dict,
        "aggregate": aggregate,
        "sensitivity": sensitivity,
        "runs": run_summaries,
    }

    summary_path = out / "monte_carlo_summary.json"
    flat_table_path = out / "monte_carlo_run_table.csv"
    sensitivity_path = out / "monte_carlo_sensitivity.json"
    write_json(str(summary_path), summary)
    write_json(str(sensitivity_path), sensitivity)
    _write_flat_table(flat_table_path, flat_rows)

    plot_paths = _save_plots(
        out_dir=out,
        min_separations=min_separations,
        thresholds=thresholds,
        tca_values=tca_values,
        plot_mode=plot_mode,
    )

    outputs = {
        "summary_json": str(summary_path),
        "flat_table_csv": str(flat_table_path),
        "sensitivity_json": str(sensitivity_path),
    }
    outputs.update(plot_paths)
    return outputs


def sample_campaign_parameters(config: MonteCarloConfig, rng: np.random.Generator) -> dict[str, Any]:
    """Sample one campaign realization from the configured uncertainty model."""
    object_ids = _resolve_object_ids(config)

    sampled = {
        "legacy_args": {
            "pos_sigma_km": float(config.pos_sigma_km),
            "vel_sigma_km_s": float(config.vel_sigma_km_s),
        },
        "by_object": {},
        "global": {},
    }

    for object_id in object_ids:
        pos_sigma = float(config.initial_position_sigma_km_by_object.get(object_id, config.pos_sigma_km))
        vel_sigma = float(config.initial_velocity_sigma_km_s_by_object.get(object_id, config.vel_sigma_km_s))
        timing_sigma = _value_for_object(config.deployment_timing_jitter_s, object_id)
        thrust_sigma = _value_for_object(config.actuator_thrust_magnitude_error_fraction, object_id)
        pointing_sigma_deg = _value_for_object(config.actuator_pointing_error_deg, object_id)
        sensor_sigma = _value_for_object(config.sensor_noise_scale_multiplier_sigma, object_id)
        cadence_sigma = _value_for_object(config.update_cadence_jitter_s, object_id)
        dropout_prob = _value_for_object(config.dropout_probability, object_id)

        sampled["by_object"][object_id] = {
            "initial_position_offset_km": rng.normal(0.0, pos_sigma, size=3).tolist(),
            "initial_velocity_offset_km_s": rng.normal(0.0, vel_sigma, size=3).tolist(),
            "deployment_timing_jitter_s": float(rng.normal(0.0, timing_sigma)),
            "thrust_magnitude_scale": float(1.0 + rng.normal(0.0, thrust_sigma)),
            "actuator_pointing_error_rad": rng.normal(
                0.0,
                math.radians(pointing_sigma_deg),
                size=3,
            ).tolist(),
            "sensor_noise_scale_multiplier": float(max(0.0, 1.0 + rng.normal(0.0, sensor_sigma))),
            "update_cadence_jitter_s": float(rng.normal(0.0, cadence_sigma)),
            "dropout_probability": None if dropout_prob is None else float(np.clip(dropout_prob, 0.0, 1.0)),
        }

    sampled["global"] = {
        "thresholds_km": _normalize_thresholds(config.threshold_km),
    }
    return sampled


def compute_threshold_probabilities(
    min_separations: Sequence[float],
    thresholds_km: float | Sequence[float],
) -> list[ThresholdProbabilitySummary]:
    """Compute threshold breach probabilities and Wilson 95% confidence intervals."""
    values = np.asarray(list(min_separations), dtype=float)
    thresholds = _normalize_thresholds(thresholds_km)
    n = int(values.size)
    rows: list[ThresholdProbabilitySummary] = []
    for threshold in thresholds:
        success_count = int(np.sum(values < threshold))
        probability = float(success_count / n) if n > 0 else 0.0
        ci_low, ci_high = _wilson_confidence_interval(success_count, n)
        rows.append(
            ThresholdProbabilitySummary(
                threshold_km=float(threshold),
                success_count=success_count,
                failure_count=int(max(n - success_count, 0)),
                probability=probability,
                confidence_interval_95={"low": ci_low, "high": ci_high},
            )
        )
    return rows


def classify_run_outcome(
    scenario_result: Mapping[str, Any],
    metrics,
    threshold_breach: bool,
) -> str:
    """Classify the run outcome, honoring explicit labels from the scenario when provided."""
    explicit = scenario_result.get("outcome_label")
    if explicit:
        return str(explicit)

    termination_reason = str(scenario_result.get("termination_reason") or getattr(scenario_result.get("log"), "termination_reason", "") or "")
    if scenario_result.get("insertion_failure") or "insertion" in termination_reason:
        return "insertion_failure"
    if scenario_result.get("fuel_exhausted") or "fuel" in termination_reason:
        return "fuel_exhausted"
    if int(sum(metrics.compute_overruns_by_object.values())) > 0:
        return "control_degraded_by_overrun"
    return DEFAULT_OUTCOME_THRESHOLD_BREACH if threshold_breach else DEFAULT_OUTCOME_NO_BREACH


def _invoke_scenario(
    scenario_fn,
    seed: int,
    config: MonteCarloConfig,
    sampled_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    kwargs = {
        "seed": seed,
        "pos_sigma_km": float(config.pos_sigma_km),
        "vel_sigma_km_s": float(config.vel_sigma_km_s),
        "mc_sample": sampled_parameters,
    }
    call_kwargs = _filter_kwargs_for_callable(scenario_fn, kwargs)
    result = scenario_fn(**call_kwargs)
    if not isinstance(result, dict):
        raise TypeError("scenario_fn must return a dictionary.")
    if "log" not in result:
        raise KeyError("scenario_fn result must include a 'log' entry.")
    return result


def _filter_kwargs_for_callable(fn, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return dict(kwargs)
    allowed = {name for name in sig.parameters.keys()}
    return {key: value for key, value in kwargs.items() if key in allowed}


def _build_run_record(
    seed: int,
    metrics,
    sampled_parameters: Mapping[str, Any],
    scenario_result: Mapping[str, Any],
    closest: Mapping[str, Any],
    outcome: str,
) -> dict[str, Any]:
    output_metrics = {
        "min_separation_km": float(metrics.min_separation_km),
        "time_inside_keepout_s": float(metrics.time_inside_keepout_s),
        "fuel_used_kg_by_object": {k: float(v) for k, v in metrics.fuel_used_kg_by_object.items()},
        "compute_overruns_by_object": {k: int(v) for k, v in metrics.compute_overruns_by_object.items()},
        "jitter_ms_by_object": {k: float(v) for k, v in metrics.jitter_ms_by_object.items()},
        "time_of_closest_approach_s": closest["time_of_closest_approach_s"],
        "closest_approach_pair": closest["closest_approach_pair"],
        "total_overruns": int(sum(metrics.compute_overruns_by_object.values())),
    }
    return {
        "seed": int(seed),
        "sampled_parameters": _to_jsonable(sampled_parameters),
        "outcome": outcome,
        "output_metrics": output_metrics,
        "scenario_summary": _extract_scenario_summary(scenario_result),
    }


def _build_campaign_metadata(config: MonteCarloConfig, scenario_fn, thresholds: list[float]) -> dict[str, Any]:
    config_payload = _to_jsonable(asdict(config))
    payload = {"config": config_payload, "thresholds_km": thresholds, "scenario_function": _scenario_name(scenario_fn)}
    hash_source = _stable_json_bytes(payload)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "scenario_function": _scenario_name(scenario_fn),
        "config_hash": hashlib.sha256(hash_source).hexdigest()[:16],
    }


def _build_aggregate_summary(
    min_separations: Sequence[float],
    threshold_probability_table: list[dict[str, Any]],
    total_overruns: Sequence[int],
    per_object_overruns: Mapping[str, int],
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = np.asarray(list(min_separations), dtype=float)
    percentiles = [5, 25, 50, 75, 95]
    outcomes: dict[str, int] = {}
    for run in runs:
        key = str(run["outcome"])
        outcomes[key] = outcomes.get(key, 0) + 1

    if values.size == 0:
        pct_payload = {str(p): 0.0 for p in percentiles}
        min_value = mean_value = median_value = std_value = 0.0
    else:
        pct_payload = {str(p): float(np.percentile(values, p)) for p in percentiles}
        min_value = float(np.min(values))
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        std_value = float(np.std(values))

    return {
        "run_count": int(len(runs)),
        "min_separation_km_min": min_value,
        "min_separation_km_mean": mean_value,
        "min_separation_km_median": median_value,
        "min_separation_km_std": std_value,
        "min_separation_km_percentiles": pct_payload,
        "threshold_probability_table": threshold_probability_table,
        "total_overruns": int(np.sum(np.asarray(list(total_overruns), dtype=int))) if total_overruns else 0,
        "compute_overruns_by_object": {k: int(v) for k, v in per_object_overruns.items()},
        "outcome_counts": outcomes,
    }


def _build_sensitivity_summary(flat_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not flat_rows:
        return {"ranked_parameters": []}

    target = np.asarray([float(row["output.min_separation_km"]) for row in flat_rows], dtype=float)
    candidates: dict[str, list[float]] = {}
    for row in flat_rows:
        for key, value in row.items():
            if not key.startswith("sampled.") or key == "sampled.global.thresholds_km":
                continue
            numeric = _to_numeric(value)
            if numeric is None:
                continue
            candidates.setdefault(key, []).append(numeric)

    ranked: list[dict[str, Any]] = []
    for key, series in candidates.items():
        values = np.asarray(series, dtype=float)
        if values.size != target.size or values.size < 2:
            continue
        if np.allclose(values, values[0]):
            continue
        pearson = _pearson_corr(values, target)
        spearman = _spearman_corr(values, target)
        if pearson is None and spearman is None:
            continue
        ranked.append(
            {
                "parameter": key,
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "absolute_max_correlation": float(
                    max(abs(pearson or 0.0), abs(spearman or 0.0))
                ),
            }
        )
    ranked.sort(key=lambda item: item["absolute_max_correlation"], reverse=True)
    return {"ranked_parameters": ranked}


def _build_flat_rows(run_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in run_summaries:
        row = {
            "seed": int(run["seed"]),
            "outcome": str(run["outcome"]),
        }
        for key, value in _flatten_mapping(run["sampled_parameters"], prefix="sampled").items():
            row[key] = value
        for key, value in _flatten_mapping(run["output_metrics"], prefix="output").items():
            row[key] = value
        rows.append(row)
    return rows


def _save_plots(
    out_dir: Path,
    min_separations: Sequence[float],
    thresholds: Sequence[float],
    tca_values: Sequence[float],
    plot_mode: Literal["interactive", "save", "both"],
) -> dict[str, str]:
    plt = _load_pyplot()
    values = np.asarray(list(min_separations), dtype=float)
    plot_paths = {
        "histogram_png": "",
        "cdf_png": "",
        "time_of_closest_approach_histogram_png": "",
    }

    bins = min(20, max(5, int(np.sqrt(max(values.size, 1)))))

    fig_hist, ax_hist = plt.subplots(figsize=cap_figsize(8, 4.5))
    ax_hist.hist(values, bins=bins, alpha=0.8)
    for threshold in thresholds:
        ax_hist.axvline(float(threshold), linestyle="--", color="tab:red", alpha=0.6)
    ax_hist.set_title("Monte Carlo Minimum Separation")
    ax_hist.set_xlabel("Minimum Separation (km)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, alpha=0.25)
    fig_hist.tight_layout()
    hist_path = out_dir / "min_separation_hist.png"
    _finalize_plot(fig_hist, hist_path, plot_mode)
    if plot_mode in ("save", "both"):
        plot_paths["histogram_png"] = str(hist_path)

    fig_cdf, ax_cdf = plt.subplots(figsize=cap_figsize(8, 4.5))
    sorted_values = np.sort(values)
    if sorted_values.size > 0:
        empirical = np.arange(1, sorted_values.size + 1, dtype=float) / sorted_values.size
        ax_cdf.step(sorted_values, empirical, where="post")
    for threshold in thresholds:
        ax_cdf.axvline(float(threshold), linestyle="--", color="tab:red", alpha=0.6)
    ax_cdf.set_title("Empirical CDF of Minimum Separation")
    ax_cdf.set_xlabel("Minimum Separation (km)")
    ax_cdf.set_ylabel("Probability")
    ax_cdf.set_ylim(0.0, 1.0)
    ax_cdf.grid(True, alpha=0.25)
    fig_cdf.tight_layout()
    cdf_path = out_dir / "min_separation_cdf.png"
    _finalize_plot(fig_cdf, cdf_path, plot_mode)
    if plot_mode in ("save", "both"):
        plot_paths["cdf_png"] = str(cdf_path)

    if tca_values:
        fig_tca, ax_tca = plt.subplots(figsize=cap_figsize(8, 4.5))
        ax_tca.hist(np.asarray(tca_values, dtype=float), bins=min(20, max(5, int(np.sqrt(len(tca_values))))), alpha=0.8, color="tab:orange")
        ax_tca.set_title("Time of Closest Approach")
        ax_tca.set_xlabel("Time (s)")
        ax_tca.set_ylabel("Count")
        ax_tca.grid(True, alpha=0.25)
        fig_tca.tight_layout()
        tca_path = out_dir / "time_of_closest_approach_hist.png"
        _finalize_plot(fig_tca, tca_path, plot_mode)
        if plot_mode in ("save", "both"):
            plot_paths["time_of_closest_approach_histogram_png"] = str(tca_path)

    return plot_paths


def _finalize_plot(fig, out_path: Path, plot_mode: Literal["interactive", "save", "both"]) -> None:
    plt = _load_pyplot()
    if plot_mode in ("save", "both"):
        fig.savefig(out_path, dpi=150)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def _compute_closest_approach_summary(log: SimLog) -> dict[str, Any]:
    object_ids = sorted(log.truth_by_object.keys())
    best_distance = math.inf
    best_pair: list[str] | None = None
    best_time: float | None = None
    for i, oid_i in enumerate(object_ids):
        ri = log.truth_by_object[oid_i][:, :3]
        for oid_j in object_ids[i + 1 :]:
            rj = log.truth_by_object[oid_j][:, :3]
            d = np.linalg.norm(ri - rj, axis=1)
            idx = int(np.argmin(d))
            distance = float(d[idx])
            if distance < best_distance:
                best_distance = distance
                best_pair = [oid_i, oid_j]
                best_time = float(log.t_s[idx]) if idx < log.t_s.size else None
    if not math.isfinite(best_distance):
        best_distance = 0.0
    return {
        "time_of_closest_approach_s": best_time,
        "closest_approach_pair": best_pair,
        "minimum_separation_km": float(best_distance),
    }


def _extract_scenario_summary(scenario_result: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("output_dir", "keepout_radius_km", "termination_reason", "outcome_label"):
        if key in scenario_result and scenario_result[key] is not None:
            summary[key] = _to_jsonable(scenario_result[key])
    log = scenario_result.get("log")
    if isinstance(log, SimLog):
        summary["terminated_early"] = bool(log.terminated_early)
        summary["termination_reason"] = log.termination_reason
    return summary


def _write_flat_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify_scalar(row.get(key)) for key in fieldnames})


def _value_for_object(value: float | dict[str, float] | None, object_id: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return float(value.get(object_id, 0.0))
    return float(value)


def _resolve_object_ids(config: MonteCarloConfig) -> list[str]:
    ids = set(config.object_ids)
    ids.update(config.initial_position_sigma_km_by_object.keys())
    ids.update(config.initial_velocity_sigma_km_s_by_object.keys())
    for value in (
        config.deployment_timing_jitter_s,
        config.actuator_thrust_magnitude_error_fraction,
        config.actuator_pointing_error_deg,
        config.sensor_noise_scale_multiplier_sigma,
        config.update_cadence_jitter_s,
        config.dropout_probability,
    ):
        if isinstance(value, Mapping):
            ids.update(value.keys())
    return sorted(str(object_id) for object_id in ids)


def _normalize_thresholds(threshold_km: float | Sequence[float]) -> list[float]:
    if isinstance(threshold_km, (int, float)):
        thresholds = [float(threshold_km)]
    else:
        thresholds = [float(value) for value in threshold_km]
    if not thresholds:
        thresholds = [DEFAULT_THRESHOLD_KM]
    return sorted(set(thresholds))


def _wilson_confidence_interval(success_count: int, sample_count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if sample_count <= 0:
        return 0.0, 0.0
    n = float(sample_count)
    p_hat = float(success_count) / n
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + ((z * z) / (4.0 * n * n)))
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def _flatten_mapping(payload: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, path))
        elif isinstance(value, list):
            if value and all(not isinstance(item, (Mapping, list)) for item in value):
                for idx, item in enumerate(value):
                    flat[f"{path}.{idx}"] = item
            else:
                flat[path] = _to_jsonable(value)
        else:
            flat[path] = value
    return flat


def _to_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0.0 or y_std <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    return _pearson_corr(_rankdata(x), _rankdata(y))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.zeros(values.size, dtype=float)
    i = 0
    while i < order.size:
        j = i
        while j + 1 < order.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _scenario_name(fn) -> str:
    return str(getattr(fn, "__name__", fn.__class__.__name__))


def _git_commit_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


def _stable_json_bytes(payload: Any) -> bytes:
    import json

    return json.dumps(_to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _stringify_scalar(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return value
    return str(value)
