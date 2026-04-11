from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.utils.io import write_json


def build_monte_carlo_report_payload(
    *,
    cfg: SimulationScenarioConfig,
    config_path: str | Path,
    root: dict[str, Any],
    repo_root: Path,
    runs: list[dict[str, Any]],
    run_details: list[dict[str, Any]],
    closest_approach_km_runs: list[float],
    duration_runs_s: list[float],
    total_dv_runs_m_s: list[float],
    guardrail_event_runs: list[int],
    failure_mode_counts: dict[str, int],
    dv_budget_m_s_by_object: dict[str, float],
    gates: dict[str, Any],
    mc_out_cfg: dict[str, Any],
    varies_metadata_seed: bool,
    parallel_active: bool,
    parallel_enabled: bool,
    total_iters: int,
    parallel_workers: int,
    parallel_fallback_reason: str | None = None,
) -> dict[str, Any]:
    from sim.master_simulator import (
        _aggregate_knowledge_consistency_from_runs,
        _aggregate_knowledge_detection_from_runs,
        _build_parameter_sensitivity_rankings,
        _get_git_commit_sha,
        _infer_model_profile,
        _quantile_stats,
        _safe_float,
    )

    durations_s = np.array([float(dict(r.get("summary", {}) or {}).get("duration_s", 0.0)) for r in runs], dtype=float)
    terminated_early_flags = np.array(
        [1.0 if bool(dict(r.get("summary", {}) or {}).get("terminated_early", False)) else 0.0 for r in runs],
        dtype=float,
    )
    termination_reason_counts: dict[str, int] = {}
    dv_by_object: dict[str, list[float]] = {}
    burn_samples_by_object: dict[str, list[float]] = {}
    for entry in runs:
        summary = dict(entry.get("summary", {}) or {})
        term_reason = summary.get("termination_reason")
        if term_reason is not None:
            key = str(term_reason)
            termination_reason_counts[key] = int(termination_reason_counts.get(key, 0) + 1)
        thrust_stats = dict(summary.get("thrust_stats", {}) or {})
        for oid, thrust in thrust_stats.items():
            thrust_dict = dict(thrust or {})
            dv_by_object.setdefault(str(oid), []).append(float(thrust_dict.get("total_dv_m_s", 0.0)))
            burn_samples_by_object.setdefault(str(oid), []).append(float(thrust_dict.get("burn_samples", 0.0)))

    dv_remaining_m_s_by_object: dict[str, list[float]] = {}
    for oid in sorted(dv_budget_m_s_by_object.keys()):
        vals: list[float] = []
        for detail in run_details:
            remaining = _safe_float(dict(detail.get("delta_v_remaining_m_s_by_object", {}) or {}).get(oid))
            if np.isfinite(remaining):
                vals.append(float(remaining))
        if vals:
            dv_remaining_m_s_by_object[oid] = vals

    by_object_stats: dict[str, dict[str, float]] = {}
    all_obj_ids = sorted(set(dv_by_object.keys()) | set(burn_samples_by_object.keys()))
    for oid in all_obj_ids:
        dv_arr = np.array(dv_by_object.get(oid, []), dtype=float)
        burn_arr = np.array(burn_samples_by_object.get(oid, []), dtype=float)
        by_object_stats[oid] = {
            "total_dv_m_s_mean": float(np.mean(dv_arr)) if dv_arr.size else 0.0,
            "total_dv_m_s_min": float(np.min(dv_arr)) if dv_arr.size else 0.0,
            "total_dv_m_s_max": float(np.max(dv_arr)) if dv_arr.size else 0.0,
            "total_dv_m_s_p95": float(np.percentile(dv_arr, 95)) if dv_arr.size else 0.0,
            "burn_samples_mean": float(np.mean(burn_arr)) if burn_arr.size else 0.0,
            "burn_samples_p95": float(np.percentile(burn_arr, 95)) if burn_arr.size else 0.0,
        }

    ca_arr_full = np.array(closest_approach_km_runs, dtype=float)
    ca_finite = ca_arr_full[np.isfinite(ca_arr_full)]
    duration_arr = np.array(duration_runs_s, dtype=float)
    total_dv_arr = np.array(total_dv_runs_m_s, dtype=float)
    guardrail_arr = np.array(guardrail_event_runs, dtype=float)
    pass_flags = np.array([1.0 if bool(d.get("pass", False)) else 0.0 for d in run_details], dtype=float)
    pass_rate = float(np.mean(pass_flags)) if pass_flags.size else 0.0
    guardrail_violation_flags = np.array([1.0 if int(d.get("guardrail_events", 0)) > 0 else 0.0 for d in run_details], dtype=float)

    aggregate_stats = {
        "duration_s_mean": float(np.mean(durations_s)) if durations_s.size else 0.0,
        "duration_s_min": float(np.min(durations_s)) if durations_s.size else 0.0,
        "duration_s_max": float(np.max(durations_s)) if durations_s.size else 0.0,
        "duration_s_p50": float(np.percentile(durations_s, 50)) if durations_s.size else float("nan"),
        "duration_s_p90": float(np.percentile(durations_s, 90)) if durations_s.size else float("nan"),
        "duration_s_p95": float(np.percentile(durations_s, 95)) if durations_s.size else float("nan"),
        "duration_s_p99": float(np.percentile(durations_s, 99)) if durations_s.size else float("nan"),
        "terminated_early_rate": float(np.mean(terminated_early_flags)) if terminated_early_flags.size else 0.0,
        "closest_approach_km_mean": float(np.mean(ca_finite)) if ca_finite.size else float("nan"),
        "closest_approach_km_min": float(np.min(ca_finite)) if ca_finite.size else float("nan"),
        "closest_approach_km_max": float(np.max(ca_finite)) if ca_finite.size else float("nan"),
        "closest_approach_km_p05": float(np.percentile(ca_finite, 5)) if ca_finite.size else float("nan"),
        "closest_approach_km_p50": float(np.percentile(ca_finite, 50)) if ca_finite.size else float("nan"),
        "closest_approach_km_p95": float(np.percentile(ca_finite, 95)) if ca_finite.size else float("nan"),
        "total_dv_m_s_mean": float(np.mean(total_dv_arr)) if total_dv_arr.size else float("nan"),
        "total_dv_m_s_p50": float(np.percentile(total_dv_arr, 50)) if total_dv_arr.size else float("nan"),
        "total_dv_m_s_p90": float(np.percentile(total_dv_arr, 90)) if total_dv_arr.size else float("nan"),
        "total_dv_m_s_p95": float(np.percentile(total_dv_arr, 95)) if total_dv_arr.size else float("nan"),
        "total_dv_m_s_p99": float(np.percentile(total_dv_arr, 99)) if total_dv_arr.size else float("nan"),
        "guardrail_events_mean": float(np.mean(guardrail_arr)) if guardrail_arr.size else float("nan"),
        "guardrail_events_p95": float(np.percentile(guardrail_arr, 95)) if guardrail_arr.size else float("nan"),
        "pass_rate": pass_rate,
        "fail_rate": 1.0 - pass_rate,
        "guardrail_violation_rate": float(np.mean(guardrail_violation_flags)) if guardrail_violation_flags.size else float("nan"),
        "failure_mode_counts": failure_mode_counts,
        "termination_reason_counts": termination_reason_counts,
        "by_object": by_object_stats,
        "delta_v_budget_m_s_by_object": dict(dv_budget_m_s_by_object),
        "delta_v_remaining_m_s_by_object": {
            oid: _quantile_stats(vals, (50.0, 90.0, 99.0)) for oid, vals in sorted(dv_remaining_m_s_by_object.items())
        },
        "knowledge_detection_by_observer": _aggregate_knowledge_detection_from_runs(run_details),
        "knowledge_consistency_by_observer": _aggregate_knowledge_consistency_from_runs(run_details),
    }

    keepout_threshold = _safe_float(gates.get("min_closest_approach_km"))
    if not np.isfinite(keepout_threshold):
        keepout_threshold = _safe_float(mc_out_cfg.get("keepout_radius_km"))
    p_keepout_violation = float("nan")
    if np.isfinite(keepout_threshold) and ca_finite.size:
        p_keepout_violation = float(np.mean(ca_finite < keepout_threshold))

    catastrophic_failure_reasons = [str(x) for x in (mc_out_cfg.get("catastrophic_failure_reasons", ["terminated_early:earth_impact"]) or [])]
    catastrophic_count = 0
    for detail in run_details:
        reasons = set(str(x) for x in list(detail.get("fail_reasons", []) or []))
        if any(reason in reasons for reason in catastrophic_failure_reasons):
            catastrophic_count += 1
    p_catastrophic_outcome = float(catastrophic_count / max(len(run_details), 1))

    max_duration_gate = _safe_float(gates.get("max_duration_s"))
    p_exceed_time_budget = float("nan")
    if np.isfinite(max_duration_gate) and duration_arr.size:
        p_exceed_time_budget = float(np.mean(duration_arr > max_duration_gate))

    max_total_dv_gate = _safe_float(gates.get("max_total_dv_m_s"))
    p_exceed_dv_budget = float("nan")
    if np.isfinite(max_total_dv_gate) and total_dv_arr.size:
        p_exceed_dv_budget = float(np.mean(total_dv_arr > max_total_dv_gate))

    if np.isfinite(max_total_dv_gate) and total_dv_arr.size:
        fuel_margin_stats = _quantile_stats(max_total_dv_gate - total_dv_arr, (5.0, 50.0, 95.0))
    else:
        fuel_margin_stats = _quantile_stats([], (5.0, 50.0, 95.0))
    if np.isfinite(max_duration_gate) and duration_arr.size:
        time_margin_stats = _quantile_stats(max_duration_gate - duration_arr, (5.0, 50.0, 95.0))
    else:
        time_margin_stats = _quantile_stats([], (5.0, 50.0, 95.0))
    aggregate_stats["p_keepout_violation"] = p_keepout_violation
    aggregate_stats["p_catastrophic_outcome"] = p_catastrophic_outcome
    aggregate_stats["p_exceed_dv_budget"] = p_exceed_dv_budget
    aggregate_stats["p_exceed_time_budget"] = p_exceed_time_budget

    top_failure_modes: list[dict[str, Any]] = []
    if failure_mode_counts:
        for reason, count in sorted(failure_mode_counts.items(), key=lambda kv: int(kv[1]), reverse=True)[:3]:
            top_failure_modes.append(
                {
                    "reason": str(reason),
                    "count": int(count),
                    "rate": float(int(count) / max(len(run_details), 1)),
                }
            )

    cfg_json = json.dumps(root, sort_keys=True, separators=(",", ":"))
    reproducibility = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _get_git_commit_sha(repo_root),
        "config_sha256": hashlib.sha256(cfg_json.encode("utf-8")).hexdigest(),
        "model_profile": _infer_model_profile(root),
        "random_seed_policy": (
            "Per-run metadata.seed set to monte_carlo.base_seed + iteration unless metadata.seed is explicitly varied."
            if not varies_metadata_seed
            else "metadata.seed controlled by monte_carlo variations."
        ),
    }
    sensitivity_rankings = _build_parameter_sensitivity_rankings(run_details)
    top_parameter_drivers = [
        {
            "parameter_path": str(row.get("parameter_path")),
            "importance_score": _safe_float(row.get("importance_score"), default=0.0),
            "abs_corr_pass": _safe_float(row.get("abs_corr_pass")),
            "abs_corr_closest_approach_km": _safe_float(row.get("abs_corr_closest_approach_km")),
            "abs_corr_total_dv_m_s": _safe_float(row.get("abs_corr_total_dv_m_s")),
        }
        for row in sensitivity_rankings[:5]
    ]
    unique_seeds = len(set(int(_safe_float(d.get("seed"), default=-1)) for d in run_details))
    finite_ca_rate = float(np.mean(np.isfinite(ca_arr_full))) if ca_arr_full.size else float("nan")
    analysis_confidence = {
        "runs_executed": int(len(run_details)),
        "run_count_sufficient_for_tail_estimates": bool(len(run_details) >= 100),
        "unique_seed_count": int(unique_seeds),
        "finite_closest_approach_rate": finite_ca_rate,
        "varied_parameter_count": int(len(set(str(v.parameter_path) for v in cfg.monte_carlo.variations))),
        "model_profile": reproducibility.get("model_profile"),
        "git_commit_sha": reproducibility.get("git_commit_sha"),
        "config_sha256": reproducibility.get("config_sha256"),
    }
    commander_brief = {
        "scenario_name": cfg.scenario_name,
        "runs": int(cfg.monte_carlo.iterations),
        "p_success": pass_rate,
        "p_fail": 1.0 - pass_rate,
        "p_keepout_violation": p_keepout_violation,
        "p_catastrophic_outcome": p_catastrophic_outcome,
        "p_exceed_dv_budget": p_exceed_dv_budget,
        "p_exceed_time_budget": p_exceed_time_budget,
        "keepout_threshold_km": keepout_threshold if np.isfinite(keepout_threshold) else None,
        "worst_case_closest_approach_km": float(np.min(ca_finite)) if ca_finite.size else float("nan"),
        "timeline_confidence_bands_s": _quantile_stats(duration_runs_s, (50.0, 90.0, 99.0)),
        "fuel_confidence_bands_total_dv_m_s": _quantile_stats(total_dv_runs_m_s, (50.0, 90.0, 99.0)),
        "fuel_confidence_bands_dv_m_s_by_object": {
            oid: _quantile_stats(dv_by_object.get(oid, []), (50.0, 90.0, 99.0)) for oid in sorted(dv_by_object.keys())
        },
        "delta_v_remaining_confidence_bands_m_s_by_object": {
            oid: _quantile_stats(vals, (50.0, 90.0, 99.0)) for oid, vals in sorted(dv_remaining_m_s_by_object.items())
        },
        "resource_margin": {
            "fuel_margin_m_s_vs_budget": fuel_margin_stats,
            "time_margin_s_vs_budget": time_margin_stats,
        },
        "constraint_violation_summary": {
            "p_guardrail_violation": float(np.mean(guardrail_violation_flags)) if guardrail_violation_flags.size else float("nan"),
            "guardrail_events_per_run": _quantile_stats(guardrail_event_runs, (50.0, 90.0, 99.0)),
            "compute_deadline_overrun_available": False,
            "control_saturation_available": False,
        },
        "top_failure_modes": top_failure_modes,
        "top_parameter_drivers": top_parameter_drivers,
        "analysis_confidence": analysis_confidence,
    }

    analyst_pack = {
        "scenario_name": cfg.scenario_name,
        "reproducibility": reproducibility,
        "gates": gates,
        "run_details": run_details,
        "failure_mode_counts": failure_mode_counts,
        "sensitivity_rankings": sensitivity_rankings,
        "catastrophic_failure_reasons": catastrophic_failure_reasons,
    }

    agg = {
        "config_path": str(Path(config_path).resolve()),
        "scenario_name": cfg.scenario_name,
        "scenario_description": cfg.scenario_description,
        "monte_carlo": {
            "enabled": True,
            "iterations": int(cfg.monte_carlo.iterations),
            "base_seed": int(cfg.monte_carlo.base_seed),
            "parallel_enabled": bool(parallel_active),
            "parallel_requested": bool(parallel_enabled and total_iters > 1),
            "parallel_workers": int(parallel_workers if parallel_active else 1),
        },
        "aggregate_stats": aggregate_stats,
        "commander_brief": commander_brief,
        "reproducibility": reproducibility,
        "analyst_pack": analyst_pack,
        "artifacts": {},
        "runs": runs,
    }
    if parallel_fallback_reason is not None:
        agg["monte_carlo"]["parallel_fallback_reason"] = str(parallel_fallback_reason)

    return {
        "agg": agg,
        "commander_brief": commander_brief,
        "analyst_pack": analyst_pack,
        "durations_s": durations_s,
        "ca_finite": ca_finite,
        "all_obj_ids": all_obj_ids,
        "dv_by_object": dv_by_object,
        "dv_remaining_m_s_by_object": dv_remaining_m_s_by_object,
        "run_details": run_details,
        "keepout_threshold": keepout_threshold,
        "failure_mode_counts": failure_mode_counts,
    }


def write_monte_carlo_report_artifacts(
    *,
    cfg: SimulationScenarioConfig,
    outdir: Path,
    agg: dict[str, Any],
    commander_brief: dict[str, Any],
    analyst_pack: dict[str, Any],
    run_details: list[dict[str, Any]],
    mc_out_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from sim.master_simulator import _write_commander_brief_markdown

    mc_outputs = dict(mc_out_cfg or cfg.outputs.monte_carlo or {})
    if not bool(mc_outputs.get("save_aggregate_summary", True)):
        return agg

    summary_path = outdir / "master_monte_carlo_summary.json"
    commander_json_path = outdir / "master_monte_carlo_commander_brief.json"
    commander_md_path = outdir / "master_monte_carlo_commander_brief.md"
    analyst_path = outdir / "master_monte_carlo_analyst_pack.json"

    write_json(str(summary_path), agg)
    write_json(str(commander_json_path), commander_brief)
    _write_commander_brief_markdown(commander_md_path, commander_brief)
    write_json(str(analyst_path), analyst_pack)

    artifacts = dict(agg.get("artifacts", {}) or {})
    artifacts["summary_json"] = str(summary_path)
    artifacts["commander_brief_json"] = str(commander_json_path)
    artifacts["commander_brief_md"] = str(commander_md_path)
    artifacts["analyst_pack_json"] = str(analyst_path)

    if bool(mc_outputs.get("save_raw_runs", False)):
        runs_path = outdir / "master_monte_carlo_run_details.json"
        write_json(str(runs_path), {"scenario_name": cfg.scenario_name, "run_details": run_details})
        artifacts["run_details_json"] = str(runs_path)

    agg["artifacts"] = artifacts
    return agg


def apply_monte_carlo_baseline_comparison(
    *,
    agg: dict[str, Any],
    commander_brief: dict[str, Any],
    config_path: str | Path,
    baseline_summary_json: str,
) -> dict[str, Any]:
    from sim.master_simulator import _build_baseline_comparison, _load_json_file

    baseline_path_text = str(baseline_summary_json or "").strip()
    if not baseline_path_text:
        return agg
    baseline_path = Path(baseline_path_text)
    if not baseline_path.is_absolute():
        baseline_path = Path(config_path).resolve().parent / baseline_path
    baseline_payload = _load_json_file(baseline_path)
    if baseline_payload is not None:
        comparison = _build_baseline_comparison(agg, baseline_payload)
        agg["baseline_comparison"] = comparison
        commander_brief["baseline_comparison"] = comparison
    else:
        agg["baseline_comparison_error"] = f"Unable to load baseline summary: {str(baseline_path)}"
    return agg
