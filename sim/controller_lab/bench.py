from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import importlib
from itertools import product
import math
import os
from pathlib import Path
import re
from typing import Any

import yaml

from sim.config import load_simulation_yaml, scenario_config_from_dict
from sim.controller_lab.metrics import evaluate_metric
from sim.controller_lab.models import (
    ControllerBenchCase,
    ControllerBenchConfig,
    ControllerBenchMetric,
    ControllerBenchObjective,
    ControllerBenchPassCriterion,
    ControllerBenchTarget,
    ControllerVariant,
)
from sim.controller_lab.plotting import render_controller_bench_visualizations, write_linear_feedback_diagnostics
from sim.controller_lab.reporting import write_controller_bench_reports
from sim.master_simulator import _analysis_study_type, _restore_env_vars, _set_parallel_worker_thread_limits
from sim.single_run import _run_single_config


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _should_use_tqdm() -> bool:
    return not _is_truthy_env("NONCOOP_GUI")


def _make_plain_progress_reporter(label: str):
    def _report(index: int, total: int, detail: str) -> None:
        print(f"{label}: {index}/{total} - {detail}")

    return _report


def _as_dict(value: Any, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping/object.")
    return dict(value)


def _deep_set(root: dict[str, Any], path: str, value: Any) -> None:
    tokens = [tok for tok in str(path).split(".") if tok]
    if not tokens:
        raise ValueError("Sweep path must be non-empty.")
    cur: dict[str, Any] = root
    for token in tokens[:-1]:
        nxt = cur.get(token)
        if nxt is None:
            nxt = {}
            cur[token] = nxt
        if not isinstance(nxt, dict):
            raise ValueError(f"Sweep path '{path}' cannot descend through non-mapping token '{token}'.")
        cur = nxt
    cur[tokens[-1]] = deepcopy(value)


def _format_sweep_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_sweep_value(v) for v in value) + "]"
    if value is None:
        return "none"
    return str(value)


def _slug_token(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return out or "value"


def _sweep_labels(paths: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    used: set[str] = set()
    token_lists = {path: [tok for tok in str(path).split(".") if tok] for path in paths}
    for path in paths:
        tokens = token_lists[path]
        candidate_parts = [tokens[-1]] if tokens else ["value"]
        idx = 2
        label = _slug_token("_".join(candidate_parts))
        while label in used:
            if idx <= len(tokens):
                candidate_parts.insert(0, tokens[-idx])
                idx += 1
            else:
                candidate_parts = tokens
            label = _slug_token("_".join(candidate_parts))
        used.add(label)
        labels[path] = label
    return labels


def _expand_variant_entry(raw_variant: dict[str, Any]) -> list[ControllerVariant]:
    d = _as_dict(raw_variant, "variants[*]")
    name = str(d.get("name", "") or "").strip()
    pointer = _as_dict(d.get("controller"), "variants[*].controller")
    if not name:
        raise ValueError("variants[*].name must be non-empty.")
    description = str(d.get("description", "") or "")

    raw_sweep = d.get("sweep")
    if raw_sweep is None:
        return [ControllerVariant(name=name, pointer=pointer, description=description)]
    sweep = _as_dict(raw_sweep, "variants[*].sweep")
    if not sweep:
        return [ControllerVariant(name=name, pointer=pointer, description=description)]

    sweep_paths = [str(path or "").strip() for path in sweep.keys()]
    if any(not path for path in sweep_paths):
        raise ValueError("variants[*].sweep keys must be non-empty dotted paths.")
    labels = _sweep_labels(sweep_paths)
    value_lists: list[list[Any]] = []
    for path in sweep_paths:
        values = sweep.get(path)
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"variants[*].sweep['{path}'] must be a non-empty list.")
        value_lists.append(list(values))

    expanded: list[ControllerVariant] = []
    for combo in product(*value_lists):
        pointer_copy = deepcopy(pointer)
        sweep_parts: list[str] = []
        sweep_desc_parts: list[str] = []
        for path, value in zip(sweep_paths, combo):
            root_path = path
            if root_path.startswith("controller."):
                root_path = root_path[len("controller.") :]
            _deep_set(pointer_copy, root_path, value)
            label = labels[path]
            pretty = _format_sweep_value(value)
            sweep_parts.append(f"{label}_{_slug_token(pretty)}")
            sweep_desc_parts.append(f"{label}={pretty}")
        variant_name = f"{name}__{'__'.join(sweep_parts)}"
        variant_desc = description.strip()
        sweep_desc = ", ".join(sweep_desc_parts)
        if variant_desc:
            variant_desc = f"{variant_desc} [{sweep_desc}]"
        else:
            variant_desc = f"Sweep: {sweep_desc}"
        expanded.append(ControllerVariant(name=variant_name, pointer=pointer_copy, description=variant_desc))
    return expanded


def _parse_metric(raw: Any) -> ControllerBenchMetric:
    d = _as_dict(raw, "metric")
    desired = d.get("desired_quat_bn")
    desired_q = None
    if desired is not None:
        vals = tuple(float(x) for x in list(desired))
        if len(vals) != 4:
            raise ValueError("metric.desired_quat_bn must be length-4.")
        desired_q = vals
    return ControllerBenchMetric(
        name=str(d.get("name", "") or "").strip(),
        source_path=str(d.get("source_path", "") or "").strip(),
        kind=str(d.get("kind", "") or "").strip(),
        object_id=str(d.get("object_id", "") or "").strip(),
        reference_object_id=str(d.get("reference_object_id", "") or "").strip(),
        desired_quat_bn=desired_q,
        keepout_radius_km=float(d.get("keepout_radius_km")) if d.get("keepout_radius_km") is not None else None,
    )


def _parse_pass_criterion(raw: Any) -> ControllerBenchPassCriterion:
    d = _as_dict(raw, "pass_criterion")
    metric = str(d.get("metric", "") or "").strip()
    op = str(d.get("op", "") or "").strip()
    if not metric or not op:
        raise ValueError("pass criteria require non-empty 'metric' and 'op'.")
    return ControllerBenchPassCriterion(metric=metric, op=op, value=d.get("value"))


def _parse_objective(raw: Any) -> ControllerBenchObjective:
    d = _as_dict(raw, "objective")
    desired = d.get("desired_quat_bn")
    desired_q = None
    if desired is not None:
        vals = tuple(float(x) for x in list(desired))
        if len(vals) != 4:
            raise ValueError("objective.desired_quat_bn must be length-4.")
        desired_q = vals
    kind = str(d.get("kind", "") or "").strip().lower()
    if not kind:
        raise ValueError("objective.kind must be non-empty.")
    return ControllerBenchObjective(
        kind=kind,
        name=str(d.get("name", "") or "").strip(),
        object_id=str(d.get("object_id", "") or "").strip(),
        reference_object_id=str(d.get("reference_object_id", "") or "").strip(),
        desired_quat_bn=desired_q,
        keepout_radius_km=float(d.get("keepout_radius_km")) if d.get("keepout_radius_km") is not None else None,
        max_final_attitude_error_deg=(
            float(d.get("max_final_attitude_error_deg"))
            if d.get("max_final_attitude_error_deg") is not None
            else None
        ),
        max_rms_attitude_error_deg=(
            float(d.get("max_rms_attitude_error_deg")) if d.get("max_rms_attitude_error_deg") is not None else None
        ),
        max_final_body_rate_norm_rad_s=(
            float(d.get("max_final_body_rate_norm_rad_s"))
            if d.get("max_final_body_rate_norm_rad_s") is not None
            else None
        ),
        max_final_relative_distance_km=(
            float(d.get("max_final_relative_distance_km"))
            if d.get("max_final_relative_distance_km") is not None
            else None
        ),
        max_rms_relative_distance_km=(
            float(d.get("max_rms_relative_distance_km"))
            if d.get("max_rms_relative_distance_km") is not None
            else None
        ),
        max_final_relative_speed_km_s=(
            float(d.get("max_final_relative_speed_km_s"))
            if d.get("max_final_relative_speed_km_s") is not None
            else None
        ),
        max_time_inside_keepout_s=(
            float(d.get("max_time_inside_keepout_s")) if d.get("max_time_inside_keepout_s") is not None else None
        ),
        max_total_dv_m_s=float(d.get("max_total_dv_m_s")) if d.get("max_total_dv_m_s") is not None else None,
        max_fuel_used_kg=float(d.get("max_fuel_used_kg")) if d.get("max_fuel_used_kg") is not None else None,
        require_not_terminated_early=bool(d.get("require_not_terminated_early", False)),
    )


def load_controller_bench_config(path: str | Path) -> ControllerBenchConfig:
    cfg_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Controller bench YAML root must be a mapping/object.")

    target_raw = _as_dict(raw.get("controller_target"), "controller_target")
    target = ControllerBenchTarget(
        object_id=str(target_raw.get("object_id", "target") or "target"),
        slot=str(target_raw.get("slot", "attitude_control") or "attitude_control"),
    )

    variants: list[ControllerVariant] = []
    for entry in list(raw.get("variants", []) or []):
        variants.extend(_expand_variant_entry(_as_dict(entry, "variants[*]")))
    variant_names = [variant.name for variant in variants]
    if len(set(variant_names)) != len(variant_names):
        raise ValueError("Controller bench variants must have unique names after sweep expansion.")

    cases: list[ControllerBenchCase] = []
    for entry in list(raw.get("cases", []) or []):
        d = _as_dict(entry, "cases[*]")
        name = str(d.get("name", "") or "").strip()
        cfg_ref = str(d.get("config_path", "") or "").strip()
        if not name or not cfg_ref:
            raise ValueError("cases[*] require non-empty 'name' and 'config_path'.")
        case_metrics = tuple(_parse_metric(m) for m in list(d.get("metrics", []) or []))
        pass_criteria = tuple(_parse_pass_criterion(p) for p in list(d.get("pass_criteria", []) or []))
        objectives = tuple(_parse_objective(o) for o in list(d.get("objectives", []) or []))
        cases.append(
            ControllerBenchCase(
                name=name,
                config_path=(cfg_path.parent / cfg_ref).resolve(),
                description=str(d.get("description", "") or ""),
                metrics=case_metrics,
                pass_criteria=pass_criteria,
                objectives=objectives,
            )
        )

    metrics = tuple(_parse_metric(m) for m in list(raw.get("metrics", []) or []))
    pass_criteria = tuple(_parse_pass_criterion(p) for p in list(raw.get("pass_criteria", []) or []))
    objectives = tuple(_parse_objective(o) for o in list(raw.get("objectives", []) or []))
    output_dir = Path(str(raw.get("output_dir", f"outputs/controller_bench/{raw.get('suite_name', 'suite')}")))
    if not output_dir.is_absolute():
        output_dir = (cfg_path.parent / output_dir).resolve()
    plot_mode = str(raw.get("plot_mode", "save") or "save").strip().lower()
    if plot_mode not in {"interactive", "save", "both"}:
        raise ValueError("controller bench plot_mode must be one of: interactive, save, both.")

    return ControllerBenchConfig(
        suite_name=str(raw.get("suite_name", cfg_path.stem) or cfg_path.stem),
        description=str(raw.get("description", "") or ""),
        output_dir=output_dir,
        plot_mode=plot_mode,
        controller_target=target,
        variants=tuple(variants),
        cases=tuple(cases),
        metrics=metrics,
        pass_criteria=pass_criteria,
        objectives=objectives,
        save_run_payloads=bool(raw.get("save_run_payloads", True)),
        disable_plots=bool(raw.get("disable_plots", True)),
        disable_animations=bool(raw.get("disable_animations", True)),
        print_individual_run_summaries=bool(raw.get("print_individual_run_summaries", False)),
        parallel_enabled=bool(raw.get("parallel_enabled", False)),
        parallel_workers=int(raw.get("parallel_workers", 0)),
    )


def _set_controller_pointer(root: dict[str, Any], target: ControllerBenchTarget, pointer: dict[str, Any]) -> None:
    section = root.setdefault(str(target.object_id), {})
    section[str(target.slot)] = deepcopy(pointer)


def _apply_benchmark_output_overrides(root: dict[str, Any], outdir: Path, suite: ControllerBenchConfig) -> None:
    outputs = root.setdefault("outputs", {})
    outputs["output_dir"] = str(outdir)
    outputs["mode"] = "save"
    stats = outputs.setdefault("stats", {})
    stats["print_summary"] = bool(suite.print_individual_run_summaries)
    stats["save_json"] = True
    stats["save_full_log"] = bool(suite.save_run_payloads)
    plots = outputs.setdefault("plots", {})
    if suite.disable_plots:
        plots["enabled"] = False
        plots["figure_ids"] = []
    animations = outputs.setdefault("animations", {})
    if suite.disable_animations:
        animations["enabled"] = False
        animations["types"] = []


def _evaluate_pass_criteria(metrics: dict[str, Any], criteria: tuple[ControllerBenchPassCriterion, ...]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for criterion in criteria:
        lhs = metrics.get(criterion.metric)
        rhs = criterion.value
        ok = False
        if criterion.op == "<=":
            ok = lhs is not None and float(lhs) <= float(rhs)
        elif criterion.op == "<":
            ok = lhs is not None and float(lhs) < float(rhs)
        elif criterion.op == ">=":
            ok = lhs is not None and float(lhs) >= float(rhs)
        elif criterion.op == ">":
            ok = lhs is not None and float(lhs) > float(rhs)
        elif criterion.op == "==":
            ok = lhs == rhs
        elif criterion.op == "!=":
            ok = lhs != rhs
        else:
            raise ValueError(f"Unsupported pass-criterion op: {criterion.op}")
        if not ok:
            failures.append(f"{criterion.metric} {criterion.op} {criterion.value}")
    return (len(failures) == 0, failures)


def _slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return out or "objective"


def _controller_linear_system_summary(pointer: dict[str, Any]) -> dict[str, Any] | None:
    module_name = str(pointer.get("module", "") or "").strip()
    class_name = str(pointer.get("class_name", "") or "").strip()
    params = dict(pointer.get("params", {}) or {})
    if not module_name or not class_name:
        return None
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        ctrl = cls(**params)
    except Exception:
        return None
    fn = getattr(ctrl, "linear_system_summary", None)
    if not callable(fn):
        return None
    try:
        summary = fn()
    except Exception:
        return None
    return dict(summary or {}) if isinstance(summary, dict) else None


def _build_relative_rendezvous_leaderboards(
    runs: list[dict[str, Any]],
    variants: list[ControllerVariant],
) -> list[dict[str, Any]]:
    objective_runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        variant_name = str(run.get("variant_name", "") or "").strip()
        for objective_result in list(run.get("objective_results", []) or []):
            if str(objective_result.get("kind", "") or "").strip() != "relative_rendezvous":
                continue
            objective_name = str(objective_result.get("name", "") or "").strip()
            if not objective_name or not variant_name:
                continue
            objective_runs.setdefault(objective_name, {}).setdefault(variant_name, []).append(dict(objective_result))

    metric_specs = [
        ("final_relative_distance_km", "Mean Final Range (km)", "asc"),
        ("rms_relative_distance_km", "Mean RMS Range (km)", "asc"),
        ("final_relative_speed_km_s", "Mean Final Speed (km/s)", "asc"),
        ("total_dv_m_s", "Mean Total dV (m/s)", "asc"),
        ("fuel_used_kg", "Mean Fuel Used (kg)", "asc"),
        ("time_inside_keepout_s", "Mean Keepout Time (s)", "asc"),
    ]

    leaderboards: list[dict[str, Any]] = []
    for objective_name, by_variant in objective_runs.items():
        rankings: list[dict[str, Any]] = []

        pass_entries: list[dict[str, Any]] = []
        for variant in variants:
            rows = by_variant.get(variant.name, [])
            if not rows:
                continue
            run_count = len(rows)
            pass_rate = float(sum(1 for row in rows if bool(row.get("passed", False))) / run_count)
            pass_entries.append(
                {
                    "variant_name": variant.name,
                    "value": pass_rate,
                    "run_count": run_count,
                }
            )
        pass_entries.sort(key=lambda row: (-float(row["value"]), str(row["variant_name"])))
        if pass_entries:
            rankings.append(
                {
                    "metric": "objective_pass_rate",
                    "label": "Objective Pass Rate",
                    "direction": "desc",
                    "entries": [
                        {
                            "rank": idx + 1,
                            "variant_name": str(row["variant_name"]),
                            "value": float(row["value"]),
                            "run_count": int(row["run_count"]),
                        }
                        for idx, row in enumerate(pass_entries)
                    ],
                }
            )

        for metric_suffix, label, direction in metric_specs:
            entries: list[dict[str, Any]] = []
            for variant in variants:
                rows = by_variant.get(variant.name, [])
                if not rows:
                    continue
                values: list[float] = []
                for row in rows:
                    prefix = str(row.get("metric_prefix", "") or "").strip()
                    metric_value = dict(row.get("metrics", {}) or {}).get(f"{prefix}_{metric_suffix}")
                    try:
                        value_f = float(metric_value)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(value_f):
                        continue
                    values.append(value_f)
                if values:
                    entries.append(
                        {
                            "variant_name": variant.name,
                            "value": float(sum(values) / len(values)),
                            "sample_count": len(values),
                        }
                    )
            if not entries:
                continue
            reverse = direction == "desc"
            entries.sort(key=lambda row: float(row["value"]), reverse=reverse)
            if not reverse:
                entries.sort(key=lambda row: (float(row["value"]), str(row["variant_name"])))
            else:
                entries.sort(key=lambda row: (-float(row["value"]), str(row["variant_name"])))
            rankings.append(
                {
                    "metric": metric_suffix,
                    "label": label,
                    "direction": direction,
                    "entries": [
                        {
                            "rank": idx + 1,
                            "variant_name": str(row["variant_name"]),
                            "value": float(row["value"]),
                            "sample_count": int(row["sample_count"]),
                        }
                        for idx, row in enumerate(entries)
                    ],
                }
            )

        leaderboards.append(
            {
                "objective_name": objective_name,
                "kind": "relative_rendezvous",
                "rankings": rankings,
            }
        )
    return leaderboards


def _dedupe_metrics(metrics: list[ControllerBenchMetric]) -> tuple[ControllerBenchMetric, ...]:
    seen: set[str] = set()
    out: list[ControllerBenchMetric] = []
    for metric in metrics:
        if metric.name in seen:
            continue
        seen.add(metric.name)
        out.append(metric)
    return tuple(out)


def _metric_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def _objective_specifications(
    objective: ControllerBenchObjective,
) -> tuple[tuple[ControllerBenchMetric, ...], tuple[ControllerBenchPassCriterion, ...]]:
    prefix = _slug(objective.name or objective.kind)
    metrics: list[ControllerBenchMetric] = []
    criteria: list[ControllerBenchPassCriterion] = []

    if objective.kind == "attitude_hold":
        if not objective.object_id:
            raise ValueError("attitude_hold objective requires object_id.")
        if objective.desired_quat_bn is not None:
            final_name = _metric_name(prefix, "final_attitude_error_deg")
            rms_name = _metric_name(prefix, "rms_attitude_error_deg")
            metrics.append(
                ControllerBenchMetric(
                    name=final_name,
                    kind="final_attitude_error_deg",
                    object_id=objective.object_id,
                    desired_quat_bn=objective.desired_quat_bn,
                )
            )
            metrics.append(
                ControllerBenchMetric(
                    name=rms_name,
                    kind="rms_attitude_error_deg",
                    object_id=objective.object_id,
                    desired_quat_bn=objective.desired_quat_bn,
                )
            )
            if objective.max_final_attitude_error_deg is not None:
                criteria.append(
                    ControllerBenchPassCriterion(
                        metric=final_name,
                        op="<=",
                        value=objective.max_final_attitude_error_deg,
                    )
                )
            if objective.max_rms_attitude_error_deg is not None:
                criteria.append(
                    ControllerBenchPassCriterion(
                        metric=rms_name,
                        op="<=",
                        value=objective.max_rms_attitude_error_deg,
                    )
                )

        rate_name = _metric_name(prefix, "final_body_rate_norm_rad_s")
        metrics.append(
            ControllerBenchMetric(
                name=rate_name,
                kind="final_body_rate_norm_rad_s",
                object_id=objective.object_id,
            )
        )
        if objective.max_final_body_rate_norm_rad_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=rate_name,
                    op="<=",
                    value=objective.max_final_body_rate_norm_rad_s,
                )
            )

    elif objective.kind == "relative_rendezvous":
        if not objective.object_id or not objective.reference_object_id:
            raise ValueError("relative_rendezvous objective requires object_id and reference_object_id.")
        final_range_name = _metric_name(prefix, "final_relative_distance_km")
        rms_range_name = _metric_name(prefix, "rms_relative_distance_km")
        closest_name = _metric_name(prefix, "closest_approach_km")
        final_speed_name = _metric_name(prefix, "final_relative_speed_km_s")
        dv_name = _metric_name(prefix, "total_dv_m_s")
        fuel_name = _metric_name(prefix, "fuel_used_kg")
        metrics.extend(
            [
                ControllerBenchMetric(
                    name=final_range_name,
                    kind="final_relative_distance_km",
                    object_id=objective.object_id,
                    reference_object_id=objective.reference_object_id,
                ),
                ControllerBenchMetric(
                    name=rms_range_name,
                    kind="rms_relative_distance_km",
                    object_id=objective.object_id,
                    reference_object_id=objective.reference_object_id,
                ),
                ControllerBenchMetric(
                    name=closest_name,
                    kind="closest_approach_km",
                    object_id=objective.object_id,
                    reference_object_id=objective.reference_object_id,
                ),
                ControllerBenchMetric(
                    name=final_speed_name,
                    kind="final_relative_speed_km_s",
                    object_id=objective.object_id,
                    reference_object_id=objective.reference_object_id,
                ),
                ControllerBenchMetric(
                    name=dv_name,
                    kind="total_dv_m_s",
                    object_id=objective.object_id,
                ),
                ControllerBenchMetric(
                    name=fuel_name,
                    kind="fuel_used_kg",
                    object_id=objective.object_id,
                ),
            ]
        )
        if objective.keepout_radius_km is not None:
            keepout_name = _metric_name(prefix, "time_inside_keepout_s")
            metrics.append(
                ControllerBenchMetric(
                    name=keepout_name,
                    kind="time_inside_keepout_s",
                    object_id=objective.object_id,
                    reference_object_id=objective.reference_object_id,
                    keepout_radius_km=objective.keepout_radius_km,
                )
            )
            if objective.max_time_inside_keepout_s is not None:
                criteria.append(
                    ControllerBenchPassCriterion(
                        metric=keepout_name,
                        op="<=",
                        value=objective.max_time_inside_keepout_s,
                    )
                )
        if objective.max_final_relative_distance_km is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=final_range_name,
                    op="<=",
                    value=objective.max_final_relative_distance_km,
                )
            )
        if objective.max_rms_relative_distance_km is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=rms_range_name,
                    op="<=",
                    value=objective.max_rms_relative_distance_km,
                )
            )
        if objective.max_final_relative_speed_km_s is not None:
            criteria.append(
                ControllerBenchPassCriterion(
                    metric=final_speed_name,
                    op="<=",
                    value=objective.max_final_relative_speed_km_s,
                )
            )
        if objective.max_total_dv_m_s is not None:
            criteria.append(ControllerBenchPassCriterion(metric=dv_name, op="<=", value=objective.max_total_dv_m_s))
        if objective.max_fuel_used_kg is not None:
            criteria.append(ControllerBenchPassCriterion(metric=fuel_name, op="<=", value=objective.max_fuel_used_kg))
    else:
        raise ValueError(f"Unsupported controller bench objective kind: {objective.kind}")

    if objective.require_not_terminated_early:
        term_name = _metric_name(prefix, "terminated_early")
        metrics.append(
            ControllerBenchMetric(
                name=term_name,
                source_path="summary.terminated_early",
            )
        )
        criteria.append(ControllerBenchPassCriterion(metric=term_name, op="==", value=False))

    return tuple(metrics), tuple(criteria)


def _resolve_metric_specs(suite: ControllerBenchConfig, case: ControllerBenchCase) -> tuple[ControllerBenchMetric, ...]:
    metrics = list(case.metrics if case.metrics else suite.metrics)
    objectives = case.objectives if case.objectives else suite.objectives
    for objective in objectives:
        objective_metrics, _ = _objective_specifications(objective)
        metrics.extend(objective_metrics)
    return _dedupe_metrics(metrics)


def _resolve_pass_criteria(
    suite: ControllerBenchConfig,
    case: ControllerBenchCase,
    objectives: tuple[ControllerBenchObjective, ...],
) -> tuple[ControllerBenchPassCriterion, ...]:
    criteria = list(case.pass_criteria if case.pass_criteria else suite.pass_criteria)
    for objective in objectives:
        _, objective_criteria = _objective_specifications(objective)
        criteria.extend(objective_criteria)
    return tuple(criteria)


def _evaluate_objectives(
    metrics: dict[str, Any],
    objectives: tuple[ControllerBenchObjective, ...],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for objective in objectives:
        prefix = _slug(objective.name or objective.kind)
        objective_metrics, objective_criteria = _objective_specifications(objective)
        scoped_metrics = {metric.name: metrics.get(metric.name) for metric in objective_metrics}
        passed, failed = _evaluate_pass_criteria(scoped_metrics, objective_criteria)
        results.append(
            {
                "name": objective.name or objective.kind,
                "kind": objective.kind,
                "passed": bool(passed),
                "failed_criteria": failed,
                "metrics": scoped_metrics,
                "metric_prefix": prefix,
            }
        )
    return results


def _run_single_bench_case(
    suite: ControllerBenchConfig,
    case: ControllerBenchCase,
    variant: ControllerVariant,
) -> dict[str, Any]:
    scenario = load_simulation_yaml(case.config_path)
    if _analysis_study_type(scenario) != "single_run":
        raise ValueError(f"Controller bench case '{case.name}' must be a single-run scenario.")

    root = scenario.to_dict()
    run_outdir = suite.output_dir / case.name / variant.name
    _set_controller_pointer(root, suite.controller_target, variant.pointer)
    _apply_benchmark_output_overrides(root, run_outdir, suite)
    cfg = scenario_config_from_dict(root)
    payload = _run_single_config(cfg)
    payload["config_path"] = str(case.config_path.resolve())

    objectives = case.objectives if case.objectives else suite.objectives
    metric_specs = _resolve_metric_specs(suite, case)
    metrics = {metric.name: evaluate_metric(metric, payload) for metric in metric_specs}
    criteria = _resolve_pass_criteria(suite, case, objectives)
    passed, failed_criteria = _evaluate_pass_criteria(metrics, criteria)
    objective_results = _evaluate_objectives(metrics, objectives)
    artifacts = {
        "output_dir": str(run_outdir),
        "summary_json": str(run_outdir / "master_run_summary.json"),
    }
    if suite.save_run_payloads:
        artifacts["run_log_json"] = str(run_outdir / "master_run_log.json")
    linear_feedback_artifacts = write_linear_feedback_diagnostics(payload, suite.controller_target.object_id, run_outdir)
    if linear_feedback_artifacts:
        artifacts["linear_feedback"] = linear_feedback_artifacts
    return {
        "variant_name": variant.name,
        "case_name": case.name,
        "description": case.description,
        "metrics": metrics,
        "passed": bool(passed),
        "failed_criteria": failed_criteria,
        "objective_results": objective_results,
        "artifacts": artifacts,
        "config_path": str(case.config_path.resolve()),
    }


def _run_single_bench_case_worker(task: dict[str, Any]) -> dict[str, Any]:
    index = int(task.get("index", 0))
    suite = task["suite"]
    case = task["case"]
    variant = task["variant"]
    return {
        "index": index,
        "run": _run_single_bench_case(suite=suite, case=case, variant=variant),
    }


def run_controller_bench(config_path: str | Path, *, compare_names: list[str] | None = None) -> dict[str, Any]:
    suite = load_controller_bench_config(config_path)
    selected = list(suite.variants)
    if compare_names:
        wanted = {str(name) for name in compare_names}
        selected = [variant for variant in selected if variant.name in wanted]
        if not selected:
            raise ValueError("No matching controller variants selected.")

    scheduled_runs = [(case, variant) for case in suite.cases for variant in selected]
    runs_by_index: dict[int, dict[str, Any]] = {}
    use_tqdm = _should_use_tqdm()
    bench_bar = None
    plain_progress = None if use_tqdm else _make_plain_progress_reporter("Controller Bench")
    total_runs = len(scheduled_runs)
    parallel_requested = bool(suite.parallel_enabled and total_runs > 1)
    default_workers = max(1, (os.cpu_count() or 1) - 1)
    configured_workers = int(max(suite.parallel_workers, 0))
    parallel_workers = configured_workers if configured_workers > 0 else default_workers
    parallel_workers = max(1, min(parallel_workers, max(total_runs, 1)))
    parallel_active = False
    parallel_fallback_reason: str | None = None
    try:
        if use_tqdm and scheduled_runs:
            try:
                from tqdm.auto import tqdm  # type: ignore

                bench_bar = tqdm(
                    total=len(scheduled_runs),
                    desc="Controller Bench",
                    unit="run",
                    dynamic_ncols=True,
                )
            except Exception:
                bench_bar = None
                plain_progress = _make_plain_progress_reporter("Controller Bench")

        if parallel_requested:
            thread_env_prev = _set_parallel_worker_thread_limits(default_threads="1")
            try:
                tasks = [
                    {
                        "index": index,
                        "suite": suite,
                        "case": case,
                        "variant": variant,
                    }
                    for index, (case, variant) in enumerate(scheduled_runs)
                ]
                with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
                    fut_to_task = {ex.submit(_run_single_bench_case_worker, task): task for task in tasks}
                    pending = set(fut_to_task.keys())
                    parallel_active = True
                    completed_count = 0
                    while pending:
                        done_now, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                        for fut in done_now:
                            item = fut.result()
                            run = dict(item.get("run", {}) or {})
                            index = int(item.get("index", 0))
                            runs_by_index[index] = run
                            completed_count += 1
                            detail = f"{run.get('case_name', 'case')} :: {run.get('variant_name', 'variant')}"
                            if bench_bar is not None:
                                bench_bar.set_postfix_str(detail)
                                bench_bar.update(1)
                            elif plain_progress is not None:
                                plain_progress(completed_count, total_runs, detail)
            except (OSError, PermissionError, NotImplementedError, EOFError) as exc:
                parallel_active = False
                parallel_fallback_reason = f"{type(exc).__name__}: {exc}"
            finally:
                _restore_env_vars(thread_env_prev)

        if not parallel_active:
            completed_count = 0
            for index, (case, variant) in enumerate(scheduled_runs):
                detail = f"{case.name} :: {variant.name}"
                if bench_bar is not None:
                    bench_bar.set_postfix_str(detail)
                elif plain_progress is not None:
                    plain_progress(completed_count + 1, total_runs, detail)
                runs_by_index[index] = _run_single_bench_case(suite, case, variant)
                completed_count += 1
                if bench_bar is not None:
                    bench_bar.update(1)
    finally:
        if bench_bar is not None:
            bench_bar.close()
    runs = [runs_by_index[idx] for idx in range(total_runs)]

    variant_summaries: list[dict[str, Any]] = []
    for variant in selected:
        subset = [run for run in runs if run["variant_name"] == variant.name]
        metric_names: list[str] = []
        objective_names: list[str] = []
        for run in subset:
            for metric_name in dict(run.get("metrics", {}) or {}).keys():
                if metric_name not in metric_names:
                    metric_names.append(metric_name)
            for objective_result in list(run.get("objective_results", []) or []):
                objective_name = str(objective_result.get("name", "") or "").strip()
                if objective_name and objective_name not in objective_names:
                    objective_names.append(objective_name)
        metric_means: dict[str, float] = {}
        for metric_name in metric_names:
            values = []
            for run in subset:
                value = dict(run.get("metrics", {}) or {}).get(metric_name)
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value_f):
                    continue
                values.append(value_f)
            if values:
                metric_means[metric_name] = float(sum(values) / len(values))
        objective_pass_rates: dict[str, float] = {}
        for objective_name in objective_names:
            objective_hits = 0
            objective_passes = 0
            for run in subset:
                for objective_result in list(run.get("objective_results", []) or []):
                    if str(objective_result.get("name", "") or "").strip() != objective_name:
                        continue
                    objective_hits += 1
                    if bool(objective_result.get("passed", False)):
                        objective_passes += 1
            if objective_hits > 0:
                objective_pass_rates[objective_name] = float(objective_passes) / float(objective_hits)
        run_count = len(subset)
        passed_runs = sum(1 for run in subset if bool(run.get("passed", False)))
        variant_summaries.append(
            {
                "variant_name": variant.name,
                "description": variant.description,
                "run_count": run_count,
                "passed_runs": passed_runs,
                "pass_rate": (float(passed_runs) / float(run_count)) if run_count > 0 else 0.0,
                "metric_means": metric_means,
                "objective_pass_rates": objective_pass_rates,
            }
        )

    result = {
        "suite_name": suite.suite_name,
        "description": suite.description,
        "controller_target": {
            "object_id": suite.controller_target.object_id,
            "slot": suite.controller_target.slot,
        },
        "variants": [
            {
                "name": variant.name,
                "description": variant.description,
                "pointer": deepcopy(variant.pointer),
                "linear_system_summary": _controller_linear_system_summary(variant.pointer),
            }
            for variant in selected
        ],
        "cases": [
            {
                "name": case.name,
                "config_path": str(case.config_path),
                "description": case.description,
                "objectives": [
                    {
                        "name": objective.name or objective.kind,
                        "kind": objective.kind,
                    }
                    for objective in (case.objectives if case.objectives else suite.objectives)
                ],
            }
            for case in suite.cases
        ],
        "runs": runs,
        "variant_summaries": variant_summaries,
        "execution": {
            "parallel_enabled": bool(parallel_active),
            "parallel_requested": bool(parallel_requested),
            "parallel_workers": int(parallel_workers if parallel_active else 1),
            "parallel_fallback_reason": parallel_fallback_reason,
        },
        "plot_mode": suite.plot_mode,
        "leaderboards": {
            "relative_rendezvous": _build_relative_rendezvous_leaderboards(runs, selected),
        },
    }
    suite.output_dir.mkdir(parents=True, exist_ok=True)
    result["artifacts"] = {}
    plot_artifacts = render_controller_bench_visualizations(result, suite.output_dir, suite.plot_mode)
    if plot_artifacts:
        result["artifacts"].update(plot_artifacts)
    result["artifacts"].update(write_controller_bench_reports(result, suite.output_dir))
    return result
