from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from sim.config import load_simulation_yaml, scenario_config_from_dict
from sim.controller_lab.metrics import evaluate_metric
from sim.controller_lab.models import (
    ControllerBenchCase,
    ControllerBenchConfig,
    ControllerBenchMetric,
    ControllerBenchPassCriterion,
    ControllerBenchTarget,
    ControllerVariant,
)
from sim.controller_lab.reporting import write_controller_bench_reports
from sim.master_simulator import _analysis_study_type, _run_single_config


def _as_dict(value: Any, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping/object.")
    return dict(value)


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
    )


def _parse_pass_criterion(raw: Any) -> ControllerBenchPassCriterion:
    d = _as_dict(raw, "pass_criterion")
    metric = str(d.get("metric", "") or "").strip()
    op = str(d.get("op", "") or "").strip()
    if not metric or not op:
        raise ValueError("pass criteria require non-empty 'metric' and 'op'.")
    return ControllerBenchPassCriterion(metric=metric, op=op, value=d.get("value"))


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
        d = _as_dict(entry, "variants[*]")
        name = str(d.get("name", "") or "").strip()
        pointer = _as_dict(d.get("controller"), "variants[*].controller")
        if not name:
            raise ValueError("variants[*].name must be non-empty.")
        variants.append(
            ControllerVariant(
                name=name,
                pointer=pointer,
                description=str(d.get("description", "") or ""),
            )
        )

    cases: list[ControllerBenchCase] = []
    for entry in list(raw.get("cases", []) or []):
        d = _as_dict(entry, "cases[*]")
        name = str(d.get("name", "") or "").strip()
        cfg_ref = str(d.get("config_path", "") or "").strip()
        if not name or not cfg_ref:
            raise ValueError("cases[*] require non-empty 'name' and 'config_path'.")
        case_metrics = tuple(_parse_metric(m) for m in list(d.get("metrics", []) or []))
        pass_criteria = tuple(_parse_pass_criterion(p) for p in list(d.get("pass_criteria", []) or []))
        cases.append(
            ControllerBenchCase(
                name=name,
                config_path=(cfg_path.parent / cfg_ref).resolve(),
                description=str(d.get("description", "") or ""),
                metrics=case_metrics,
                pass_criteria=pass_criteria,
            )
        )

    metrics = tuple(_parse_metric(m) for m in list(raw.get("metrics", []) or []))
    pass_criteria = tuple(_parse_pass_criterion(p) for p in list(raw.get("pass_criteria", []) or []))
    output_dir = Path(str(raw.get("output_dir", f"outputs/controller_bench/{raw.get('suite_name', 'suite')}")))
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    return ControllerBenchConfig(
        suite_name=str(raw.get("suite_name", cfg_path.stem) or cfg_path.stem),
        description=str(raw.get("description", "") or ""),
        output_dir=output_dir,
        controller_target=target,
        variants=tuple(variants),
        cases=tuple(cases),
        metrics=metrics,
        pass_criteria=pass_criteria,
        save_run_payloads=bool(raw.get("save_run_payloads", True)),
        disable_plots=bool(raw.get("disable_plots", True)),
        disable_animations=bool(raw.get("disable_animations", True)),
        print_individual_run_summaries=bool(raw.get("print_individual_run_summaries", False)),
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

    metric_specs = case.metrics if case.metrics else suite.metrics
    metrics = {metric.name: evaluate_metric(metric, payload) for metric in metric_specs}
    criteria = case.pass_criteria if case.pass_criteria else suite.pass_criteria
    passed, failed_criteria = _evaluate_pass_criteria(metrics, criteria)
    artifacts = {
        "output_dir": str(run_outdir),
        "summary_json": str(run_outdir / "master_run_summary.json"),
        "run_log_json": str(run_outdir / "master_run_log.json"),
    }
    return {
        "variant_name": variant.name,
        "case_name": case.name,
        "description": case.description,
        "metrics": metrics,
        "passed": bool(passed),
        "failed_criteria": failed_criteria,
        "artifacts": artifacts,
        "config_path": str(case.config_path.resolve()),
    }


def run_controller_bench(config_path: str | Path, *, compare_names: list[str] | None = None) -> dict[str, Any]:
    suite = load_controller_bench_config(config_path)
    selected = list(suite.variants)
    if compare_names:
        wanted = {str(name) for name in compare_names}
        selected = [variant for variant in selected if variant.name in wanted]
        if not selected:
            raise ValueError("No matching controller variants selected.")

    runs: list[dict[str, Any]] = []
    for case in suite.cases:
        for variant in selected:
            runs.append(_run_single_bench_case(suite, case, variant))

    variant_summaries: list[dict[str, Any]] = []
    for variant in selected:
        subset = [run for run in runs if run["variant_name"] == variant.name]
        metric_names: list[str] = []
        for run in subset:
            for metric_name in dict(run.get("metrics", {}) or {}).keys():
                if metric_name not in metric_names:
                    metric_names.append(metric_name)
        metric_means: dict[str, float] = {}
        for metric_name in metric_names:
            values = []
            for run in subset:
                value = dict(run.get("metrics", {}) or {}).get(metric_name)
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
            if values:
                metric_means[metric_name] = float(sum(values) / len(values))
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
            }
            for variant in selected
        ],
        "cases": [
            {
                "name": case.name,
                "config_path": str(case.config_path),
                "description": case.description,
            }
            for case in suite.cases
        ],
        "runs": runs,
        "variant_summaries": variant_summaries,
    }
    result["artifacts"] = write_controller_bench_reports(result, suite.output_dir)
    return result
