from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.config import load_simulation_yaml, validate_scenario_plugins
from sim.master_simulator import _closest_approach_from_run_payload
from sim.master_simulator import _coerce_noninteractive_for_automation, _run_single_config, run_master_simulation
from sim.utils.io import write_json


DEFAULT_TOLERANCE_TABLES: dict[str, dict[str, dict[str, Any]]] = {
    "leo": {
        "pos_err_rms_m": {"max": 5.0e4},
        "pos_err_max_m": {"max": 1.5e5},
        "vel_err_rms_mm_s": {"max": 2.0e4},
        "vel_err_max_mm_s": {"max": 5.0e4},
    },
    "high_leo": {
        "pos_err_rms_m": {"max": 7.5e4},
        "pos_err_max_m": {"max": 2.5e5},
        "vel_err_rms_mm_s": {"max": 3.0e4},
        "vel_err_max_mm_s": {"max": 7.5e4},
    },
    "cislunar": {
        "pos_err_rms_m": {"max": 5.0e5},
        "pos_err_max_m": {"max": 1.0e6},
        "vel_err_rms_mm_s": {"max": 2.0e5},
        "vel_err_max_mm_s": {"max": 5.0e5},
    },
}


def _default_hpop_root_dir() -> Path:
    return REPO_ROOT / "validation" / "High Precision Orbit Propagator_4-2" / "High Precision Orbit Propagator_4.2.2"


def _invoke_hpop_validation(**kwargs: Any) -> dict[str, str]:
    from validation.hpop_compare import run_validation as run_hpop_validation

    return run_hpop_validation(**kwargs)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    kind: str
    description: str = ""
    enabled: bool = True
    config_path: str | None = None
    hpop_root: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    envelope: str | None = None
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessSpec:
    suite_name: str
    output_dir: str
    tolerance_tables: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    benchmarks: list[BenchmarkSpec] = field(default_factory=list)


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(raw)


def _resolve_path(path_str: str | None, *, base_dir: Path) -> Path | None:
    if path_str is None:
        return None
    txt = str(path_str).strip()
    if not txt:
        return None
    p = Path(txt).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def load_harness_spec(path: str | Path) -> HarnessSpec:
    cfg_path = Path(path).expanduser().resolve()
    raw = _load_yaml_dict(cfg_path)
    benches_raw = raw.get("benchmarks", []) or []
    if not isinstance(benches_raw, list):
        raise ValueError("benchmarks must be a list.")
    benchmarks: list[BenchmarkSpec] = []
    for i, item in enumerate(benches_raw):
        if not isinstance(item, dict):
            raise ValueError(f"benchmarks[{i}] must be a mapping.")
        benchmarks.append(
            BenchmarkSpec(
                name=str(item.get("name", f"benchmark_{i+1}")),
                kind=str(item.get("kind", "")).strip().lower(),
                description=str(item.get("description", "")),
                enabled=bool(item.get("enabled", True)),
                config_path=item.get("config_path"),
                hpop_root=item.get("hpop_root"),
                params=dict(item.get("params", {}) or {}),
                envelope=str(item.get("envelope")).strip() if item.get("envelope") is not None else None,
                checks=dict(item.get("checks", {}) or {}),
            )
        )
    tolerance_tables = dict(DEFAULT_TOLERANCE_TABLES)
    user_tables = raw.get("tolerance_tables", {}) or {}
    if not isinstance(user_tables, dict):
        raise ValueError("tolerance_tables must be a mapping.")
    for name, table in user_tables.items():
        if not isinstance(table, dict):
            raise ValueError(f"tolerance_tables.{name} must be a mapping.")
        merged = dict(tolerance_tables.get(str(name), {}))
        for metric, rule in table.items():
            if not isinstance(rule, dict):
                raise ValueError(f"tolerance_tables.{name}.{metric} must be a mapping.")
            merged[str(metric)] = dict(rule)
        tolerance_tables[str(name)] = merged
    return HarnessSpec(
        suite_name=str(raw.get("suite_name", cfg_path.stem)),
        output_dir=str(raw.get("output_dir", f"outputs/{cfg_path.stem}")),
        tolerance_tables=tolerance_tables,
        benchmarks=benchmarks,
    )


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        try:
            if any(ch in s for ch in (".", "e", "E")):
                return float(s)
            return int(s)
        except ValueError:
            return value
    return value


def _extract_metric(payload: Any, path: str) -> Any:
    cur = payload
    for token in str(path).split("."):
        if "[" in token and token.endswith("]"):
            key, idx_txt = token[:-1].split("[", 1)
            if key:
                if not isinstance(cur, dict):
                    raise KeyError(path)
                cur = cur[key]
            idx = int(idx_txt)
            if not isinstance(cur, list):
                raise KeyError(path)
            cur = cur[idx]
            continue
        if not isinstance(cur, dict):
            raise KeyError(path)
        cur = cur[token]
    return _coerce_scalar(cur)


def _evaluate_rule(actual: Any, rule: dict[str, Any]) -> tuple[bool, str]:
    checks: list[tuple[bool, str]] = []
    if "equals" in rule:
        expected = _coerce_scalar(rule["equals"])
        checks.append((actual == expected, f"equals {expected!r}"))
    if "min" in rule:
        expected = float(rule["min"])
        checks.append((float(actual) >= expected, f">= {expected}"))
    if "max" in rule:
        expected = float(rule["max"])
        checks.append((float(actual) <= expected, f"<= {expected}"))
    if "truthy" in rule:
        expected = bool(rule["truthy"])
        checks.append((bool(actual) is expected, f"truthy is {expected}"))
    if "contains" in rule:
        target = rule["contains"]
        checks.append((target in actual, f"contains {target!r}"))
    ok = all(item[0] for item in checks) if checks else True
    summary = ", ".join(item[1] for item in checks) if checks else "no-op"
    return ok, summary


def _merge_checks(spec: BenchmarkSpec, tables: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    if spec.envelope:
        merged.update({k: dict(v) for k, v in dict(tables.get(spec.envelope, {})).items()})
    merged.update({k: dict(v) for k, v in spec.checks.items()})
    return merged


def _write_temp_sim_config(source_path: Path, benchmark_output_dir: Path) -> Path:
    raw = _load_yaml_dict(source_path)
    outputs = dict(raw.get("outputs", {}) or {})
    outputs["output_dir"] = str(benchmark_output_dir)
    mode = str(outputs.get("mode", "save")).strip().lower()
    if mode == "interactive":
        outputs["mode"] = "save"
    mc_outputs = dict(outputs.get("monte_carlo", {}) or {})
    mc_outputs["save_aggregate_summary"] = True
    outputs["monte_carlo"] = mc_outputs
    raw["outputs"] = outputs
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=f"{source_path.stem}_harness_",
        dir=str(benchmark_output_dir),
        delete=False,
        encoding="utf-8",
    )
    with tmp:
        yaml.safe_dump(raw, tmp, sort_keys=False)
    return Path(tmp.name)


def _run_plugin_validation(config_path: Path) -> dict[str, Any]:
    cfg = load_simulation_yaml(config_path)
    errors = validate_scenario_plugins(cfg)
    return {
        "valid": len(errors) == 0,
        "error_count": int(len(errors)),
        "errors": list(errors),
    }


def _run_simulation_validation(config_path: Path, benchmark_output_dir: Path) -> dict[str, Any]:
    temp_cfg = _write_temp_sim_config(config_path, benchmark_output_dir)
    prev = os.environ.get("SIM_AUTOMATION")
    os.environ["SIM_AUTOMATION"] = "1"
    try:
        cfg = _coerce_noninteractive_for_automation(load_simulation_yaml(temp_cfg))
        if not bool(cfg.monte_carlo.enabled):
            run_payload = _run_single_config(cfg)
            return {
                "config_path": str(temp_cfg.resolve()),
                "scenario_name": cfg.scenario_name,
                "monte_carlo": {"enabled": False},
                "run": dict(run_payload.get("summary", {}) or {}),
                "derived": _build_single_run_derived_metrics(run_payload),
                "full_run_payload_available": True,
            }
        return run_master_simulation(temp_cfg)
    finally:
        if prev is None:
            os.environ.pop("SIM_AUTOMATION", None)
        else:
            os.environ["SIM_AUTOMATION"] = prev


def _finite_ratio(values: np.ndarray) -> float:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.isfinite(arr)))


def _build_single_run_derived_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    run = dict(payload.get("run", {}) or {})
    thrust_stats = dict(run.get("thrust_stats", {}) or {})
    guardrail_stats = dict(run.get("attitude_guardrail_stats", {}) or {})
    knowledge = dict(payload.get("knowledge_by_observer", {}) or {})
    knowledge_finite_fraction: dict[str, dict[str, float]] = {}
    for obs, by_target in knowledge.items():
        obs_map: dict[str, float] = {}
        for target, arr in dict(by_target or {}).items():
            obs_map[str(target)] = _finite_ratio(np.array(arr, dtype=float))
        knowledge_finite_fraction[str(obs)] = obs_map
    total_dv_m_s = float(
        np.sum(np.array([float(dict(v or {}).get("total_dv_m_s", 0.0)) for v in thrust_stats.values()], dtype=float))
    ) if thrust_stats else 0.0
    return {
        "closest_approach_km": _closest_approach_from_run_payload(payload),
        "total_dv_m_s": total_dv_m_s,
        "guardrail_events": int(sum(int(v) for v in guardrail_stats.values())) if guardrail_stats else 0,
        "knowledge_finite_fraction": knowledge_finite_fraction,
    }


def _run_hpop_benchmark(spec: BenchmarkSpec, benchmark_output_dir: Path, base_dir: Path) -> dict[str, Any]:
    params = dict(spec.params)
    result = _invoke_hpop_validation(
        hpop_root=_resolve_path(spec.hpop_root, base_dir=base_dir) or _default_hpop_root_dir(),
        model=str(params.get("model", "sh8x8")),
        validation_dt_s=float(params.get("validation_dt_s", params.get("dt_s", 1.0))),
        validation_duration_s=float(params.get("validation_duration_s", params.get("duration_s", 150.0 * 60.0))),
        atmosphere_model=str(params.get("atmosphere_model", "exponential")),
        density_override_kg_m3=(
            float(params["density_override_kg_m3"]) if params.get("density_override_kg_m3") is not None else None
        ),
        sun_dir_eci=params.get("sun_dir_eci"),
        plot_mode=str(params.get("plot_mode", "save")),
        output_dir=benchmark_output_dir,
    )
    return {k: _coerce_scalar(v) for k, v in result.items()}


def _evaluate_checks(payload: dict[str, Any], checks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for metric_path, rule in sorted(checks.items()):
        try:
            actual = _extract_metric(payload, metric_path)
            passed, expectation = _evaluate_rule(actual, rule)
            evaluations.append(
                {
                    "metric": metric_path,
                    "passed": bool(passed),
                    "actual": actual,
                    "rule": dict(rule),
                    "expectation": expectation,
                }
            )
        except Exception as exc:
            evaluations.append(
                {
                    "metric": metric_path,
                    "passed": False,
                    "actual": None,
                    "rule": dict(rule),
                    "expectation": "metric available",
                    "error": str(exc),
                }
            )
    return evaluations


def run_harness(spec: HarnessSpec, *, base_dir: Path) -> dict[str, Any]:
    outdir = _resolve_path(spec.output_dir, base_dir=base_dir) or (base_dir / "outputs" / spec.suite_name)
    outdir.mkdir(parents=True, exist_ok=True)
    benchmark_reports: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    for spec_item in spec.benchmarks:
        if not spec_item.enabled:
            benchmark_reports.append(
                {
                    "name": spec_item.name,
                    "kind": spec_item.kind,
                    "enabled": False,
                    "passed": True,
                    "skipped": True,
                }
            )
            continue
        bench_out = outdir / spec_item.name
        bench_out.mkdir(parents=True, exist_ok=True)
        merged_checks = _merge_checks(spec_item, spec.tolerance_tables)
        report: dict[str, Any] = {
            "name": spec_item.name,
            "kind": spec_item.kind,
            "description": spec_item.description,
            "enabled": True,
            "envelope": spec_item.envelope,
            "output_dir": str(bench_out),
            "checks": merged_checks,
        }
        try:
            if spec_item.kind == "plugin_validation":
                cfg_path = _resolve_path(spec_item.config_path, base_dir=base_dir)
                if cfg_path is None:
                    raise ValueError("plugin_validation benchmark requires config_path.")
                payload = _run_plugin_validation(cfg_path)
            elif spec_item.kind == "simulation":
                cfg_path = _resolve_path(spec_item.config_path, base_dir=base_dir)
                if cfg_path is None:
                    raise ValueError("simulation benchmark requires config_path.")
                payload = _run_simulation_validation(cfg_path, bench_out)
            elif spec_item.kind == "hpop":
                payload = _run_hpop_benchmark(spec_item, bench_out, base_dir)
            else:
                raise ValueError(f"Unsupported benchmark kind: {spec_item.kind}")
            evaluations = _evaluate_checks(payload, merged_checks)
            report["evaluations"] = evaluations
            report["metrics"] = payload
            report["passed"] = all(bool(e.get("passed", False)) for e in evaluations) if evaluations else True
        except Exception as exc:
            report["passed"] = False
            report["error"] = str(exc)
            report["traceback"] = traceback.format_exc()
        benchmark_reports.append(report)

    finished = datetime.now(timezone.utc)
    summary = {
        "suite_name": spec.suite_name,
        "generated_utc": finished.isoformat(),
        "duration_s": float((finished - started).total_seconds()),
        "output_dir": str(outdir),
        "passed": all(bool(item.get("passed", False)) for item in benchmark_reports),
        "benchmarks_total": int(len(benchmark_reports)),
        "benchmarks_passed": int(sum(1 for item in benchmark_reports if bool(item.get("passed", False)))),
        "benchmarks_failed": int(sum(1 for item in benchmark_reports if not bool(item.get("passed", False)))),
        "benchmarks": benchmark_reports,
    }
    json_path = outdir / "validation_harness_report.json"
    md_path = outdir / "validation_harness_report.md"
    write_json(str(json_path), summary)
    md_path.write_text(_build_markdown_report(summary), encoding="utf-8")
    summary["artifacts"] = {
        "report_json": str(json_path),
        "report_md": str(md_path),
    }
    write_json(str(json_path), summary)
    return summary


def _build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Automated Validation Harness Report",
        "",
        f"- Suite: {report.get('suite_name', 'unknown')}",
        f"- Generated: {report.get('generated_utc', '')}",
        f"- Overall Pass: {'YES' if bool(report.get('passed', False)) else 'NO'}",
        f"- Benchmarks Passed: {int(report.get('benchmarks_passed', 0))}/{int(report.get('benchmarks_total', 0))}",
        "",
    ]
    for bench in list(report.get("benchmarks", []) or []):
        lines.append(f"## {bench.get('name', 'unnamed')}")
        lines.append("")
        lines.append(f"- Kind: {bench.get('kind', '')}")
        lines.append(f"- Pass: {'YES' if bool(bench.get('passed', False)) else 'NO'}")
        if bench.get("description"):
            lines.append(f"- Description: {bench.get('description')}")
        if bench.get("output_dir"):
            lines.append(f"- Output Dir: {bench.get('output_dir')}")
        if bench.get("error"):
            lines.append(f"- Error: {bench.get('error')}")
            lines.append("")
            continue
        evals = list(bench.get("evaluations", []) or [])
        if evals:
            lines.append("")
            lines.append("| Metric | Actual | Rule | Pass |")
            lines.append("| --- | ---: | --- | --- |")
            for row in evals:
                lines.append(
                    f"| {row.get('metric', '')} | {row.get('actual', '')} | {row.get('expectation', '')} | "
                    f"{'YES' if bool(row.get('passed', False)) else 'NO'} |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _default_harness_spec() -> HarnessSpec:
    return HarnessSpec(
        suite_name="validation_smoke",
        output_dir="outputs/validation_smoke",
        tolerance_tables=dict(DEFAULT_TOLERANCE_TABLES),
        benchmarks=[
            BenchmarkSpec(
                name="plugin_validation_smoke",
                kind="plugin_validation",
                config_path="configs/automation_smoke.yaml",
                checks={"valid": {"equals": True}, "error_count": {"equals": 0}},
            ),
            BenchmarkSpec(
                name="simulation_smoke",
                kind="simulation",
                config_path="configs/automation_smoke.yaml",
                checks={
                    "monte_carlo.enabled": {"equals": False},
                    "run.terminated_early": {"equals": False},
                    "run.duration_s": {"max": 120.0},
                },
            ),
            BenchmarkSpec(
                name="hpop_two_body_leo",
                kind="hpop",
                envelope="leo",
                hpop_root=str(_default_hpop_root_dir()),
                params={"model": "two_body", "plot_mode": "save", "duration_s": 600.0, "dt_s": 1.0},
            ),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated validation harness for benchmark suites and tolerance gates.")
    parser.add_argument(
        "--spec",
        type=str,
        default="",
        help="Path to harness YAML spec. If omitted, runs the built-in smoke suite.",
    )
    args = parser.parse_args()

    if str(args.spec).strip():
        spec_path = Path(args.spec).expanduser().resolve()
        spec = load_harness_spec(spec_path)
        base_dir = spec_path.parent
    else:
        spec = _default_harness_spec()
        base_dir = REPO_ROOT
    report = run_harness(spec, base_dir=base_dir)
    print(f"Suite          : {report['suite_name']}")
    print(f"Overall pass   : {report['passed']}")
    print(f"Benchmarks     : {report['benchmarks_passed']}/{report['benchmarks_total']} passed")
    print(f"JSON report    : {report['artifacts']['report_json']}")
    print(f"Markdown report: {report['artifacts']['report_md']}")
    return 0 if bool(report.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
