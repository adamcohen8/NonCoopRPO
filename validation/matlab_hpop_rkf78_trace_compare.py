from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.config import load_simulation_yaml
from sim.dynamics.orbit.accelerations import OrbitContext, accel_two_body
from sim.dynamics.orbit.integrators import rkf78_stage_trace, rkf78_step
from sim.runtime_support import _build_orbit_propagator
from sim.single_run import _coerce_noninteractive_for_automation, _run_single_config
from sim.utils.io import write_json
from validation.matlab_hpop_bridge import (
    _agent_section,
    _build_single_run_cfg,
    _default_hpop_root,
    _default_object_id,
    _finite_max_norm,
    _force_model_from_cfg,
    _invoke_matlab_case,
    _satellite_properties,
    _write_case_input,
)
from sim.dynamics.orbit.epoch import julian_date_to_datetime


def _parse_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            parts = s.split()
            row: dict[str, Any] = {"kind": parts[0]}
            for item in parts[1:]:
                if "=" not in item:
                    continue
                k, v = item.split("=", 1)
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def _parse_csv_vector(value: str) -> list[float]:
    text = str(value).strip()
    if not text:
        return []
    return [float(item) for item in text.split(",") if item]


def _parse_stage_trace(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"meta": {}, "stages": []}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            parts = s.split()
            kind = parts[0]
            row: dict[str, Any] = {}
            for item in parts[1:]:
                if "=" not in item:
                    continue
                k, v = item.split("=", 1)
                if k in {"state", "deriv"}:
                    row[k] = _parse_csv_vector(v)
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            if kind == "meta":
                result["meta"] = row
            elif kind == "stage":
                result["stages"].append(row)
    return result


def _python_rkf78_trace(deriv_fn, *, t_s: float, x: np.ndarray, dt_s: float, tolerance: float, h_init: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    min_scale = 0.125
    max_scale = 4.0
    err_exponent = 1.0 / 7.0
    x_now = float(t_s)
    x_end = float(t_s + dt_s)
    h = float(h_init)
    last_interval = False
    if h > (x_end - x_now):
        h = x_end - x_now
        last_interval = True
    tol_per_unit = float(tolerance) / (x_end - x_now)
    y = np.array(x, dtype=float, copy=True)
    while x_now < x_end:
        scale = 1.0
        for _ in range(12):
            y_trial, err_vec = rkf78_step(deriv_fn, x_now, y, h)
            err = float(np.linalg.norm(err_vec))
            if err == 0.0:
                scale = max_scale
                rows.append({"kind": "attempt", "x": x_now, "h": h, "err": err, "yy": float("nan"), "scale": scale, "accepted": 1.0})
                break
            y_norm = float(np.linalg.norm(y))
            yy = tol_per_unit if y_norm == 0.0 else y_norm
            scale = 0.8 * (tol_per_unit * yy / err) ** err_exponent
            scale = min(max(scale, min_scale), max_scale)
            accepted = float(err < (tol_per_unit * yy))
            rows.append({"kind": "attempt", "x": x_now, "h": h, "err": err, "yy": yy, "scale": scale, "accepted": accepted})
            if accepted:
                break
            h *= scale
            if x_now + h > x_end:
                h = x_end - x_now
            elif x_now + h + 0.5 * h > x_end:
                h = 0.5 * h
        y = y_trial
        x_now += h
        rows.append({"kind": "accept", "x": x_now, "h_used": h, "h_next": h * scale})
        h *= scale
        if last_interval:
            break
        if x_now + h > x_end:
            last_interval = True
            h = x_end - x_now
        elif x_now + h + 0.5 * h > x_end:
            h = 0.5 * h
    return rows


def _python_stage_trace(deriv_fn, *, t_s: float, x: np.ndarray, dt_s: float) -> dict[str, Any]:
    stages = rkf78_stage_trace(deriv_fn, t_s, x, dt_s)
    return {
        "meta": {"x0": float(t_s), "h": float(dt_s)},
        "stages": [
            {
                "name": str(stage["name"]),
                "t": float(stage["t"]),
                "state": np.asarray(stage["x"], dtype=float).tolist(),
                "deriv": np.asarray(stage["k"], dtype=float).tolist(),
            }
            for stage in stages
        ],
    }


def run_trace_compare(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    hpop_root: str | Path | None = None,
    object_id: str | None = None,
    matlab_executable: str = "matlab",
    timeout_s: float | None = None,
) -> dict[str, Any]:
    cfg_path = Path(config_path).expanduser().resolve()
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    hpop_root_path = _default_hpop_root() if hpop_root is None else Path(hpop_root).expanduser().resolve()
    cfg = load_simulation_yaml(cfg_path)
    selected_object_id = str(object_id or _default_object_id(cfg))
    _agent_section(cfg, selected_object_id)

    sim_run_dir = outdir / "simulator_run"
    runtime_cfg = _coerce_noninteractive_for_automation(_build_single_run_cfg(cfg_path, sim_run_dir))
    payload = _run_single_config(runtime_cfg)
    time_s = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
    truth = np.array((payload.get("truth_by_object", {}) or {}).get(selected_object_id, []), dtype=float)
    if truth.ndim != 2 or truth.shape[0] < 1 or truth.shape[1] < 14:
        raise ValueError("Missing truth history.")
    thrust_hist = np.array((payload.get("applied_thrust_by_object", {}) or {}).get(selected_object_id, []), dtype=float)
    if thrust_hist.size > 0 and _finite_max_norm(thrust_hist.reshape(-1, 3)) > 1.0e-12:
        raise ValueError("Trace comparison currently supports passive orbit cases only.")

    truth0 = truth[0, :]
    props = _satellite_properties(cfg, selected_object_id, mass_kg=float(truth0[13]))
    force_model = _force_model_from_cfg(cfg)
    epoch_dt = julian_date_to_datetime(float(cfg.simulator.initial_jd_utc)).astimezone(timezone.utc)
    dt = float(time_s[1] - time_s[0])
    case_dir = outdir / "matlab_hpop_rkf78_trace_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    matlab_trace_path = case_dir / "matlab_rkf78_trace.txt"
    matlab_stage_trace_path = case_dir / "matlab_rkf78_stage_trace.txt"
    case_params = {
        "scenario_name": cfg.scenario_name,
        "object_id": selected_object_id,
        "epoch_year": epoch_dt.year,
        "epoch_month": epoch_dt.month,
        "epoch_day": epoch_dt.day,
        "epoch_hour": epoch_dt.hour,
        "epoch_minute": epoch_dt.minute,
        "epoch_second": float(epoch_dt.second + epoch_dt.microsecond * 1.0e-6),
        "initial_state_eci_m_m_s": (truth0[:6] * 1e3).tolist(),
        "step_s": dt,
        "duration_s": dt,
        "rkf78_trace_path": str(matlab_trace_path),
        "rkf78_stage_trace_path": str(matlab_stage_trace_path),
        **props,
        **force_model,
    }
    _write_case_input(case_dir, case_params)
    _invoke_matlab_case(case_dir=case_dir, hpop_root=hpop_root_path, matlab_executable=matlab_executable, timeout_s=timeout_s)
    matlab_trace = _parse_trace(matlab_trace_path)

    orbit_cfg = dict(runtime_cfg.simulator.dynamics.get("orbit", {}) or {})
    env = dict(runtime_cfg.simulator.environment or {})
    from sim.dynamics.orbit.spherical_harmonics import configure_spherical_harmonics_env
    env = configure_spherical_harmonics_env(env, orbit_cfg)
    env["jd_utc_start"] = float(runtime_cfg.simulator.initial_jd_utc)
    propagator = _build_orbit_propagator(runtime_cfg)
    x0 = np.array(truth0[:6], dtype=float)
    ctx = OrbitContext(mu_km3_s2=398600.4415, mass_kg=float(truth0[13]), area_m2=float(props["area_drag_m2"]), cd=float(props["cd"]), cr=float(props["cr"]))

    def deriv(t_local: float, x_local: np.ndarray) -> np.ndarray:
        dx = np.empty(6, dtype=float)
        dx[:3] = x_local[3:]
        a = accel_two_body(x_local[:3], ctx.mu_km3_s2)
        for plugin in propagator.plugins:
            a += plugin(t_local, x_local, env, ctx)
        dx[3:] = a
        return dx

    python_trace = _python_rkf78_trace(deriv, t_s=0.0, x=x0, dt_s=dt, tolerance=float(orbit_cfg.get("adaptive_rtol", 1e-10)), h_init=0.01)
    python_stage_trace = _python_stage_trace(deriv, t_s=0.0, x=x0, dt_s=0.01)
    matlab_stage_trace = _parse_stage_trace(matlab_stage_trace_path)
    result = {
        "config_path": str(cfg_path),
        "object_id": selected_object_id,
        "dt_s": dt,
        "python_trace": python_trace,
        "matlab_trace": matlab_trace,
        "python_stage_trace": python_stage_trace,
        "matlab_stage_trace": matlab_stage_trace,
    }
    outpath = outdir / "matlab_hpop_rkf78_trace_compare.json"
    write_json(str(outpath), result)
    result["result_path"] = str(outpath)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Python/MATLAB RKF78 internal step traces.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/validation_matlab_hpop_rkf78_trace_compare")
    parser.add_argument("--object-id", default="")
    parser.add_argument("--hpop-root", default="")
    parser.add_argument("--matlab-executable", default="matlab")
    parser.add_argument("--timeout-s", type=float, default=None)
    args = parser.parse_args(argv)
    result = run_trace_compare(
        config_path=args.config,
        output_dir=args.output_dir,
        hpop_root=(None if not str(args.hpop_root).strip() else str(args.hpop_root).strip()),
        object_id=(None if not str(args.object_id).strip() else str(args.object_id).strip()),
        matlab_executable=str(args.matlab_executable),
        timeout_s=args.timeout_s,
    )
    print(f"Object            : {result['object_id']}")
    print(f"Python trace rows : {len(result['python_trace'])}")
    print(f"MATLAB trace rows : {len(result['matlab_trace'])}")
    print(f"Python stage rows : {len((result.get('python_stage_trace') or {}).get('stages', []))}")
    print(f"MATLAB stage rows : {len((result.get('matlab_stage_trace') or {}).get('stages', []))}")
    print(f"Result path       : {result['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
