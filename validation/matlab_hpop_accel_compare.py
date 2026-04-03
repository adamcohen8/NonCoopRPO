from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.config import load_simulation_yaml
from sim.dynamics.orbit.accelerations import OrbitContext, accel_two_body
from sim.dynamics.orbit.frames import eci_to_ecef_rotation_hpop_like, eci_to_ecef_rotation
from sim.dynamics.orbit.spherical_harmonics import configure_spherical_harmonics_env
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
    _invoke_matlab_ephemeris_case,
    _invoke_matlab_case,
    _matlab_quote,
    _parse_ephemeris_export,
    _shared_ephemeris_bodies_from_cfg,
    _shared_ephemeris_step_s_from_cfg,
    _satellite_properties,
    _write_case_input,
    _write_ephemeris_case_input,
)
from sim.dynamics.orbit.epoch import julian_date_to_datetime


def _parse_accel_result(path: Path) -> np.ndarray:
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or "=" not in s:
                continue
            key, value = s.split("=", 1)
            values[key.strip()] = float(value.strip())
    return np.array(
        [
            values["ax_m_s2"],
            values["ay_m_s2"],
            values["az_m_s2"],
        ],
        dtype=float,
    )


def _parse_accel_debug(path: Path) -> dict[str, Any]:
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or "=" not in s:
                continue
            key, value = s.split("=", 1)
            values[key.strip()] = float(value.strip())
    e = np.array(
        [[values[f"E_{r}_{c}"] for c in range(1, 4)] for r in range(1, 4)],
        dtype=float,
    )
    return {
        "total_accel_m_s2": np.array([values["ax_m_s2"], values["ay_m_s2"], values["az_m_s2"]], dtype=float),
        "harmonic_accel_m_s2": np.array(
            [values["ax_harmonic_m_s2"], values["ay_harmonic_m_s2"], values["az_harmonic_m_s2"]],
            dtype=float,
        ),
        "harmonic_accel_bf_m_s2": np.array(
            [values["ax_bf_harmonic_m_s2"], values["ay_bf_harmonic_m_s2"], values["az_bf_harmonic_m_s2"]],
            dtype=float,
        ),
        "r_bf_m": np.array([values["rx_bf_m"], values["ry_bf_m"], values["rz_bf_m"]], dtype=float),
        "E": e,
    }


def _invoke_matlab_accel_case(
    *,
    case_dir: Path,
    hpop_root: Path,
    matlab_executable: str,
    timeout_s: float | None,
) -> dict[str, str]:
    runner_path = (REPO_ROOT / "validation" / "run_hpop_accel_case.m").resolve()
    batch_expr = (
        f"addpath('{_matlab_quote(str(runner_path.parent))}'); "
        f"run_hpop_accel_case('{_matlab_quote(str(case_dir))}', '{_matlab_quote(str(hpop_root))}');"
    )
    from validation.matlab_hpop_bridge import _resolve_matlab_executable

    matlab_executable_resolved = _resolve_matlab_executable(matlab_executable)
    cmd = [matlab_executable_resolved, "-batch", batch_expr]
    completed = subprocess.run(
        cmd,
        cwd=str(hpop_root),
        check=False,
        capture_output=True,
        text=True,
        timeout=None if timeout_s is None else float(timeout_s),
    )
    stdout_path = case_dir / "matlab_accel_stdout.txt"
    stderr_path = case_dir / "matlab_accel_stderr.txt"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"MATLAB HPOP acceleration run failed with exit code {completed.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )
    return {
        "matlab_stdout_path": str(stdout_path),
        "matlab_stderr_path": str(stderr_path),
        "matlab_executable": matlab_executable_resolved,
        "matlab_command": " ".join(cmd),
    }


def run_matlab_hpop_accel_compare(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    hpop_root: str | Path | None = None,
    object_id: str | None = None,
    matlab_executable: str = "matlab",
    timeout_s: float | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    cfg_path = Path(config_path).expanduser().resolve()
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    hpop_root_path = _default_hpop_root() if hpop_root is None else Path(hpop_root).expanduser().resolve()
    cfg = load_simulation_yaml(cfg_path)
    selected_object_id = str(object_id or _default_object_id(cfg))
    _agent_section(cfg, selected_object_id)
    if cfg.simulator.initial_jd_utc is None:
        raise ValueError("matlab_hpop acceleration comparison requires simulator.initial_jd_utc to be set.")

    sim_run_dir = outdir / "simulator_run"
    shared_ephemeris_bodies = _shared_ephemeris_bodies_from_cfg(cfg)
    shared_ephemeris_step_s = _shared_ephemeris_step_s_from_cfg(cfg)
    environment_overrides: dict[str, Any] = {}
    shared_ephemeris_artifacts: dict[str, Any] = {}
    if shared_ephemeris_bodies:
        epoch_dt = julian_date_to_datetime(float(cfg.simulator.initial_jd_utc)).astimezone(timezone.utc)
        ephemeris_case_dir = outdir / "matlab_hpop_ephemeris_case"
        ephemeris_case_dir.mkdir(parents=True, exist_ok=True)
        ephemeris_case_params = {
            "epoch_year": epoch_dt.year,
            "epoch_month": epoch_dt.month,
            "epoch_day": epoch_dt.day,
            "epoch_hour": epoch_dt.hour,
            "epoch_minute": epoch_dt.minute,
            "epoch_second": float(epoch_dt.second + epoch_dt.microsecond * 1.0e-6),
            "step_s": float(shared_ephemeris_step_s or cfg.simulator.dt_s),
            "duration_s": float(cfg.simulator.duration_s),
            "export_bodies": shared_ephemeris_bodies,
        }
        ephemeris_case_input_path = _write_ephemeris_case_input(ephemeris_case_dir, ephemeris_case_params)
        shared_ephemeris_artifacts["ephemeris_case_input_path"] = str(ephemeris_case_input_path)
        shared_ephemeris_artifacts["shared_ephemeris_bodies"] = list(shared_ephemeris_bodies)
        shared_ephemeris_artifacts["shared_ephemeris_step_s"] = float(ephemeris_case_params["step_s"])
        shared_ephemeris_artifacts.update(
            _invoke_matlab_ephemeris_case(
                case_dir=ephemeris_case_dir,
                hpop_root=hpop_root_path,
                matlab_executable=matlab_executable,
                timeout_s=timeout_s,
            )
        )
        ephemeris_export_path = ephemeris_case_dir / "BodyEphemeris.txt"
        ephemeris_export = _parse_ephemeris_export(ephemeris_export_path)
        shared_ephemeris_artifacts["ephemeris_export_path"] = str(ephemeris_export_path)
        if "moon" in shared_ephemeris_bodies:
            environment_overrides["moon_ephemeris_time_s"] = ephemeris_export["time_s"].tolist()
            environment_overrides["moon_ephemeris_eci_km"] = ephemeris_export["moon_eci_km"].tolist()
        if "sun" in shared_ephemeris_bodies:
            environment_overrides["sun_ephemeris_time_s"] = ephemeris_export["time_s"].tolist()
            environment_overrides["sun_ephemeris_eci_km"] = ephemeris_export["sun_eci_km"].tolist()

    runtime_cfg = _build_single_run_cfg(cfg_path, sim_run_dir, environment_overrides=environment_overrides)
    runtime_cfg = _coerce_noninteractive_for_automation(runtime_cfg)
    payload = _run_single_config(runtime_cfg)

    time_s = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
    truth_by_object = dict(payload.get("truth_by_object", {}) or {})
    if selected_object_id not in truth_by_object:
        raise ValueError(f"Selected object_id '{selected_object_id}' was not present in the simulation payload.")
    truth = np.array(truth_by_object[selected_object_id], dtype=float)
    if truth.ndim != 2 or truth.shape[1] < 14 or truth.shape[0] != time_s.size:
        raise ValueError("Selected object truth history has an unexpected shape.")
    if time_s.size < 1:
        raise ValueError("Simulation output must contain at least one sample.")
    if int(sample_index) < 0 or int(sample_index) >= time_s.size:
        raise ValueError(f"sample_index {sample_index} is out of range for {time_s.size} samples.")

    thrust_hist = np.array((payload.get("applied_thrust_by_object", {}) or {}).get(selected_object_id, []), dtype=float)
    if thrust_hist.size > 0 and _finite_max_norm(thrust_hist.reshape(-1, 3)) > 1.0e-12:
        raise ValueError(
            f"Selected object '{selected_object_id}' has non-zero applied thrust. "
            "matlab_hpop acceleration comparison currently supports passive orbit cases only."
        )

    sample_idx = int(sample_index)
    sample_t_s = float(time_s[sample_idx])
    truth0 = truth[sample_idx, :]
    props = _satellite_properties(cfg, selected_object_id, mass_kg=float(truth0[13]))
    force_model = _force_model_from_cfg(cfg)
    epoch_dt = julian_date_to_datetime(float(cfg.simulator.initial_jd_utc) + sample_t_s / 86400.0).astimezone(timezone.utc)

    case_dir = outdir / "matlab_hpop_accel_case"
    case_dir.mkdir(parents=True, exist_ok=True)
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
        "step_s": float(cfg.simulator.dt_s),
        "duration_s": 0.0,
        **props,
        **force_model,
    }
    case_input_path = _write_case_input(case_dir, case_params)
    manifest_path = case_dir / "hpop_accel_case_manifest.json"
    write_json(str(manifest_path), {"config_path": str(cfg_path), "case_params": case_params})

    matlab_artifacts = _invoke_matlab_accel_case(
        case_dir=case_dir,
        hpop_root=hpop_root_path,
        matlab_executable=matlab_executable,
        timeout_s=timeout_s,
    )
    matlab_debug = _parse_accel_debug(case_dir / "hpop_accel_result.txt")
    matlab_accel_m_s2 = matlab_debug["total_accel_m_s2"]

    orbit_cfg = dict(runtime_cfg.simulator.dynamics.get("orbit", {}) or {})
    env = configure_spherical_harmonics_env(dict(runtime_cfg.simulator.environment or {}), orbit_cfg)
    env["jd_utc_start"] = float(runtime_cfg.simulator.initial_jd_utc)
    propagator = _build_orbit_propagator(runtime_cfg)
    x0 = np.array(truth0[:6], dtype=float)
    ctx = OrbitContext(
        mu_km3_s2=runtime_cfg.simulator.mu_km3_s2 if hasattr(runtime_cfg.simulator, "mu_km3_s2") else 398600.4415,
        mass_kg=float(truth0[13]),
        area_m2=float(props["area_drag_m2"]),
        cd=float(props["cd"]),
        cr=float(props["cr"]),
    )
    python_accel_km_s2 = accel_two_body(x0[:3], ctx.mu_km3_s2)
    python_harmonic_accel_km_s2 = np.zeros(3, dtype=float)
    for plugin in propagator.plugins:
        contrib = plugin(sample_t_s, x0, env, ctx)
        python_accel_km_s2 += contrib
        if getattr(plugin, "__name__", "") == "spherical_harmonics_plugin":
            python_harmonic_accel_km_s2 += contrib
    python_accel_m_s2 = python_accel_km_s2 * 1e3
    python_harmonic_accel_m_s2 = python_harmonic_accel_km_s2 * 1e3
    two_body_accel_m_s2 = accel_two_body(x0[:3], ctx.mu_km3_s2) * 1e3

    frame_model = str(env.get("spherical_harmonics_frame_model", "simple"))
    eop_path = env.get("spherical_harmonics_eop_path")
    jd_utc_start = float(env["jd_utc_start"])
    if frame_model == "hpop_like":
        E_py = eci_to_ecef_rotation_hpop_like(
            sample_t_s,
            jd_utc_start=jd_utc_start,
            eop_path=None if eop_path is None else str(eop_path),
        )
    else:
        E_py = eci_to_ecef_rotation(sample_t_s, jd_utc_start=jd_utc_start)
    r_bf_py_m = (E_py @ x0[:3]) * 1e3
    a_harmonic_bf_py_m_s2 = E_py @ python_harmonic_accel_m_s2
    matlab_harmonic_perturbation_m_s2 = matlab_debug["harmonic_accel_m_s2"] - two_body_accel_m_s2
    matlab_harmonic_perturbation_bf_m_s2 = matlab_debug["harmonic_accel_bf_m_s2"] - (matlab_debug["E"] @ two_body_accel_m_s2)

    diff_m_s2 = python_accel_m_s2 - matlab_accel_m_s2
    diff_harmonic_m_s2 = python_harmonic_accel_m_s2 - matlab_harmonic_perturbation_m_s2
    diff_r_bf_m = r_bf_py_m - matlab_debug["r_bf_m"]
    diff_e = E_py - matlab_debug["E"]
    diff_a_bf_m_s2 = a_harmonic_bf_py_m_s2 - matlab_harmonic_perturbation_bf_m_s2
    result = {
        "config_path": str(cfg_path),
        "scenario_name": cfg.scenario_name,
        "object_id": selected_object_id,
        "sample_index": sample_idx,
        "sample_t_s": sample_t_s,
        "case_dir": str(case_dir),
        "case_input_path": str(case_input_path),
        "case_manifest_path": str(manifest_path),
        "matlab_accel_m_s2": matlab_accel_m_s2.tolist(),
        "python_accel_m_s2": python_accel_m_s2.tolist(),
        "diff_m_s2": diff_m_s2.tolist(),
        "diff_norm_m_s2": float(np.linalg.norm(diff_m_s2)),
        "two_body_accel_m_s2": two_body_accel_m_s2.tolist(),
        "matlab_harmonic_accel_full_m_s2": matlab_debug["harmonic_accel_m_s2"].tolist(),
        "matlab_harmonic_accel_m_s2": matlab_harmonic_perturbation_m_s2.tolist(),
        "python_harmonic_accel_m_s2": python_harmonic_accel_m_s2.tolist(),
        "diff_harmonic_m_s2": diff_harmonic_m_s2.tolist(),
        "diff_harmonic_norm_m_s2": float(np.linalg.norm(diff_harmonic_m_s2)),
        "matlab_harmonic_accel_full_bf_m_s2": matlab_debug["harmonic_accel_bf_m_s2"].tolist(),
        "matlab_harmonic_accel_bf_m_s2": matlab_harmonic_perturbation_bf_m_s2.tolist(),
        "python_harmonic_accel_bf_m_s2": a_harmonic_bf_py_m_s2.tolist(),
        "diff_harmonic_bf_m_s2": diff_a_bf_m_s2.tolist(),
        "diff_harmonic_bf_norm_m_s2": float(np.linalg.norm(diff_a_bf_m_s2)),
        "matlab_r_bf_m": matlab_debug["r_bf_m"].tolist(),
        "python_r_bf_m": r_bf_py_m.tolist(),
        "diff_r_bf_m": diff_r_bf_m.tolist(),
        "diff_r_bf_norm_m": float(np.linalg.norm(diff_r_bf_m)),
        "matlab_E": matlab_debug["E"].tolist(),
        "python_E": E_py.tolist(),
        "diff_E": diff_e.tolist(),
        "diff_E_fro": float(np.linalg.norm(diff_e)),
        **shared_ephemeris_artifacts,
        **matlab_artifacts,
    }
    result_path = outdir / "matlab_hpop_accel_compare.json"
    write_json(str(result_path), result)
    result["result_path"] = str(result_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare one-step simulator acceleration against MATLAB HPOP.")
    parser.add_argument("--config", required=True, help="Path to simulator YAML config.")
    parser.add_argument("--output-dir", default="outputs/validation_matlab_hpop_accel_compare", help="Output directory.")
    parser.add_argument("--object-id", default="", help="Object id to validate (target or chaser).")
    parser.add_argument("--hpop-root", default="", help="Path to the MATLAB HPOP root directory.")
    parser.add_argument("--matlab-executable", default="matlab", help="MATLAB executable path or command.")
    parser.add_argument("--timeout-s", type=float, default=None, help="Optional MATLAB timeout in seconds.")
    parser.add_argument("--sample-index", type=int, default=0, help="Truth sample index to compare.")
    args = parser.parse_args(argv)

    result = run_matlab_hpop_accel_compare(
        config_path=args.config,
        output_dir=args.output_dir,
        hpop_root=(None if not str(args.hpop_root).strip() else str(args.hpop_root).strip()),
        object_id=(None if not str(args.object_id).strip() else str(args.object_id).strip()),
        matlab_executable=str(args.matlab_executable),
        timeout_s=args.timeout_s,
        sample_index=int(args.sample_index),
    )
    print(f"Sample Index    : {result['sample_index']}")
    print(f"Sample Time s   : {result['sample_t_s']}")
    print(f"Object          : {result['object_id']}")
    print("Python accel    : [{:.12e}, {:.12e}, {:.12e}] m/s^2".format(*result["python_accel_m_s2"]))
    print("MATLAB accel    : [{:.12e}, {:.12e}, {:.12e}] m/s^2".format(*result["matlab_accel_m_s2"]))
    print("Diff accel      : [{:.12e}, {:.12e}, {:.12e}] m/s^2".format(*result["diff_m_s2"]))
    print(f"Diff norm m/s^2 : {result['diff_norm_m_s2']:.12e}")
    print("Python r_bf     : [{:.12e}, {:.12e}, {:.12e}] m".format(*result["python_r_bf_m"]))
    print("MATLAB r_bf     : [{:.12e}, {:.12e}, {:.12e}] m".format(*result["matlab_r_bf_m"]))
    print("Diff r_bf       : [{:.12e}, {:.12e}, {:.12e}] m".format(*result["diff_r_bf_m"]))
    print(f"Diff r_bf norm m: {result['diff_r_bf_norm_m']:.12e}")
    print("Python a_harm   : [{:.12e}, {:.12e}, {:.12e}] m/s^2".format(*result["python_harmonic_accel_m_s2"]))
    print("MATLAB a_harm   : [{:.12e}, {:.12e}, {:.12e}] m/s^2".format(*result["matlab_harmonic_accel_m_s2"]))
    print("Diff a_harm     : [{:.12e}, {:.12e}, {:.12e}] m/s^2".format(*result["diff_harmonic_m_s2"]))
    print(f"Diff a_harm norm: {result['diff_harmonic_norm_m_s2']:.12e}")
    print(f"Diff E fro      : {result['diff_E_fro']:.12e}")
    print(f"Result path     : {result['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
