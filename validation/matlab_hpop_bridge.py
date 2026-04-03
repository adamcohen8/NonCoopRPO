from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.config import SimulationScenarioConfig, load_simulation_yaml, scenario_config_from_dict
from sim.dynamics.orbit.epoch import julian_date_to_datetime
from sim.single_run import _coerce_noninteractive_for_automation, _run_single_config
from sim.utils.io import write_json
from validation.hpop_compare import _parse_hpop_satellite_states, compare_state_histories


def _default_hpop_root() -> Path:
    return REPO_ROOT / "validation" / "High Precision Orbit Propagator_4-2" / "High Precision Orbit Propagator_4.2.2"


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(raw)


def _build_single_run_cfg(
    config_path: Path,
    output_dir: Path,
    *,
    environment_overrides: dict[str, Any] | None = None,
) -> SimulationScenarioConfig:
    raw = _load_yaml_dict(config_path)
    outputs = dict(raw.get("outputs", {}) or {})
    outputs["output_dir"] = str(output_dir)
    outputs["mode"] = "save"
    stats = dict(outputs.get("stats", {}) or {})
    stats["print_summary"] = False
    stats["save_json"] = False
    stats["save_csv"] = False
    stats["save_full_log"] = False
    outputs["stats"] = stats
    outputs["plots"] = {"enabled": False, "figure_ids": []}
    outputs["animations"] = {"enabled": False, "types": []}
    mc_outputs = dict(outputs.get("monte_carlo", {}) or {})
    mc_outputs["save_iteration_summaries"] = False
    mc_outputs["save_aggregate_summary"] = False
    outputs["monte_carlo"] = mc_outputs
    raw["outputs"] = outputs

    monte_carlo = dict(raw.get("monte_carlo", {}) or {})
    monte_carlo["enabled"] = False
    monte_carlo["iterations"] = 1
    monte_carlo["variations"] = []
    raw["monte_carlo"] = monte_carlo

    analysis = dict(raw.get("analysis", {}) or {})
    analysis["enabled"] = False
    raw["analysis"] = analysis

    if environment_overrides:
        sim = dict(raw.get("simulator", {}) or {})
        env = dict(sim.get("environment", {}) or {})
        env.update(environment_overrides)
        sim["environment"] = env
        raw["simulator"] = sim

    return _coerce_noninteractive_for_automation(scenario_config_from_dict(raw))


def _default_object_id(cfg: SimulationScenarioConfig) -> str:
    if cfg.target.enabled:
        return "target"
    if cfg.chaser.enabled:
        return "chaser"
    raise ValueError("No enabled satellite object found. Specify object_id explicitly.")


def _agent_section(cfg: SimulationScenarioConfig, object_id: str):
    if object_id == "target":
        return cfg.target
    if object_id == "chaser":
        return cfg.chaser
    raise ValueError("matlab_hpop validation currently supports only 'target' or 'chaser' object_id values.")


def _finite_max_norm(arr: np.ndarray) -> float:
    vals = np.array(arr, dtype=float)
    if vals.ndim != 2 or vals.shape[1] != 3:
        return float("nan")
    norms = np.linalg.norm(vals, axis=1)
    finite = norms[np.isfinite(norms)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite))


def _force_model_from_cfg(cfg: SimulationScenarioConfig) -> dict[str, Any]:
    orbit = dict(cfg.simulator.dynamics.get("orbit", {}) or {})
    sh = dict(orbit.get("spherical_harmonics", {}) or {})
    gravity_degree = 0
    gravity_order = 0
    if bool(sh.get("enabled", False)):
        gravity_degree = int(max(int(sh.get("degree", 0)), 0))
        gravity_order = int(max(min(int(sh.get("order", gravity_degree)), gravity_degree), 0))
    elif bool(orbit.get("j4", False)):
        gravity_degree = 4
        gravity_order = 0
    elif bool(orbit.get("j3", False)):
        gravity_degree = 3
        gravity_order = 0
    elif bool(orbit.get("j2", False)):
        gravity_degree = 2
        gravity_order = 0
    return {
        "gravity_degree": gravity_degree,
        "gravity_order": gravity_order,
        "atmosphere_model": str(((cfg.simulator.environment or {}).get("atmosphere_model", "nrlmsise00"))).lower(),
        "enable_sun": int(bool(orbit.get("third_body_sun", False))),
        "enable_moon": int(bool(orbit.get("third_body_moon", False))),
        "enable_solar_radiation": int(bool(orbit.get("srp", False))),
        "enable_drag": int(bool(orbit.get("drag", False))),
        "enable_planets": 0,
        "enable_solid_earth_tides": 0,
        "enable_ocean_tides": 0,
        "enable_relativity": int(bool(orbit.get("relativity", False))),
    }


def _satellite_properties(cfg: SimulationScenarioConfig, object_id: str, mass_kg: float) -> dict[str, float]:
    agent = _agent_section(cfg, object_id)
    specs = dict(agent.specs or {})
    env = dict(cfg.simulator.environment or {})
    drag_area_m2 = float(
        specs.get(
            "drag_area_m2",
            specs.get("area_drag_m2", env.get("drag_area_m2", specs.get("area_m2", 1.0))),
        )
    )
    solar_area_m2 = float(
        specs.get(
            "solar_area_m2",
            specs.get("area_solar_m2", env.get("srp_area_m2", specs.get("area_m2", drag_area_m2))),
        )
    )
    cd = float(specs.get("cd", env.get("cd", 2.2)))
    cr = float(specs.get("cr", env.get("cr", 1.2)))
    return {
        "mass_kg": float(mass_kg),
        "area_drag_m2": drag_area_m2,
        "area_solar_m2": solar_area_m2,
        "cd": cd,
        "cr": cr,
    }


def _matlab_quote(value: str) -> str:
    return str(value).replace("'", "''")


def _resolve_matlab_executable(matlab_executable: str) -> str:
    raw = str(matlab_executable).strip()
    if not raw:
        raw = "matlab"
    p = Path(raw).expanduser()
    if p.is_absolute() or "/" in raw:
        resolved = p.resolve()
        if resolved.exists():
            return str(resolved)
        raise FileNotFoundError(f"MATLAB executable was not found at: {resolved}")

    found = shutil.which(raw)
    if found:
        return str(Path(found).resolve())

    app_matches = sorted(Path("/Applications").glob("MATLAB_*.app/bin/matlab"), reverse=True)
    if app_matches:
        return str(app_matches[0].resolve())
    raise FileNotFoundError(
        f"MATLAB executable '{raw}' was not found on PATH and no /Applications/MATLAB_*.app/bin/matlab install was detected."
    )


def _write_case_input(case_dir: Path, case_params: dict[str, Any]) -> Path:
    lines = [
        "case_params = struct();",
        f"case_params.scenario_name = '{_matlab_quote(str(case_params['scenario_name']))}';",
        f"case_params.object_id = '{_matlab_quote(str(case_params['object_id']))}';",
        f"case_params.epoch_year = {int(case_params['epoch_year'])};",
        f"case_params.epoch_month = {int(case_params['epoch_month'])};",
        f"case_params.epoch_day = {int(case_params['epoch_day'])};",
        f"case_params.epoch_hour = {int(case_params['epoch_hour'])};",
        f"case_params.epoch_minute = {int(case_params['epoch_minute'])};",
        f"case_params.epoch_second = {float(case_params['epoch_second']):.9f};",
        "case_params.initial_state_eci_m_m_s = ["
        + " ".join(f"{float(v):.12f}" for v in list(case_params["initial_state_eci_m_m_s"]))
        + "];",
        f"case_params.step_s = {float(case_params['step_s']):.9f};",
        f"case_params.duration_s = {float(case_params['duration_s']):.9f};",
        f"case_params.mass_kg = {float(case_params['mass_kg']):.12f};",
        f"case_params.area_drag_m2 = {float(case_params['area_drag_m2']):.12f};",
        f"case_params.area_solar_m2 = {float(case_params['area_solar_m2']):.12f};",
        f"case_params.cd = {float(case_params['cd']):.12f};",
        f"case_params.cr = {float(case_params['cr']):.12f};",
        f"case_params.gravity_degree = {int(case_params['gravity_degree'])};",
        f"case_params.gravity_order = {int(case_params['gravity_order'])};",
        f"case_params.atmosphere_model = '{_matlab_quote(str(case_params.get('atmosphere_model', 'nrlmsise00')))}';",
        f"case_params.enable_sun = {int(case_params['enable_sun'])};",
        f"case_params.enable_moon = {int(case_params['enable_moon'])};",
        f"case_params.enable_solar_radiation = {int(case_params['enable_solar_radiation'])};",
        f"case_params.enable_drag = {int(case_params['enable_drag'])};",
        f"case_params.enable_planets = {int(case_params['enable_planets'])};",
        f"case_params.enable_solid_earth_tides = {int(case_params['enable_solid_earth_tides'])};",
        f"case_params.enable_ocean_tides = {int(case_params['enable_ocean_tides'])};",
        f"case_params.enable_relativity = {int(case_params['enable_relativity'])};",
    ]
    if case_params.get("rkf78_trace_path"):
        lines.append(f"case_params.rkf78_trace_path = '{_matlab_quote(str(case_params['rkf78_trace_path']))}';")
    if case_params.get("rkf78_stage_trace_path"):
        lines.append(f"case_params.rkf78_stage_trace_path = '{_matlab_quote(str(case_params['rkf78_stage_trace_path']))}';")
    path = case_dir / "hpop_case_input.m"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_ephemeris_case_input(case_dir: Path, case_params: dict[str, Any]) -> Path:
    lines = [
        "case_params = struct();",
        f"case_params.epoch_year = {int(case_params['epoch_year'])};",
        f"case_params.epoch_month = {int(case_params['epoch_month'])};",
        f"case_params.epoch_day = {int(case_params['epoch_day'])};",
        f"case_params.epoch_hour = {int(case_params['epoch_hour'])};",
        f"case_params.epoch_minute = {int(case_params['epoch_minute'])};",
        f"case_params.epoch_second = {float(case_params['epoch_second']):.9f};",
        f"case_params.step_s = {float(case_params['step_s']):.9f};",
        f"case_params.duration_s = {float(case_params['duration_s']):.9f};",
        "case_params.export_bodies = {"
        + ", ".join(f"'{_matlab_quote(str(v))}'" for v in list(case_params.get("export_bodies", [])))
        + "};",
    ]
    path = case_dir / "hpop_ephemeris_input.m"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _invoke_matlab_case(
    *,
    case_dir: Path,
    hpop_root: Path,
    matlab_executable: str,
    timeout_s: float | None,
) -> dict[str, str]:
    matlab_executable_resolved = _resolve_matlab_executable(matlab_executable)
    runner_path = (REPO_ROOT / "validation" / "run_hpop_case.m").resolve()
    batch_expr = (
        f"addpath('{_matlab_quote(str(runner_path.parent))}'); "
        f"run_hpop_case('{_matlab_quote(str(case_dir))}', '{_matlab_quote(str(hpop_root))}');"
    )
    cmd = [matlab_executable_resolved, "-batch", batch_expr]
    completed = subprocess.run(
        cmd,
        cwd=str(hpop_root),
        check=False,
        capture_output=True,
        text=True,
        timeout=None if timeout_s is None else float(timeout_s),
    )
    stdout_path = case_dir / "matlab_stdout.txt"
    stderr_path = case_dir / "matlab_stderr.txt"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"MATLAB HPOP run failed with exit code {completed.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )
    return {
        "matlab_stdout_path": str(stdout_path),
        "matlab_stderr_path": str(stderr_path),
        "matlab_executable": matlab_executable_resolved,
        "matlab_command": " ".join(cmd),
    }


def _invoke_matlab_ephemeris_case(
    *,
    case_dir: Path,
    hpop_root: Path,
    matlab_executable: str,
    timeout_s: float | None,
) -> dict[str, str]:
    matlab_executable_resolved = _resolve_matlab_executable(matlab_executable)
    runner_path = (REPO_ROOT / "validation" / "run_hpop_ephemeris_case.m").resolve()
    batch_expr = (
        f"addpath('{_matlab_quote(str(runner_path.parent))}'); "
        f"run_hpop_ephemeris_case('{_matlab_quote(str(case_dir))}', '{_matlab_quote(str(hpop_root))}');"
    )
    cmd = [matlab_executable_resolved, "-batch", batch_expr]
    completed = subprocess.run(
        cmd,
        cwd=str(hpop_root),
        check=False,
        capture_output=True,
        text=True,
        timeout=None if timeout_s is None else float(timeout_s),
    )
    stdout_path = case_dir / "matlab_ephemeris_stdout.txt"
    stderr_path = case_dir / "matlab_ephemeris_stderr.txt"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"MATLAB HPOP ephemeris export failed with exit code {completed.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )
    return {
        "matlab_ephemeris_stdout_path": str(stdout_path),
        "matlab_ephemeris_stderr_path": str(stderr_path),
        "matlab_executable": matlab_executable_resolved,
        "matlab_ephemeris_command": " ".join(cmd),
    }


def _parse_ephemeris_export(path: Path) -> dict[str, np.ndarray]:
    arr = np.loadtxt(path, dtype=float, ndmin=2)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"Unexpected ephemeris export shape from {path}: {arr.shape}")
    out: dict[str, np.ndarray] = {
        "time_s": np.asarray(arr[:, 0], dtype=float).reshape(-1),
        "moon_eci_km": np.asarray(arr[:, 1:4], dtype=float),
        "sun_eci_km": np.asarray(arr[:, 4:7], dtype=float),
    }
    return out


def _shared_ephemeris_bodies_from_cfg(cfg: SimulationScenarioConfig) -> list[str]:
    env = dict(cfg.simulator.environment or {})
    raw = env.get("shared_hpop_ephemeris_bodies", [])
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    else:
        items = list(raw)
    out: list[str] = []
    for item in items:
        name = str(item).strip().lower()
        if name in ("sun", "moon") and name not in out:
            out.append(name)
    return out


def _shared_ephemeris_step_s_from_cfg(cfg: SimulationScenarioConfig) -> float | None:
    env = dict(cfg.simulator.environment or {})
    raw = env.get("shared_hpop_ephemeris_step_s", None)
    if raw is None:
        return None
    step_s = float(raw)
    if step_s <= 0.0:
        raise ValueError("shared_hpop_ephemeris_step_s must be positive when provided.")
    return step_s


def run_matlab_hpop_validation(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    hpop_root: str | Path | None = None,
    object_id: str | None = None,
    matlab_executable: str = "matlab",
    plot_mode: str = "none",
    timeout_s: float | None = None,
) -> dict[str, Any]:
    cfg_path = Path(config_path).expanduser().resolve()
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    hpop_root_path = _default_hpop_root() if hpop_root is None else Path(hpop_root).expanduser().resolve()
    cfg = load_simulation_yaml(cfg_path)
    selected_object_id = str(object_id or _default_object_id(cfg))
    _agent_section(cfg, selected_object_id)
    if cfg.simulator.initial_jd_utc is None:
        raise ValueError("matlab_hpop validation requires simulator.initial_jd_utc to be set.")
    if bool(cfg.monte_carlo.enabled):
        raise ValueError("matlab_hpop validation requires monte_carlo.enabled=false.")
    if bool(cfg.analysis.enabled):
        raise ValueError("matlab_hpop validation requires analysis.enabled=false.")

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
    payload = _run_single_config(runtime_cfg)

    time_s = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
    truth_by_object = dict(payload.get("truth_by_object", {}) or {})
    if selected_object_id not in truth_by_object:
        raise ValueError(f"Selected object_id '{selected_object_id}' was not present in the simulation payload.")
    truth = np.array(truth_by_object[selected_object_id], dtype=float)
    if truth.ndim != 2 or truth.shape[1] < 14 or truth.shape[0] != time_s.size:
        raise ValueError("Selected object truth history has an unexpected shape.")
    if time_s.size < 2:
        raise ValueError("Simulation output must contain at least two samples.")

    thrust_hist = np.array((payload.get("applied_thrust_by_object", {}) or {}).get(selected_object_id, []), dtype=float)
    if thrust_hist.size > 0 and _finite_max_norm(thrust_hist.reshape(-1, 3)) > 1.0e-12:
        raise ValueError(
            f"Selected object '{selected_object_id}' has non-zero applied thrust. "
            "matlab_hpop validation currently supports passive orbit cases only."
        )

    dt_series = np.diff(time_s)
    if np.any(dt_series <= 0.0):
        raise ValueError("Simulation output time history must be strictly increasing.")
    step_s = float(np.median(dt_series))
    if not np.allclose(dt_series, step_s, atol=1.0e-9, rtol=0.0):
        raise ValueError("Simulation output time history must use a uniform step size for matlab_hpop validation.")

    truth0 = truth[0, :]
    props = _satellite_properties(cfg, selected_object_id, mass_kg=float(truth0[13]))
    force_model = _force_model_from_cfg(cfg)
    epoch_dt = julian_date_to_datetime(float(cfg.simulator.initial_jd_utc) + float(time_s[0]) / 86400.0).astimezone(timezone.utc)

    case_dir = outdir / "matlab_hpop_case"
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
        "step_s": step_s,
        "duration_s": float(time_s[-1] - time_s[0]),
        **props,
        **force_model,
    }
    case_input_path = _write_case_input(case_dir, case_params)
    manifest = {
        "config_path": str(cfg_path),
        "scenario_name": cfg.scenario_name,
        "object_id": selected_object_id,
        "initial_jd_utc": float(cfg.simulator.initial_jd_utc),
        "case_params": case_params,
    }
    manifest_path = case_dir / "hpop_case_manifest.json"
    write_json(str(manifest_path), manifest)

    matlab_artifacts = _invoke_matlab_case(
        case_dir=case_dir,
        hpop_root=hpop_root_path,
        matlab_executable=matlab_executable,
        timeout_s=timeout_s,
    )

    sat_states_path = case_dir / "SatelliteStates.txt"
    hpop = _parse_hpop_satellite_states(sat_states_path)
    common_end_s = min(float(time_s[-1]), float(hpop.t_s[-1]))
    if common_end_s <= 0.0:
        raise ValueError("No overlapping positive-duration time span was available for comparison.")
    use_mask = hpop.t_s <= common_end_s + 1.0e-9
    hpop_t = hpop.t_s[use_mask]
    hpop_x = hpop.x_eci_km_km_s[use_mask, :]
    comparison = compare_state_histories(
        sim_t_s=time_s,
        sim_x_eci_km_km_s=truth[:, :6],
        ref_t_s=hpop_t,
        ref_x_eci_km_km_s=hpop_x,
        model=f"matlab_hpop_{selected_object_id}",
        plot_mode=plot_mode,
        output_dir=outdir,
        ref_label="MATLAB HPOP",
    )

    result = {
        "config_path": str(cfg_path),
        "scenario_name": cfg.scenario_name,
        "object_id": selected_object_id,
        "hpop_root": str(hpop_root_path),
        "case_dir": str(case_dir),
        "case_input_path": str(case_input_path),
        "case_manifest_path": str(manifest_path),
        "satellite_states_path": str(sat_states_path),
        "simulator_output_dir": str(sim_run_dir),
        "sim_samples": int(time_s.size),
        "hpop_samples": int(hpop.t_s.size),
        "validation_dt_s": step_s,
        "requested_duration_s": float(time_s[-1] - time_s[0]),
        "mass_kg": float(props["mass_kg"]),
        "drag_area_m2": float(props["area_drag_m2"]),
        "solar_area_m2": float(props["area_solar_m2"]),
        "cd": float(props["cd"]),
        "cr": float(props["cr"]),
        "gravity_degree": int(force_model["gravity_degree"]),
        "gravity_order": int(force_model["gravity_order"]),
        "enable_drag": bool(force_model["enable_drag"]),
        "enable_solar_radiation": bool(force_model["enable_solar_radiation"]),
        "enable_sun": bool(force_model["enable_sun"]),
        "enable_moon": bool(force_model["enable_moon"]),
        **shared_ephemeris_artifacts,
        **comparison,
        **matlab_artifacts,
    }
    write_json(str(outdir / "matlab_hpop_validation_report.json"), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a config-driven master simulator vs MATLAB HPOP comparison.")
    parser.add_argument("--config", required=True, help="Simulation YAML config path.")
    parser.add_argument("--output-dir", default="outputs/matlab_hpop_validation", help="Artifact output directory.")
    parser.add_argument("--hpop-root", default=str(_default_hpop_root()), help="Path to the MATLAB HPOP package root.")
    parser.add_argument("--object-id", default="", help="Satellite object to compare: target or chaser.")
    parser.add_argument("--matlab-executable", default="matlab", help="MATLAB executable to invoke.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both", "none"], default="none")
    parser.add_argument("--timeout-s", type=float, default=0.0, help="Optional MATLAB timeout in seconds; 0 disables timeout.")
    args = parser.parse_args()

    result = run_matlab_hpop_validation(
        config_path=args.config,
        output_dir=args.output_dir,
        hpop_root=args.hpop_root,
        object_id=(None if not str(args.object_id).strip() else str(args.object_id).strip()),
        matlab_executable=str(args.matlab_executable),
        plot_mode=str(args.plot_mode),
        timeout_s=(None if float(args.timeout_s) <= 0.0 else float(args.timeout_s)),
    )
    print(f"Scenario        : {result['scenario_name']}")
    print(f"Object          : {result['object_id']}")
    print(f"Case Dir        : {result['case_dir']}")
    print(f"Validation dt s : {result['validation_dt_s']}")
    print(f"Pos RMS m       : {result['pos_err_rms_m']}")
    print(f"Pos Max m       : {result['pos_err_max_m']}")
    print(f"Vel RMS mm/s    : {result['vel_err_rms_mm_s']}")
    print(f"Vel Max mm/s    : {result['vel_err_max_mm_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
