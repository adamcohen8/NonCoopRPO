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
from sim.single_run import _coerce_noninteractive_for_automation, _run_single_config
from sim.utils.io import write_json
from validation.hpop_compare import _parse_hpop_satellite_states
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


def run_matlab_hpop_step_compare(
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
    if truth.ndim != 2 or truth.shape[0] < 2 or truth.shape[1] < 14:
        raise ValueError("Need at least two truth samples for one-step comparison.")

    thrust_hist = np.array((payload.get("applied_thrust_by_object", {}) or {}).get(selected_object_id, []), dtype=float)
    if thrust_hist.size > 0 and _finite_max_norm(thrust_hist.reshape(-1, 3)) > 1.0e-12:
        raise ValueError("Step comparison currently supports passive orbit cases only.")

    truth0 = truth[0, :]
    props = _satellite_properties(cfg, selected_object_id, mass_kg=float(truth0[13]))
    force_model = _force_model_from_cfg(cfg)
    epoch_dt = julian_date_to_datetime(float(cfg.simulator.initial_jd_utc) + float(time_s[0]) / 86400.0).astimezone(timezone.utc)
    dt = float(time_s[1] - time_s[0])

    case_dir = outdir / "matlab_hpop_step_case"
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
        "step_s": dt,
        "duration_s": dt,
        **props,
        **force_model,
    }
    case_input_path = _write_case_input(case_dir, case_params)
    write_json(str(case_dir / "hpop_step_case_manifest.json"), {"config_path": str(cfg_path), "case_params": case_params})
    matlab_artifacts = _invoke_matlab_case(
        case_dir=case_dir,
        hpop_root=hpop_root_path,
        matlab_executable=matlab_executable,
        timeout_s=timeout_s,
    )

    hpop = _parse_hpop_satellite_states(case_dir / "SatelliteStates.txt")
    if hpop.x_eci_km_km_s.shape[0] < 2:
        raise ValueError("MATLAB HPOP did not return two samples for one-step comparison.")

    python_step = truth[1, :6]
    matlab_step = hpop.x_eci_km_km_s[1, :]
    diff = python_step - matlab_step
    pos_diff_m = diff[:3] * 1e3
    vel_diff_mm_s = diff[3:] * 1e6

    result = {
        "config_path": str(cfg_path),
        "object_id": selected_object_id,
        "dt_s": dt,
        "python_step_state": python_step.tolist(),
        "matlab_step_state": matlab_step.tolist(),
        "pos_diff_m": pos_diff_m.tolist(),
        "vel_diff_mm_s": vel_diff_mm_s.tolist(),
        "pos_diff_norm_m": float(np.linalg.norm(pos_diff_m)),
        "vel_diff_norm_mm_s": float(np.linalg.norm(vel_diff_mm_s)),
        "case_dir": str(case_dir),
        "case_input_path": str(case_input_path),
        **matlab_artifacts,
    }
    outpath = outdir / "matlab_hpop_step_compare.json"
    write_json(str(outpath), result)
    result["result_path"] = str(outpath)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare one 1-step propagated state against MATLAB HPOP.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/validation_matlab_hpop_step_compare")
    parser.add_argument("--object-id", default="")
    parser.add_argument("--hpop-root", default="")
    parser.add_argument("--matlab-executable", default="matlab")
    parser.add_argument("--timeout-s", type=float, default=None)
    args = parser.parse_args(argv)

    result = run_matlab_hpop_step_compare(
        config_path=args.config,
        output_dir=args.output_dir,
        hpop_root=(None if not str(args.hpop_root).strip() else str(args.hpop_root).strip()),
        object_id=(None if not str(args.object_id).strip() else str(args.object_id).strip()),
        matlab_executable=str(args.matlab_executable),
        timeout_s=args.timeout_s,
    )
    print(f"Object            : {result['object_id']}")
    print(f"dt s              : {result['dt_s']}")
    print("Pos diff m        : [{:.12e}, {:.12e}, {:.12e}]".format(*result["pos_diff_m"]))
    print("Vel diff mm/s     : [{:.12e}, {:.12e}, {:.12e}]".format(*result["vel_diff_mm_s"]))
    print(f"Pos diff norm m   : {result['pos_diff_norm_m']:.12e}")
    print(f"Vel diff norm mm/s: {result['vel_diff_norm_mm_s']:.12e}")
    print(f"Result path       : {result['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
