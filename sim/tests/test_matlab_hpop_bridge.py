from __future__ import annotations

from datetime import timedelta, timezone
from pathlib import Path
import subprocess

import numpy as np
import yaml
from unittest.mock import patch

from sim.dynamics.orbit.epoch import julian_date_to_datetime
from validation.matlab_hpop_bridge import run_matlab_hpop_validation


def _truth_row(x_km: float, t_s: float) -> list[float]:
    return [
        x_km,
        0.0,
        0.0,
        0.0,
        7.5,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        100.0,
    ]


def _write_satellite_states(path: Path, jd_utc: float, truth_rows: list[list[float]]) -> None:
    epoch = julian_date_to_datetime(jd_utc).astimezone(timezone.utc)
    lines: list[str] = []
    for idx, row in enumerate(truth_rows):
        stamp = epoch + timedelta(seconds=float(idx))
        lines.append(
            f"  {stamp.year:04d}/{stamp.month:02d}/{stamp.day:02d}  "
            f"{stamp.hour:2d}:{stamp.minute:02d}:{stamp.second:06.3f}"
            f"  {row[0]*1e3:14.3f}{row[1]*1e3:14.3f}{row[2]*1e3:14.3f}"
            f"{row[3]*1e3:12.3f}{row[4]*1e3:12.3f}{row[5]*1e3:12.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@patch("validation.matlab_hpop_bridge._run_single_config")
@patch("validation.matlab_hpop_bridge.subprocess.run")
@patch("validation.matlab_hpop_bridge._resolve_matlab_executable", return_value="matlab")
def test_run_matlab_hpop_validation_generates_case_and_compares(
    mock_resolve_matlab_executable,
    mock_subprocess_run,
    mock_run_single_config,
    tmp_path,
) -> None:
    cfg_path = tmp_path / "case.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "scenario_name": "matlab_hpop_demo",
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "specs": {
                        "mass_kg": 100.0,
                        "drag_area_m2": 2.5,
                        "solar_area_m2": 3.5,
                        "cd": 2.3,
                        "cr": 1.4,
                    },
                    "initial_state": {
                        "position_eci_km": [7000.0, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 7.5, 0.0],
                    },
                },
                "simulator": {
                    "duration_s": 2.0,
                    "dt_s": 1.0,
                    "initial_jd_utc": 2451545.0,
                    "dynamics": {"orbit": {"j2": True, "drag": True, "third_body_sun": True}},
                },
                "outputs": {"output_dir": "outputs/demo", "mode": "save"},
                "monte_carlo": {"enabled": False, "iterations": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    truth_rows = [_truth_row(7000.0, 0.0), _truth_row(7001.0, 1.0), _truth_row(7002.0, 2.0)]
    mock_run_single_config.return_value = {
        "summary": {"scenario_name": "matlab_hpop_demo"},
        "time_s": [0.0, 1.0, 2.0],
        "truth_by_object": {"target": truth_rows},
        "applied_thrust_by_object": {"target": [[0.0, 0.0, 0.0]] * 3},
    }

    def _fake_subprocess_run(cmd, cwd, check, capture_output, text, timeout):
        case_dir = (tmp_path / "out" / "matlab_hpop_case").resolve()
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_satellite_states(case_dir / "SatelliteStates.txt", 2451545.0, truth_rows)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    mock_subprocess_run.side_effect = _fake_subprocess_run

    result = run_matlab_hpop_validation(
        config_path=cfg_path,
        output_dir=tmp_path / "out",
        hpop_root=tmp_path / "fake_hpop_root",
        matlab_executable="matlab",
        plot_mode="none",
    )

    mock_resolve_matlab_executable.assert_called_once_with("matlab")
    assert float(result["pos_err_max_m"]) == 0.0
    assert float(result["vel_err_max_mm_s"]) == 0.0
    assert int(result["gravity_degree"]) == 2
    assert bool(result["enable_drag"]) is True
    assert bool(result["enable_sun"]) is True
    assert Path(result["case_input_path"]).exists()
    assert Path(result["case_manifest_path"]).exists()
    assert Path(result["satellite_states_path"]).exists()
    case_input_text = Path(result["case_input_path"]).read_text(encoding="utf-8")
    assert "case_params.gravity_degree = 2;" in case_input_text
    assert "case_params.enable_drag = 1;" in case_input_text
    assert "case_params.area_drag_m2 = 2.500000000000;" in case_input_text


@patch("validation.matlab_hpop_bridge._run_single_config")
@patch("validation.matlab_hpop_bridge.subprocess.run")
@patch("validation.matlab_hpop_bridge._resolve_matlab_executable", return_value="matlab")
def test_run_matlab_hpop_validation_injects_shared_moon_ephemeris(
    mock_resolve_matlab_executable,
    mock_subprocess_run,
    mock_run_single_config,
    tmp_path,
) -> None:
    cfg_path = tmp_path / "case_shared_moon.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "scenario_name": "matlab_hpop_shared_moon",
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "specs": {"mass_kg": 100.0},
                    "initial_state": {
                        "position_eci_km": [7000.0, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 7.5, 0.0],
                    },
                },
                "simulator": {
                    "duration_s": 2.0,
                    "dt_s": 1.0,
                    "initial_jd_utc": 2451545.0,
                    "dynamics": {"orbit": {"third_body_moon": True}},
                    "environment": {
                        "ephemeris_mode": "analytic_enhanced",
                        "shared_hpop_ephemeris_bodies": ["moon"],
                        "shared_hpop_ephemeris_step_s": 0.25,
                    },
                },
                "outputs": {"output_dir": "outputs/demo", "mode": "save"},
                "monte_carlo": {"enabled": False, "iterations": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    truth_rows = [_truth_row(7000.0, 0.0), _truth_row(7001.0, 1.0), _truth_row(7002.0, 2.0)]

    def _fake_run_single_config(runtime_cfg):
        env = dict(runtime_cfg.simulator.environment or {})
        assert env["moon_ephemeris_time_s"] == [0.0, 1.0, 2.0]
        assert env["moon_ephemeris_eci_km"][1] == [11.0, 21.0, 31.0]
        return {
            "summary": {"scenario_name": "matlab_hpop_shared_moon"},
            "time_s": [0.0, 1.0, 2.0],
            "truth_by_object": {"target": truth_rows},
            "applied_thrust_by_object": {"target": [[0.0, 0.0, 0.0]] * 3},
        }

    mock_run_single_config.side_effect = _fake_run_single_config

    def _fake_subprocess_run(cmd, cwd, check, capture_output, text, timeout):
        case_dir = (tmp_path / "out_shared" / ("matlab_hpop_ephemeris_case" if "run_hpop_ephemeris_case" in cmd[-1] else "matlab_hpop_case")).resolve()
        case_dir.mkdir(parents=True, exist_ok=True)
        if "run_hpop_ephemeris_case" in cmd[-1]:
            (case_dir / "BodyEphemeris.txt").write_text(
                "\n".join(
                    [
                        "0.000000000 10.0 20.0 30.0 100.0 200.0 300.0",
                        "1.000000000 11.0 21.0 31.0 101.0 201.0 301.0",
                        "2.000000000 12.0 22.0 32.0 102.0 202.0 302.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            _write_satellite_states(case_dir / "SatelliteStates.txt", 2451545.0, truth_rows)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    mock_subprocess_run.side_effect = _fake_subprocess_run

    result = run_matlab_hpop_validation(
        config_path=cfg_path,
        output_dir=tmp_path / "out_shared",
        hpop_root=tmp_path / "fake_hpop_root",
        matlab_executable="matlab",
        plot_mode="none",
    )

    assert mock_resolve_matlab_executable.call_count == 2
    assert result["shared_ephemeris_bodies"] == ["moon"]
    assert float(result["shared_ephemeris_step_s"]) == 0.25
    assert Path(result["ephemeris_export_path"]).exists()
