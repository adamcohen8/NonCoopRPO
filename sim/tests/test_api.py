from __future__ import annotations

from pathlib import Path
import tempfile
import json
from unittest.mock import patch

import numpy as np
import yaml

from sim import SimulationConfig, SimulationResult, SimulationSession, SimulationSnapshot
from sim.master_simulator import run_master_simulation


def _api_config(output_dir: Path, *, monte_carlo: bool = False) -> dict:
    return {
        "scenario_name": "api_smoke",
        "scenario_description": "API smoke test scenario",
        "rocket": {"enabled": False},
        "target": {
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
            },
        },
        "chaser": {"enabled": False},
        "simulator": {
            "duration_s": 2.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {
                "print_summary": False,
                "save_json": False,
                "save_full_log": False,
            },
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
            "monte_carlo": {
                "save_iteration_summaries": False,
                "save_histograms": False,
                "display_histograms": False,
                "save_ops_dashboard": False,
                "display_ops_dashboard": False,
            },
        },
        "monte_carlo": {
            "enabled": bool(monte_carlo),
            "iterations": 2 if monte_carlo else 1,
            "base_seed": 7,
            "parallel_enabled": False,
            "variations": [],
        },
        "metadata": {"seed": 123},
    }


def _sensitivity_api_config(output_dir: Path) -> dict:
    cfg = _api_config(output_dir, monte_carlo=False)
    cfg["analysis"] = {
        "enabled": True,
        "study_type": "sensitivity",
        "execution": {
            "parallel_enabled": False,
            "parallel_workers": 0,
        },
        "metrics": [
            "summary.duration_s",
            "derived.closest_approach_km",
        ],
        "baseline": {
            "enabled": True,
        },
        "sensitivity": {
            "method": "one_at_a_time",
            "parameters": [
                {
                    "parameter_path": "simulator.dt_s",
                    "values": [0.5, 1.0],
                }
            ],
        },
    }
    return cfg


def _lhs_sensitivity_api_config(output_dir: Path) -> dict:
    cfg = _api_config(output_dir, monte_carlo=False)
    cfg["analysis"] = {
        "enabled": True,
        "study_type": "sensitivity",
        "execution": {
            "parallel_enabled": False,
            "parallel_workers": 0,
        },
        "metrics": [
            "summary.duration_s",
            "derived.closest_approach_km",
        ],
        "sensitivity": {
            "method": "lhs",
            "samples": 5,
            "seed": 19,
            "parameters": [
                {
                    "parameter_path": "simulator.dt_s",
                    "distribution": "uniform",
                    "low": 0.5,
                    "high": 1.5,
                }
            ],
        },
    }
    return cfg


def _attitude_api_config(output_dir: Path) -> dict:
    cfg = _api_config(output_dir, monte_carlo=False)
    cfg["target"]["initial_state"].update(
        {
            "attitude_quat_bn": [1.0, 0.0, 0.0, 0.0],
            "angular_rate_body_rad_s": [0.01, 0.02, -0.01],
        }
    )
    cfg["target"]["specs"]["inertia_kg_m2"] = [[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 8.0]]
    cfg["simulator"]["dynamics"]["attitude"] = {"enabled": True}
    return cfg


class TestSimulationApi:
    def test_session_from_yaml_and_legacy_single_run_callback_still_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "api_smoke.yaml"
            cfg_dict = _api_config(Path(tmpdir))
            cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

            session = SimulationSession.from_yaml(cfg_path)
            result = session.run()
            callback_events: list[tuple[int, int]] = []

            legacy = run_master_simulation(cfg_path, step_callback=lambda step, total: callback_events.append((step, total)))

            assert result.summary["scenario_name"] == "api_smoke"
            assert legacy["run"]["samples"] == result.summary["samples"]
            assert callback_events[0] == (0, 2)
            assert callback_events[-1] == (2, 2)

    def test_session_reset_step_and_run_single_scenario(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimulationConfig.from_dict(_api_config(Path(tmpdir)))
            session = SimulationSession.from_config(cfg)

            snap0 = session.reset(seed=42)

            assert isinstance(snap0, SimulationSnapshot)
            assert snap0.step_index == 0
            assert snap0.time_s == 0.0
            assert "target" in snap0.truth
            assert session.done is False

            snap1 = session.step()
            snap2 = session.step()

            assert snap1.step_index == 1
            assert snap1.time_s == 1.0
            assert snap2.step_index == 2
            assert snap2.time_s == 2.0
            assert session.done is True

            result = session.run()

            assert isinstance(result, SimulationResult)
            assert result.is_monte_carlo is False
            assert result.num_steps == 3
            assert result.summary["samples"] == 3
            assert result.metrics["scenario_name"] == "api_smoke"
            assert result.summary["scenario_description"] == "API smoke test scenario"
            assert np.isfinite(result.truth["target"]).all()

    def test_session_step_uses_live_engine_not_full_run_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimulationConfig.from_dict(_api_config(Path(tmpdir)))
            session = SimulationSession.from_config(cfg)

            with patch("sim.master_simulator._run_single_config", side_effect=AssertionError("full run helper should not be used")):
                snap0 = session.reset()
                snap1 = session.step()

            assert snap0 is not None
            assert snap1.step_index == 1
            assert snap1.time_s == 1.0

    def test_session_run_after_reset_binds_step_callback_to_existing_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimulationConfig.from_dict(_api_config(Path(tmpdir)))
            session = SimulationSession.from_config(cfg)
            callback_events: list[tuple[int, int]] = []

            snap0 = session.reset()
            result = session.run(step_callback=lambda step, total: callback_events.append((step, total)))

            assert snap0 is not None
            assert isinstance(result, SimulationResult)
            assert callback_events[0] == (0, 2)
            assert callback_events[-1] == (2, 2)

    def test_session_run_monte_carlo_scenario(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimulationConfig.from_dict(_api_config(Path(tmpdir), monte_carlo=True))
            session = SimulationSession.from_config(cfg)

            assert session.reset() is None

            result = session.run()

            assert result.is_monte_carlo is True
            assert result.payload["monte_carlo"]["enabled"] is True
            assert result.payload["monte_carlo"]["iterations"] == 2
            assert "pass_rate" in result.metrics

    def test_session_run_sensitivity_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimulationConfig.from_dict(_sensitivity_api_config(Path(tmpdir)))
            session = SimulationSession.from_config(cfg)

            assert session.reset() is None

            result = session.run()

            assert result.is_batch_analysis is True
            assert result.analysis_study_type == "sensitivity"
            assert result.payload["analysis"]["run_count"] == 2
            assert len(result.payload["parameter_summaries"]) == 1
            assert result.metrics["run_count"] == 2

    def test_session_run_sensitivity_lhs_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimulationConfig.from_dict(_lhs_sensitivity_api_config(Path(tmpdir)))
            session = SimulationSession.from_config(cfg)

            assert session.reset() is None

            result = session.run()

            assert result.analysis_study_type == "sensitivity"
            assert result.payload["analysis"]["method"] == "lhs"
            assert result.payload["analysis"]["run_count"] == 5
            assert result.payload["analysis"]["samples"] == 5
            assert len(result.payload["runs"]) == 5
            assert len(result.payload["parameter_rankings"]) == 1

    def test_session_preserves_relative_baseline_paths_for_batch_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_path = root / "baseline.json"
            baseline_path.write_text(
                json.dumps(
                    {
                        "aggregate_stats": {"closest_approach_km_min": 123.0},
                        "summary": {"scenario_name": "baseline"},
                    }
                ),
                encoding="utf-8",
            )
            cfg_dict = _sensitivity_api_config(root)
            cfg_dict["analysis"]["baseline"] = {
                "enabled": False,
                "summary_json": "baseline.json",
            }
            cfg_path = root / "api_baseline.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

            result = SimulationSession.from_yaml(cfg_path).run()

            assert result.payload["baseline"]["source"] == "file"
            assert Path(result.payload["baseline"]["path"]).resolve() == baseline_path.resolve()
            assert result.payload["config_path"] == str(cfg_path.resolve())

    def test_session_preserves_full_belief_state_when_attitude_is_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimulationConfig.from_dict(_attitude_api_config(Path(tmpdir)))
            result = SimulationSession.from_config(cfg).run()

            assert result.truth["target"].shape[1] == 14
            assert result.belief["target"].shape[1] == 13
