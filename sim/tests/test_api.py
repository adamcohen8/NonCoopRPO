from __future__ import annotations

from pathlib import Path
import tempfile
import json
import sys
from unittest.mock import patch

import numpy as np
import yaml

import run_simulation
from sim import SimulationConfig, SimulationResult, SimulationSession, SimulationSnapshot
from sim.execution import run_simulation_config_file
from sim.execution.campaigns import prepare_monte_carlo_runs, run_serial_monte_carlo_runs
from sim.master_simulator import run_master_simulation
from sim.execution.sensitivity import prepare_sensitivity_runs, run_sensitivity_runs
from sim.reporting.monte_carlo import (
    apply_monte_carlo_baseline_comparison,
    build_monte_carlo_report_payload,
    write_monte_carlo_report_artifacts,
)
from sim.reporting.monte_carlo_plots import write_monte_carlo_plot_artifacts
from sim.reporting.sensitivity import build_sensitivity_report_payload, write_sensitivity_summary_artifact


class ConstantIntegratedThrustMission:
    def __init__(self, thrust_eci_km_s2: list[float]):
        self.thrust_eci_km_s2 = np.array(thrust_eci_km_s2, dtype=float)

    def update(self, **_: object) -> dict[str, object]:
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": self.thrust_eci_km_s2.copy(),
        }


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
                    "parameter_path": "target.specs.mass_kg",
                    "distribution": "uniform",
                    "low": 90.0,
                    "high": 110.0,
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


def _target_reference_orbit_api_config(output_dir: Path) -> dict:
    cfg = _api_config(output_dir, monte_carlo=False)
    cfg["target"]["reference_orbit"] = {"enabled": True}
    cfg["target"]["initial_state"] = {
        "position_eci_km": [7000.0, 0.0, 0.0],
        "velocity_eci_km_s": [0.0, 0.0, 0.0],
    }
    cfg["target"]["mission_objectives"] = [
        {
            "module": "sim.tests.test_api",
            "class_name": "ConstantIntegratedThrustMission",
            "params": {"thrust_eci_km_s2": [1.0, 0.0, 0.0]},
        }
    ]
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

    def test_execution_service_file_single_run_matches_api_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "api_smoke.yaml"
            cfg_dict = _api_config(Path(tmpdir))
            cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

            service_out = run_simulation_config_file(cfg_path)
            api_result = SimulationSession.from_yaml(cfg_path).run()

            assert service_out["config_path"] == str(cfg_path.resolve())
            assert service_out["scenario_name"] == api_result.summary["scenario_name"]
            assert service_out["run"]["samples"] == api_result.summary["samples"]
            assert service_out["run"]["duration_s"] == api_result.summary["duration_s"]

    def test_cli_single_run_keeps_existing_config_command_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "api_smoke.yaml"
            cfg_path.write_text(yaml.safe_dump(_api_config(Path(tmpdir)), sort_keys=False), encoding="utf-8")
            service_payload = {
                "config_path": str(cfg_path.resolve()),
                "scenario_name": "api_smoke",
                "scenario_description": "API smoke test scenario",
                "monte_carlo": {"enabled": False},
                "run": {
                    "scenario_name": "api_smoke",
                    "scenario_description": "API smoke test scenario",
                    "objects": ["target"],
                    "samples": 3,
                    "dt_s": 1.0,
                    "duration_s": 2.0,
                    "terminated_early": False,
                },
            }

            with patch.object(sys, "argv", ["run_simulation.py", "--config", str(cfg_path)]):
                with patch("run_simulation.run_simulation_config_file", return_value=service_payload) as run_file:
                    run_simulation.main()

            run_file.assert_called_once()
            assert Path(run_file.call_args.kwargs["config_path"]).resolve() == cfg_path.resolve()

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

    def test_execution_service_dispatches_serial_monte_carlo_to_campaign_runner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "api_mc.yaml"
            cfg_path.write_text(yaml.safe_dump(_api_config(Path(tmpdir), monte_carlo=True), sort_keys=False), encoding="utf-8")
            payload = {
                "config_path": str(cfg_path.resolve()),
                "scenario_name": "api_smoke",
                "monte_carlo": {"enabled": True, "iterations": 2},
                "aggregate_stats": {"pass_rate": 1.0},
                "runs": [],
            }

            with patch("sim.execution.campaigns.run_monte_carlo_campaign", return_value=payload) as run_campaign:
                out = run_simulation_config_file(cfg_path)

            run_campaign.assert_called_once()
            assert Path(run_campaign.call_args.kwargs["config_path"]).resolve() == cfg_path.resolve()
            assert out["monte_carlo"]["enabled"] is True
            assert out["aggregate_stats"]["pass_rate"] == 1.0

    def test_campaign_runner_prepares_serial_monte_carlo_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "out"
            cfg = SimulationConfig.from_dict(_api_config(outdir, monte_carlo=True)).to_scenario_config()

            prepared = prepare_monte_carlo_runs(cfg=cfg, root=cfg.to_dict(), outdir=outdir)

            assert [int(item["iteration"]) for item in prepared] == [0, 1]
            assert [int(item["seed"]) for item in prepared] == [123, 123]
            assert prepared[0]["config_dict"]["outputs"]["output_dir"] == str(outdir / "mc_run_0000")
            assert prepared[1]["config_dict"]["outputs"]["output_dir"] == str(outdir / "mc_run_0001")

    def test_campaign_runner_executes_serial_monte_carlo_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "out"
            cfg = SimulationConfig.from_dict(_api_config(outdir, monte_carlo=True)).to_scenario_config()
            callback_events: list[tuple[int, int]] = []

            result = run_serial_monte_carlo_runs(
                cfg=cfg,
                root=cfg.to_dict(),
                outdir=outdir,
                strict_plugins=True,
                batch_callback=lambda done, total: callback_events.append((done, total)),
            )

            assert result["parallel_active"] is False
            assert sorted(result["completed"].keys()) == [0, 1]
            assert result["completed"][0]["summary"]["samples"] == 3
            assert callback_events == [(1, 2), (2, 2)]

    def test_monte_carlo_reporting_builds_aggregate_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            cfg = SimulationConfig.from_dict(_api_config(root_dir / "out", monte_carlo=True)).to_scenario_config()
            runs = [
                {
                    "iteration": 0,
                    "sampled_parameters": {},
                    "summary": {
                        "scenario_name": "api_smoke",
                        "duration_s": 2.0,
                        "terminated_early": False,
                        "thrust_stats": {},
                    },
                    "closest_approach_km": float("nan"),
                    "assessment": {},
                }
            ]
            run_details = [
                {
                    "iteration": 0,
                    "seed": 123,
                    "pass": True,
                    "fail_reasons": [],
                    "duration_s": 2.0,
                    "closest_approach_km": float("nan"),
                    "guardrail_events": 0,
                    "total_dv_m_s_total": 0.0,
                    "total_dv_m_s_by_object": {},
                    "delta_v_remaining_m_s_by_object": {},
                }
            ]

            report = build_monte_carlo_report_payload(
                cfg=cfg,
                config_path=root_dir / "mc.yaml",
                root=cfg.to_dict(),
                repo_root=root_dir,
                runs=runs,
                run_details=run_details,
                closest_approach_km_runs=[float("nan")],
                duration_runs_s=[2.0],
                total_dv_runs_m_s=[0.0],
                guardrail_event_runs=[0],
                failure_mode_counts={},
                dv_budget_m_s_by_object={},
                gates={},
                mc_out_cfg={},
                varies_metadata_seed=False,
                parallel_active=False,
                parallel_enabled=False,
                total_iters=1,
                parallel_workers=1,
            )

            assert report["agg"]["monte_carlo"]["enabled"] is True
            assert report["agg"]["aggregate_stats"]["pass_rate"] == 1.0
            assert report["commander_brief"]["runs"] == 2
            assert report["analyst_pack"]["run_details"][0]["seed"] == 123

    def test_monte_carlo_reporting_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "out"
            outdir.mkdir()
            cfg = SimulationConfig.from_dict(_api_config(outdir, monte_carlo=True)).to_scenario_config()
            agg = {
                "scenario_name": "api_smoke",
                "artifacts": {},
                "aggregate_stats": {"pass_rate": 1.0},
            }
            commander = {
                "scenario_name": "api_smoke",
                "runs": 2,
                "p_success": 1.0,
                "p_fail": 0.0,
                "top_failure_modes": [],
            }
            analyst = {"scenario_name": "api_smoke", "run_details": []}

            out = write_monte_carlo_report_artifacts(
                cfg=cfg,
                outdir=outdir,
                agg=agg,
                commander_brief=commander,
                analyst_pack=analyst,
                run_details=[{"iteration": 0}],
                mc_out_cfg={"save_aggregate_summary": True, "save_raw_runs": True},
            )

            artifacts = out["artifacts"]
            assert Path(artifacts["summary_json"]).exists()
            assert Path(artifacts["commander_brief_json"]).exists()
            assert Path(artifacts["commander_brief_md"]).exists()
            assert Path(artifacts["analyst_pack_json"]).exists()
            assert Path(artifacts["run_details_json"]).exists()

    def test_monte_carlo_reporting_applies_baseline_comparison_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "mc.yaml"
            cfg_path.write_text("scenario_name: missing_baseline\n", encoding="utf-8")
            agg = {"aggregate_stats": {}, "commander_brief": {}}
            commander: dict[str, object] = {}

            out = apply_monte_carlo_baseline_comparison(
                agg=agg,
                commander_brief=commander,
                config_path=cfg_path,
                baseline_summary_json="does_not_exist.json",
            )

            assert "baseline_comparison_error" in out

    def test_monte_carlo_plot_writer_noops_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            cfg = SimulationConfig.from_dict(_api_config(outdir, monte_carlo=True)).to_scenario_config()
            agg = {"artifacts": {}}

            out = write_monte_carlo_plot_artifacts(
                cfg=cfg,
                outdir=outdir,
                agg=agg,
                runs=[],
                run_details=[],
                relative_range_series_runs=[],
                durations_s=np.array([], dtype=float),
                ca_finite=np.array([], dtype=float),
                all_obj_ids=[],
                dv_by_object={},
                dv_remaining_m_s_by_object={},
                dv_budget_m_s_by_object={},
                failure_mode_counts={},
                keepout_threshold=float("nan"),
                gates={},
                mc_out_cfg={"save_ops_dashboard": False, "display_ops_dashboard": False},
            )

            assert out["artifacts"] == {}

    def test_sensitivity_runner_executes_serial_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimulationConfig.from_dict(_sensitivity_api_config(Path(tmpdir))).to_scenario_config()
            prepared = [
                {
                    "iteration": 0,
                    "config_dict": cfg.to_dict(),
                }
            ]
            prepared[0]["config_dict"]["analysis"]["enabled"] = False

            result = run_sensitivity_runs(
                cfg=cfg,
                prepared=prepared,
                strict_plugins=True,
            )

            assert result["parallel_active"] is False
            assert result["completed"][0]["summary"]["samples"] == 3

    def test_sensitivity_prep_and_artifact_helpers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "out"
            outdir.mkdir()
            cfg = SimulationConfig.from_dict(_sensitivity_api_config(outdir)).to_scenario_config()

            prepared = prepare_sensitivity_runs(
                cfg=cfg,
                root=cfg.to_dict(),
                outdir=outdir,
                sensitivity_method="one_at_a_time",
            )
            payload = write_sensitivity_summary_artifact(
                outdir=outdir,
                payload={"scenario_name": "api_smoke", "artifacts": {}},
            )

            assert len(prepared) == 2
            assert Path(payload["artifacts"]["summary_json"]).exists()

    def test_sensitivity_reporting_builds_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimulationConfig.from_dict(_sensitivity_api_config(Path(tmpdir))).to_scenario_config()
            prepared = [
                {
                    "iteration": 0,
                    "parameter_path": "simulator.dt_s",
                    "parameter_value": 1.0,
                    "value_index": 0,
                    "sampled_parameters": {"simulator.dt_s": 1.0},
                }
            ]
            completed = {
                0: {
                    "summary": {"duration_s": 2.0, "terminated_early": False},
                    "closest_approach_km": float("nan"),
                    "payload": {"summary": {"duration_s": 2.0, "terminated_early": False}},
                }
            }

            payload = build_sensitivity_report_payload(
                cfg=cfg,
                config_path=Path(tmpdir) / "sensitivity.yaml",
                prepared=prepared,
                completed=completed,
                baseline=None,
                metric_paths=["summary.duration_s"],
                sensitivity_method="one_at_a_time",
                parallel_enabled=False,
                parallel_active=False,
                parallel_workers=1,
                parallel_fallback_reason=None,
            )

            assert payload["analysis"]["run_count"] == 1
            assert payload["runs"][0]["metrics"]["summary.duration_s"] == 2.0
            assert payload["parameter_rankings"][0]["parameter_path"] == "simulator.dt_s"

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

    def test_session_exposes_target_reference_orbit_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimulationConfig.from_dict(_target_reference_orbit_api_config(Path(tmpdir)))

            with patch("sim.runtime_support.EARTH_MU_KM3_S2", 0.0):
                result = SimulationSession.from_config(cfg).run()

            assert result.summary["target_reference_orbit_enabled"] is True
            assert result.target_reference_orbit.shape == (3, 6)
            assert np.allclose(result.target_reference_orbit[0, :3], [7000.0, 0.0, 0.0])
            assert np.allclose(result.target_reference_orbit[-1, :3], [7000.0, 0.0, 0.0])
            assert not np.allclose(result.truth["target"][-1, :3], result.target_reference_orbit[-1, :3])
