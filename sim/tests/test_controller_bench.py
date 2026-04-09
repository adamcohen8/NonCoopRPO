from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml

from sim.config import scenario_config_from_dict
from sim.controller_lab import load_controller_bench_config, run_controller_bench
from sim.controller_lab.models import ControllerBenchConfig
from sim.master_simulator import _run_single_config


def _attitude_case_dict(output_dir: str) -> dict:
    return {
        "scenario_name": "bench_case",
        "rocket": {"enabled": False},
        "chaser": {"enabled": False},
        "target": {
            "enabled": True,
            "specs": {
                "mass_kg": 120.0,
                "mass_properties": {
                    "inertia_kg_m2": [[12.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 8.0]],
                },
            },
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
                "attitude_quat_bn": [0.9063078, 0.0, 0.4226183, 0.0],
                "angular_rate_body_rad_s": [0.02, -0.015, 0.01],
            },
            "attitude_control": {
                "module": "sim.control.attitude.baseline",
                "class_name": "ReactionWheelPDController",
                "params": {
                    "wheel_axes_body": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "wheel_torque_limits_nm": [0.05, 0.05, 0.05],
                    "desired_attitude_quat_bn": [1.0, 0.0, 0.0, 0.0],
                    "desired_rate_body_rad_s": [0.0, 0.0, 0.0],
                    "kp": [0.18, 0.18, 0.18],
                    "kd": [2.0, 2.0, 2.0],
                },
            },
        },
        "simulator": {
            "duration_s": 5.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": True}},
        },
        "outputs": {
            "output_dir": output_dir,
            "mode": "save",
            "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
        "monte_carlo": {"enabled": False},
    }


def _rendezvous_case_dict(output_dir: str) -> dict:
    return {
        "scenario_name": "rendezvous_bench_case",
        "rocket": {"enabled": False},
        "chaser": {
            "enabled": True,
            "specs": {
                "preset_satellite": "BASIC_SATELLITE",
                "dry_mass_kg": 180.0,
                "fuel_mass_kg": 20.0,
                "thruster": "BASIC_CHEMICAL_Z_BOTTOM",
                "attitude_system": "BASIC_REACTION_WHEELS_3AXIS",
            },
            "initial_state": {
                "relative_to_target_ric": {"frame": "curv", "state": [0.0, -3.0, 0.0, 0.0, 0.0, 0.0]},
            },
            "orbit_control": {
                "module": "sim.control.orbit.lqr",
                "class_name": "HCWLQRController",
                "params": {
                    "mean_motion_rad_s": 0.001078,
                    "max_accel_km_s2": 2.0e-5,
                    "design_dt_s": 1.0,
                    "ric_curv_state_slice": [0, 6],
                    "chief_eci_state_slice": [6, 12],
                },
            },
        },
        "target": {
            "enabled": True,
            "specs": {
                "preset_satellite": "BASIC_SATELLITE",
                "dry_mass_kg": 360.0,
                "fuel_mass_kg": 0.0,
            },
            "initial_state": {
                "coes": {
                    "a_km": 7000.0,
                    "ecc": 0.0,
                    "inc_deg": 45.0,
                    "raan_deg": 0.0,
                    "argp_deg": 0.0,
                    "true_anomaly_deg": 0.0,
                }
            },
            "orbit_control": {
                "module": "sim.control.orbit.zero_controller",
                "class_name": "ZeroController",
                "params": {},
            },
        },
        "simulator": {
            "duration_s": 30.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": output_dir,
            "mode": "save",
            "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
        "monte_carlo": {"enabled": False},
    }


def test_single_run_payload_includes_controller_debug_trace() -> None:
    cfg = scenario_config_from_dict(_attitude_case_dict("outputs/test_controller_debug"))

    payload = _run_single_config(cfg)

    debug = list(payload["controller_debug_by_object"]["target"])
    assert debug
    first = dict(debug[0])
    assert "controller_runtime_ms" in first
    assert "command_raw" in first
    assert "command_applied" in first
    assert first["attitude_belief"] is not None


def test_controller_bench_runs_and_writes_reports() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        case_path = root / "case.yaml"
        case_path.write_text(yaml.safe_dump(_attitude_case_dict(str(root / "case_out")), sort_keys=False), encoding="utf-8")

        suite_path = root / "suite.yaml"
        suite_path.write_text(
            yaml.safe_dump(
                {
                    "suite_name": "attitude_bench_test",
                    "output_dir": str(root / "bench_out"),
                    "parallel_enabled": True,
                    "parallel_workers": 2,
                    "controller_target": {"object_id": "target", "slot": "attitude_control"},
                    "variants": [
                        {
                            "name": "baseline",
                            "controller": {
                                "module": "sim.control.attitude.baseline",
                                "class_name": "ReactionWheelPDController",
                                "params": {
                                    "wheel_axes_body": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                    "wheel_torque_limits_nm": [0.05, 0.05, 0.05],
                                    "desired_attitude_quat_bn": [1.0, 0.0, 0.0, 0.0],
                                    "desired_rate_body_rad_s": [0.0, 0.0, 0.0],
                                    "kp": [0.18, 0.18, 0.18],
                                    "kd": [2.0, 2.0, 2.0],
                                },
                            },
                        },
                        {
                            "name": "tuned",
                            "controller": {
                                "module": "sim.control.attitude.baseline",
                                "class_name": "ReactionWheelPDController",
                                "params": {
                                    "wheel_axes_body": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                    "wheel_torque_limits_nm": [0.05, 0.05, 0.05],
                                    "desired_attitude_quat_bn": [1.0, 0.0, 0.0, 0.0],
                                    "desired_rate_body_rad_s": [0.0, 0.0, 0.0],
                                    "kp": [0.3, 0.3, 0.3],
                                    "kd": [3.0, 3.0, 3.0],
                                },
                            },
                        },
                    ],
                    "metrics": [
                        {
                            "name": "final_attitude_error_deg",
                            "kind": "final_attitude_error_deg",
                            "object_id": "target",
                            "desired_quat_bn": [1.0, 0.0, 0.0, 0.0],
                        },
                        {
                            "name": "mean_controller_runtime_ms",
                            "kind": "mean_controller_runtime_ms",
                            "object_id": "target",
                        },
                    ],
                    "pass_criteria": [
                        {"metric": "final_attitude_error_deg", "op": "<=", "value": 45.0},
                    ],
                    "cases": [
                        {"name": "nominal", "config_path": str(case_path)},
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        suite = load_controller_bench_config(suite_path)
        assert suite.suite_name == "attitude_bench_test"

        result = run_controller_bench(suite_path)

        assert len(result["runs"]) == 2
        assert len(result["variant_summaries"]) == 2
        assert any(dict(v).get("linear_system_summary") is None for v in result["variants"])
        assert bool(result["execution"]["parallel_requested"])
        if bool(result["execution"]["parallel_enabled"]):
            assert int(result["execution"]["parallel_workers"]) >= 1
            assert result["execution"]["parallel_fallback_reason"] is None
        else:
            assert result["execution"]["parallel_fallback_reason"]
        assert Path(result["artifacts"]["summary_json"]).exists()
        assert Path(result["artifacts"]["summary_md"]).exists()
        assert Path(result["artifacts"]["comparison_csv"]).exists()
        assert Path(result["artifacts"]["leaderboard_csv"]).exists()


def test_controller_bench_supports_rendezvous_objectives_and_metrics() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        case_path = root / "rendezvous_case.yaml"
        case_path.write_text(
            yaml.safe_dump(_rendezvous_case_dict(str(root / "rendezvous_case_out")), sort_keys=False),
            encoding="utf-8",
        )

        suite_path = root / "rendezvous_suite.yaml"
        suite_path.write_text(
            yaml.safe_dump(
                {
                    "suite_name": "rendezvous_bench_test",
                    "output_dir": str(root / "bench_out"),
                    "controller_target": {"object_id": "chaser", "slot": "orbit_control"},
                    "variants": [
                        {
                            "name": "coast",
                            "controller": {
                                "module": "sim.control.orbit.zero_controller",
                                "class_name": "ZeroController",
                                "params": {},
                            },
                        },
                        {
                            "name": "lqr",
                            "controller": {
                                "module": "sim.control.orbit.lqr",
                                "class_name": "HCWLQRController",
                                "params": {
                                    "mean_motion_rad_s": 0.001078,
                                    "max_accel_km_s2": 2.0e-5,
                                    "design_dt_s": 1.0,
                                    "ric_curv_state_slice": [0, 6],
                                    "chief_eci_state_slice": [6, 12],
                                },
                            },
                        },
                    ],
                    "objectives": [
                        {
                            "kind": "relative_rendezvous",
                            "name": "safe_rendezvous",
                            "object_id": "chaser",
                            "reference_object_id": "target",
                            "keepout_radius_km": 0.5,
                            "max_time_inside_keepout_s": 0.0,
                            "require_not_terminated_early": True,
                        }
                    ],
                    "cases": [
                        {"name": "nominal", "config_path": str(case_path)},
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        suite = load_controller_bench_config(suite_path)
        assert suite.objectives
        assert suite.objectives[0].kind == "relative_rendezvous"

        result = run_controller_bench(suite_path)

        assert len(result["runs"]) == 2
        by_variant = {str(run["variant_name"]): run for run in result["runs"]}
        coast_run = by_variant["coast"]
        lqr_run = by_variant["lqr"]

        for run in (coast_run, lqr_run):
            assert run["objective_results"]
            objective_result = dict(run["objective_results"][0])
            assert objective_result["name"] == "safe_rendezvous"
            assert "safe_rendezvous_final_relative_distance_km" in run["metrics"]
            assert "safe_rendezvous_closest_approach_km" in run["metrics"]
            assert "safe_rendezvous_time_inside_keepout_s" in run["metrics"]
            assert "safe_rendezvous_fuel_used_kg" in run["metrics"]
            assert objective_result["metrics"]["safe_rendezvous_terminated_early"] is False

        assert float(lqr_run["metrics"]["safe_rendezvous_total_dv_m_s"]) > float(
            coast_run["metrics"]["safe_rendezvous_total_dv_m_s"]
        )
        assert float(lqr_run["metrics"]["safe_rendezvous_fuel_used_kg"]) >= 0.0
        assert float(coast_run["metrics"]["safe_rendezvous_time_inside_keepout_s"]) == 0.0
        assert float(lqr_run["metrics"]["safe_rendezvous_time_inside_keepout_s"]) == 0.0
        assert Path(result["artifacts"]["leaderboard_csv"]).exists()

        leaderboards = list(result["leaderboards"]["relative_rendezvous"])
        assert len(leaderboards) == 1
        leaderboard = leaderboards[0]
        assert leaderboard["objective_name"] == "safe_rendezvous"
        ranking_by_metric = {str(item["metric"]): item for item in leaderboard["rankings"]}
        assert "objective_pass_rate" in ranking_by_metric
        assert "final_relative_distance_km" in ranking_by_metric
        ranked_variants = [entry["variant_name"] for entry in ranking_by_metric["objective_pass_rate"]["entries"]]
        assert "coast" in ranked_variants
        assert "lqr" in ranked_variants


def test_controller_bench_writes_linear_feedback_csv_artifacts_for_hcw_lqr() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        case_path = root / "rendezvous_case.yaml"
        case_path.write_text(
            yaml.safe_dump(_rendezvous_case_dict(str(root / "rendezvous_case_out")), sort_keys=False),
            encoding="utf-8",
        )

        suite_path = root / "rendezvous_suite.yaml"
        suite_path.write_text(
            yaml.safe_dump(
                {
                    "suite_name": "rendezvous_feedback_plots",
                    "output_dir": str(root / "bench_out"),
                    "controller_target": {"object_id": "chaser", "slot": "orbit_control"},
                    "variants": [
                        {
                            "name": "lqr",
                            "controller": {
                                "module": "sim.control.orbit.lqr",
                                "class_name": "HCWLQRController",
                                "params": {
                                    "mean_motion_rad_s": 0.001078,
                                    "max_accel_km_s2": 2.0e-5,
                                    "design_dt_s": 1.0,
                                    "ric_curv_state_slice": [0, 6],
                                    "chief_eci_state_slice": [6, 12],
                                },
                            },
                        }
                    ],
                    "cases": [
                        {"name": "nominal", "config_path": str(case_path)},
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = run_controller_bench(suite_path)

        assert len(result["runs"]) == 1
        feedback = dict(result["runs"][0]["artifacts"]["linear_feedback"])
        assert Path(feedback["output_dir"]).exists()
        assert feedback["control_axes"] == ["R", "I", "C"]
        axis_artifacts = dict(feedback["axes"])
        for axis_name in ("R", "I", "C"):
            assert Path(axis_artifacts[axis_name]["csv"]).exists()
        variant_summary = dict(result["variants"][0]["linear_system_summary"])
        assert variant_summary["control_axes"] == ["R", "I", "C"]
        assert len(variant_summary["closed_loop_poles"]) == 6


def test_load_controller_bench_resolves_output_dir_relative_to_suite_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        suite_dir = root / "nested" / "suite_dir"
        suite_dir.mkdir(parents=True)
        case_path = suite_dir / "case.yaml"
        case_path.write_text(yaml.safe_dump(_attitude_case_dict("unused"), sort_keys=False), encoding="utf-8")

        suite_path = suite_dir / "suite.yaml"
        suite_path.write_text(
            yaml.safe_dump(
                {
                    "suite_name": "relative_output_dir_suite",
                    "output_dir": "bench_outputs",
                    "controller_target": {"object_id": "target", "slot": "attitude_control"},
                    "variants": [
                        {
                            "name": "baseline",
                            "controller": {
                                "module": "sim.control.attitude.baseline",
                                "class_name": "ReactionWheelPDController",
                                "params": {},
                            },
                        }
                    ],
                    "cases": [
                        {"name": "nominal", "config_path": str(case_path.name)},
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        suite = load_controller_bench_config(suite_path)

        assert suite.output_dir == (suite_dir / "bench_outputs").resolve()
        assert suite.plot_mode == "save"


def test_load_controller_bench_parses_plot_mode() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        case_path = root / "case.yaml"
        case_path.write_text(yaml.safe_dump(_attitude_case_dict("unused"), sort_keys=False), encoding="utf-8")

        suite_path = root / "suite.yaml"
        suite_path.write_text(
            yaml.safe_dump(
                {
                    "suite_name": "interactive_plot_mode_suite",
                    "plot_mode": "interactive",
                    "controller_target": {"object_id": "target", "slot": "attitude_control"},
                    "variants": [
                        {
                            "name": "baseline",
                            "controller": {
                                "module": "sim.control.attitude.baseline",
                                "class_name": "ReactionWheelPDController",
                                "params": {},
                            },
                        }
                    ],
                    "cases": [
                        {"name": "nominal", "config_path": str(case_path)},
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        suite = load_controller_bench_config(suite_path)

        assert suite.plot_mode == "interactive"


def test_load_controller_bench_expands_sweep_variants() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        case_path = root / "case.yaml"
        case_path.write_text(yaml.safe_dump(_attitude_case_dict(str(root / "case_out")), sort_keys=False), encoding="utf-8")

        suite_path = root / "suite.yaml"
        suite_path.write_text(
            yaml.safe_dump(
                {
                    "suite_name": "sweep_suite",
                    "output_dir": str(root / "bench_out"),
                    "controller_target": {"object_id": "target", "slot": "attitude_control"},
                    "variants": [
                        {
                            "name": "rw_pd_sweep",
                            "description": "Sweep gains",
                            "controller": {
                                "module": "sim.control.attitude.baseline",
                                "class_name": "ReactionWheelPDController",
                                "params": {
                                    "kp": [0.18, 0.18, 0.18],
                                    "kd": [2.0, 2.0, 2.0],
                                },
                            },
                            "sweep": {
                                "controller.params.kp": [
                                    [0.18, 0.18, 0.18],
                                    [0.30, 0.30, 0.30],
                                ],
                                "controller.params.kd": [
                                    [2.0, 2.0, 2.0],
                                    [3.0, 3.0, 3.0],
                                ],
                            },
                        }
                    ],
                    "cases": [
                        {"name": "nominal", "config_path": str(case_path)},
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        suite = load_controller_bench_config(suite_path)

        assert len(suite.variants) == 4
        combos = {
            (
                tuple(variant.pointer["params"]["kp"]),
                tuple(variant.pointer["params"]["kd"]),
            )
            for variant in suite.variants
        }
        assert combos == {
            ((0.18, 0.18, 0.18), (2.0, 2.0, 2.0)),
            ((0.18, 0.18, 0.18), (3.0, 3.0, 3.0)),
            ((0.3, 0.3, 0.3), (2.0, 2.0, 2.0)),
            ((0.3, 0.3, 0.3), (3.0, 3.0, 3.0)),
        }
        assert all(variant.name.startswith("rw_pd_sweep__") for variant in suite.variants)


def test_controller_bench_omits_run_log_artifact_when_payload_saving_disabled() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        case_path = root / "case.yaml"
        case_path.write_text(yaml.safe_dump(_attitude_case_dict(str(root / "case_out")), sort_keys=False), encoding="utf-8")

        suite_path = root / "suite.yaml"
        suite_path.write_text(
            yaml.safe_dump(
                {
                    "suite_name": "no_payload_artifact_suite",
                    "output_dir": str(root / "bench_out"),
                    "save_run_payloads": False,
                    "controller_target": {"object_id": "target", "slot": "attitude_control"},
                    "variants": [
                        {
                            "name": "baseline",
                            "controller": {
                                "module": "sim.control.attitude.baseline",
                                "class_name": "ReactionWheelPDController",
                                "params": {
                                    "wheel_axes_body": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                    "wheel_torque_limits_nm": [0.05, 0.05, 0.05],
                                    "desired_attitude_quat_bn": [1.0, 0.0, 0.0, 0.0],
                                    "desired_rate_body_rad_s": [0.0, 0.0, 0.0],
                                    "kp": [0.18, 0.18, 0.18],
                                    "kd": [2.0, 2.0, 2.0],
                                },
                            },
                        }
                    ],
                    "metrics": [
                        {
                            "name": "final_attitude_error_deg",
                            "kind": "final_attitude_error_deg",
                            "object_id": "target",
                            "desired_quat_bn": [1.0, 0.0, 0.0, 0.0],
                        }
                    ],
                    "cases": [
                        {"name": "nominal", "config_path": str(case_path)},
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = run_controller_bench(suite_path)

        assert len(result["runs"]) == 1
        assert "run_log_json" not in result["runs"][0]["artifacts"]


def test_controller_bench_variant_metric_means_ignore_nan_values() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        case_path = root / "case.yaml"
        case_path.write_text(yaml.safe_dump(_attitude_case_dict(str(root / "case_out")), sort_keys=False), encoding="utf-8")

        suite_path = root / "suite.yaml"
        suite_path.write_text(
            yaml.safe_dump(
                {
                    "suite_name": "nan_metric_suite",
                    "output_dir": str(root / "bench_out"),
                    "controller_target": {"object_id": "target", "slot": "attitude_control"},
                    "variants": [
                        {
                            "name": "baseline",
                            "controller": {
                                "module": "sim.control.attitude.baseline",
                                "class_name": "ReactionWheelPDController",
                                "params": {},
                            },
                        }
                    ],
                    "cases": [
                        {"name": "case_a", "config_path": str(case_path)},
                        {"name": "case_b", "config_path": str(case_path)},
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        suite = load_controller_bench_config(suite_path)

        from sim.controller_lab.models import ControllerBenchCase, ControllerBenchMetric
        from sim.controller_lab import bench as bench_mod

        metric = ControllerBenchMetric(name="runtime_ms", source_path="summary.runtime_ms")
        suite = ControllerBenchConfig(
            suite_name=suite.suite_name,
            description=suite.description,
            output_dir=suite.output_dir,
            controller_target=suite.controller_target,
            variants=suite.variants,
            cases=tuple(
                ControllerBenchCase(
                    name=case.name,
                    config_path=case.config_path,
                    description=case.description,
                    metrics=(metric,),
                    pass_criteria=case.pass_criteria,
                    objectives=case.objectives,
                )
                for case in suite.cases
            ),
            metrics=suite.metrics,
            pass_criteria=suite.pass_criteria,
            objectives=suite.objectives,
            save_run_payloads=suite.save_run_payloads,
            disable_plots=suite.disable_plots,
            disable_animations=suite.disable_animations,
            print_individual_run_summaries=suite.print_individual_run_summaries,
            parallel_enabled=False,
            parallel_workers=suite.parallel_workers,
        )

        original_loader = bench_mod.load_controller_bench_config
        original_runner = bench_mod._run_single_bench_case
        call_count = {"n": 0}

        def _fake_loader(path):
            return suite

        def _fake_run_single_bench_case(suite, case, variant):
            call_count["n"] += 1
            val = float("nan") if call_count["n"] == 1 else 5.0
            return {
                "variant_name": variant.name,
                "case_name": case.name,
                "description": case.description,
                "metrics": {"runtime_ms": val},
                "passed": True,
                "failed_criteria": [],
                "objective_results": [],
                "artifacts": {"output_dir": str(root / case.name / variant.name)},
                "config_path": str(case.config_path),
            }

        bench_mod.load_controller_bench_config = _fake_loader
        bench_mod._run_single_bench_case = _fake_run_single_bench_case
        try:
            result = run_controller_bench(suite_path)
        finally:
            bench_mod.load_controller_bench_config = original_loader
            bench_mod._run_single_bench_case = original_runner

        assert len(result["variant_summaries"]) == 1
        assert result["variant_summaries"][0]["metric_means"]["runtime_ms"] == 5.0


def test_controller_bench_invokes_visualization_render_with_configured_plot_mode() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        case_path = root / "case.yaml"
        case_path.write_text(yaml.safe_dump(_attitude_case_dict(str(root / "case_out")), sort_keys=False), encoding="utf-8")

        suite_path = root / "suite.yaml"
        suite_path.write_text(
            yaml.safe_dump(
                {
                    "suite_name": "both_plot_mode_suite",
                    "plot_mode": "both",
                    "output_dir": str(root / "bench_out"),
                    "controller_target": {"object_id": "target", "slot": "attitude_control"},
                    "variants": [
                        {
                            "name": "baseline",
                            "controller": {
                                "module": "sim.control.attitude.baseline",
                                "class_name": "ReactionWheelPDController",
                                "params": {},
                            },
                        }
                    ],
                    "cases": [
                        {"name": "nominal", "config_path": str(case_path)},
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        from sim.controller_lab import bench as bench_mod

        original_render = bench_mod.render_controller_bench_visualizations
        seen: dict[str, object] = {}

        def _fake_render(result, outdir, mode):
            seen["mode"] = mode
            seen["outdir"] = outdir
            return {"pass_rate_plot_png": str(Path(outdir) / "pass_rates.png")}

        bench_mod.render_controller_bench_visualizations = _fake_render
        try:
            result = run_controller_bench(suite_path)
        finally:
            bench_mod.render_controller_bench_visualizations = original_render

        assert seen["mode"] == "both"
        assert Path(str(seen["outdir"])) == (root / "bench_out")
        assert result["artifacts"]["pass_rate_plot_png"].endswith("pass_rates.png")
        saved = json.loads(Path(result["artifacts"]["summary_json"]).read_text(encoding="utf-8"))
        assert saved["artifacts"]["pass_rate_plot_png"].endswith("pass_rates.png")
        assert saved["artifacts"]["summary_json"] == result["artifacts"]["summary_json"]
