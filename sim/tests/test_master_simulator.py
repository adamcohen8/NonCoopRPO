from __future__ import annotations

import json
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from sim.config import scenario_config_from_dict
from sim.master_simulator import _run_single_config, run_master_simulation


class ConstantIntegratedThrustMission:
    def __init__(self, thrust_eci_km_s2: list[float] | tuple[float, float, float]):
        self.thrust_eci_km_s2 = np.array(thrust_eci_km_s2, dtype=float)

    def update(self, **kwargs):
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": self.thrust_eci_km_s2.copy(),
        }


class TrackTargetXMission:
    def __init__(self, gain_km_s2_per_km: float = 1.0):
        self.gain_km_s2_per_km = float(gain_km_s2_per_km)

    def update(self, *, object_id, truth, world_truth, **kwargs):
        target_truth = world_truth.get("target")
        if target_truth is None or object_id == "target":
            return {}
        accel_x = self.gain_km_s2_per_km * float(target_truth.position_eci_km[0])
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": np.array([accel_x, 0.0, 0.0], dtype=float),
        }


class TimeSplitThrustMission:
    def __init__(self, split_time_s: float, low_km_s2: float, high_km_s2: float):
        self.split_time_s = float(split_time_s)
        self.low_km_s2 = float(low_km_s2)
        self.high_km_s2 = float(high_km_s2)

    def update(self, *, t_s, **kwargs):
        accel_x = self.low_km_s2 if float(t_s) <= self.split_time_s else self.high_km_s2
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": np.array([accel_x, 0.0, 0.0], dtype=float),
        }


class TestMasterSimulator(unittest.TestCase):
    def test_master_runner_updates_knowledge_and_world_truth_after_propagation(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "world_truth_live",
                "rocket": {"enabled": False},
                "target": {
                    "enabled": True,
                    "specs": {"mass_kg": 100.0},
                    "initial_state": {
                        "position_eci_km": [1.0, 0.0, 0.0],
                        "velocity_eci_km_s": [1.0, 0.0, 0.0],
                    },
                    "mission_objectives": [
                        {
                            "module": "sim.tests.test_master_simulator",
                            "class_name": "ConstantIntegratedThrustMission",
                            "params": {"thrust_eci_km_s2": [1.0, 0.0, 0.0]},
                        }
                    ],
                },
                "chaser": {
                    "enabled": True,
                    "specs": {"mass_kg": 100.0},
                    "initial_state": {
                        "position_eci_km": [10.0, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 0.0, 0.0],
                    },
                    "mission_objectives": [
                        {
                            "module": "sim.tests.test_master_simulator",
                            "class_name": "TrackTargetXMission",
                            "params": {"gain_km_s2_per_km": 1.0},
                        }
                    ],
                    "knowledge": {
                        "targets": ["target"],
                        "refresh_rate_s": 1.0,
                        "sensor_error": {
                            "pos_sigma_km": [0.0, 0.0, 0.0],
                            "vel_sigma_km_s": [0.0, 0.0, 0.0],
                        },
                    },
                },
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "termination": {"earth_impact_enabled": False},
                    "dynamics": {"attitude": {"enabled": False}},
                },
                "outputs": {
                    "output_dir": "outputs/test_world_truth_live",
                    "mode": "save",
                    "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
                    "plots": {"enabled": False, "figure_ids": []},
                    "animations": {"enabled": False, "types": []},
                },
                "monte_carlo": {"enabled": False},
            }
        )

        with patch("sim.master_simulator.EARTH_MU_KM3_S2", 0.0):
            payload = _run_single_config(cfg)

        target_truth = np.array(payload["truth_by_object"]["target"], dtype=float)
        chaser_truth = np.array(payload["truth_by_object"]["chaser"], dtype=float)
        chaser_knowledge = np.array(payload["knowledge_by_observer"]["chaser"]["target"], dtype=float)

        self.assertAlmostEqual(float(target_truth[-1, 0]), 2.5, places=9)
        self.assertAlmostEqual(float(chaser_truth[-1, 3]), 2.5, places=9)
        self.assertAlmostEqual(float(chaser_knowledge[-1, 0]), 2.5, places=9)

    def test_master_runner_accumulates_delta_v_across_substeps(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "delta_v_substeps",
                "rocket": {"enabled": False},
                "chaser": {"enabled": False},
                "target": {
                    "enabled": True,
                    "specs": {"mass_kg": 100.0},
                    "initial_state": {
                        "position_eci_km": [1.0, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 0.0, 0.0],
                    },
                    "mission_objectives": [
                        {
                            "module": "sim.tests.test_master_simulator",
                            "class_name": "TimeSplitThrustMission",
                            "params": {"split_time_s": 0.5, "low_km_s2": 1.0, "high_km_s2": 3.0},
                        }
                    ],
                },
                "simulator": {
                    "duration_s": 1.0,
                    "dt_s": 1.0,
                    "termination": {"earth_impact_enabled": False},
                    "dynamics": {
                        "attitude": {"enabled": False},
                        "orbit": {"orbit_substep_s": 0.5},
                    },
                },
                "outputs": {
                    "output_dir": "outputs/test_delta_v_substeps",
                    "mode": "save",
                    "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
                    "plots": {"enabled": False, "figure_ids": []},
                    "animations": {"enabled": False, "types": []},
                },
                "monte_carlo": {"enabled": False},
            }
        )

        with patch("sim.master_simulator.EARTH_MU_KM3_S2", 0.0):
            payload = _run_single_config(cfg)

        summary = dict(payload["summary"])
        thrust_stats = dict(summary["thrust_stats"]["target"])
        thrust_hist = np.array(payload["applied_thrust_by_object"]["target"], dtype=float)

        self.assertAlmostEqual(float(thrust_stats["total_dv_m_s"]), 2000.0, places=9)
        self.assertEqual(int(thrust_stats["burn_samples"]), 1)
        self.assertAlmostEqual(float(thrust_stats["max_accel_km_s2"]), 3.0, places=9)
        self.assertAlmostEqual(float(thrust_hist[-1, 0]), 2.0, places=9)

    def test_master_runner_executes_rocket_ascent_from_yaml(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")

        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "out"
            cfg_path = Path(td) / "cfg.yaml"
            cfg_path.write_text(
                """
scenario_name: "master_smoke"
rocket:
  enabled: true
  specs:
    payload_mass_kg: 50.0
    thrust_axis_body: [1.0, 0.0, 0.0]
  initial_state:
    launch_lat_deg: 28.5
    launch_lon_deg: -80.6
    launch_alt_km: 0.0
    launch_azimuth_deg: 90.0
target:
  enabled: false
chaser:
  enabled: false
simulator:
  scenario_type: "rocket_ascent"
  duration_s: 8.0
  dt_s: 1.0
  dynamics:
    rocket:
      atmosphere_model: "ussa1976"
  termination:
    earth_impact_enabled: true
    earth_radius_km: 6378.137
outputs:
  output_dir: "{outdir}"
  mode: "interactive"
  stats:
    save_json: true
monte_carlo:
  enabled: false
                """.strip().format(outdir=str(outdir).replace("\\", "\\\\")),
                encoding="utf-8",
            )
            res = run_master_simulation(cfg_path)
            self.assertEqual(res["scenario_name"], "master_smoke")
            self.assertIn("run", res)
            self.assertIn("objects", res["run"])
            self.assertTrue((outdir / "master_run_summary.json").exists())

    def test_master_runner_executes_rocket_ascent_with_tvc_wrapper_fields(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")

        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "out_tvc"
            cfg_path = Path(td) / "cfg_tvc.yaml"
            cfg_path.write_text(
                """
scenario_name: "master_tvc_smoke"
rocket:
  enabled: true
  specs:
    payload_mass_kg: 50.0
    thrust_axis_body: [1.0, 0.0, 0.0]
  initial_state:
    launch_lat_deg: 28.5
    launch_lon_deg: -80.6
    launch_alt_km: 0.0
    launch_azimuth_deg: 90.0
  guidance:
    module: "sim.rocket.guidance"
    class_name: "OpenLoopPitchProgramGuidance"
    params:
      vertical_hold_s: 2.0
      pitch_start_s: 2.0
      pitch_end_s: 20.0
      pitch_final_deg: 45.0
      max_throttle: 1.0
target:
  enabled: false
chaser:
  enabled: false
simulator:
  scenario_type: "rocket_ascent"
  duration_s: 8.0
  dt_s: 1.0
  dynamics:
    rocket:
      atmosphere_model: "ussa1976"
      attitude_mode: "dynamic"
      use_wgs84_geodesy: true
      wind_enu_m_s: [5.0, 0.0, 0.0]
      tvc_steering_enabled: true
      tvc_pass_through_attitude: true
      tvc_time_constant_s: 0.2
      tvc_max_gimbal_deg: 5.0
      tvc_rate_limit_deg_s: 8.0
      tvc_pivot_offset_body_m: [-3.0, 0.0, 0.0]
  termination:
    earth_impact_enabled: true
    earth_radius_km: 6378.137
outputs:
  output_dir: "{outdir}"
  mode: "save"
  stats:
    save_json: true
  plots:
    enabled: false
  animations:
    enabled: false
monte_carlo:
  enabled: false
                """.strip().format(outdir=str(outdir).replace("\\", "\\\\")),
                encoding="utf-8",
            )
            res = run_master_simulation(cfg_path)
            self.assertEqual(res["scenario_name"], "master_tvc_smoke")
            self.assertTrue((outdir / "master_run_summary.json").exists())

    def test_master_runner_fails_fast_on_invalid_plugin_when_strict(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "bad_cfg.yaml"
            cfg_path.write_text(
                """
scenario_name: "bad_plugins"
rocket:
  enabled: true
  guidance:
    module: "sim.control.orbit.zero_controller"
    class_name: "ZeroController"
    params: {}
chaser:
  enabled: false
target:
  enabled: false
simulator:
  duration_s: 10.0
  dt_s: 1.0
  plugin_validation:
    strict: true
outputs:
  output_dir: "outputs/bad_plugins"
  mode: "save"
monte_carlo:
  enabled: false
                """.strip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _ = run_master_simulation(cfg_path)

    def test_master_runner_monte_carlo_parallel_mode_executes(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")

        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "out_parallel_mc"
            cfg_path = Path(td) / "cfg_parallel_mc.yaml"
            cfg_path.write_text(
                """
scenario_name: "parallel_mc_smoke"
rocket:
  enabled: false
chaser:
  enabled: true
  specs:
    preset_satellite: "BASIC_SATELLITE"
    dry_mass_kg: 180.0
    fuel_mass_kg: 20.0
    thruster: "BASIC_CHEMICAL_Z_BOTTOM"
    attitude_system: "BASIC_REACTION_WHEELS_3AXIS"
  initial_state:
    relative_to_target_ric:
      frame: "curv"
      state: [0.0, -3.0, 0.0, 0.0, 0.0, 0.0]
  guidance:
    module: "sim.control.orbit.zero_controller"
    class_name: "ZeroController"
  orbit_control:
    module: "sim.control.orbit.lqr"
    class_name: "HCWLQRController"
    params:
      mean_motion_rad_s: 0.001078
      max_accel_km_s2: 2.0e-5
      design_dt_s: 1.0
      ric_curv_state_slice: [0, 6]
      chief_eci_state_slice: [6, 12]
  attitude_control:
    module: "sim.control.attitude.surrogate_snap"
    class_name: "SurrogateSnapECIController"
target:
  enabled: true
  specs:
    preset_satellite: "BASIC_SATELLITE"
    dry_mass_kg: 360.0
    fuel_mass_kg: 0.0
  initial_state:
    coes:
      a_km: 7000.0
      ecc: 0.0
      inc_deg: 45.0
      raan_deg: 0.0
      argp_deg: 0.0
      true_anomaly_deg: 0.0
  guidance:
    module: "sim.control.orbit.zero_controller"
    class_name: "ZeroController"
  orbit_control:
    module: "sim.control.orbit.zero_controller"
    class_name: "ZeroController"
  attitude_control:
    module: "sim.control.attitude.surrogate_snap"
    class_name: "SurrogateSnapECIController"
simulator:
  duration_s: 20.0
  dt_s: 1.0
outputs:
  output_dir: "{outdir}"
  mode: "save"
  stats:
    print_summary: false
    save_json: false
    save_full_log: false
  plots:
    enabled: false
    figure_ids: []
  animations:
    enabled: false
    types: []
  monte_carlo:
    save_iteration_summaries: false
    save_aggregate_summary: false
    save_histograms: false
    display_histograms: false
    save_ops_dashboard: false
    display_ops_dashboard: false
monte_carlo:
  enabled: true
  iterations: 2
  base_seed: 42
  parallel_enabled: true
  parallel_workers: 2
  variations:
    - parameter_path: "chaser.initial_state.relative_to_target_ric.state[1]"
      mode: "normal"
      mean: -3.0
      std: 0.1
                """.strip().format(outdir=str(outdir).replace("\\", "\\\\")),
                encoding="utf-8",
            )
            res = run_master_simulation(cfg_path)
            self.assertTrue(bool(res.get("monte_carlo", {}).get("parallel_requested", False)))
            self.assertGreaterEqual(int(res.get("monte_carlo", {}).get("parallel_workers", 0)), 1)
            self.assertEqual(len(list(res.get("runs", []) or [])), 2)

    def test_attitude_disabled_runs_orbital_only_for_satellite(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")

        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "out_att_off"
            cfg_path = Path(td) / "cfg_att_off.yaml"
            cfg_path.write_text(
                """
scenario_name: "attitude_disabled_smoke"
rocket:
  enabled: false
chaser:
  enabled: true
  specs:
    preset_satellite: "BASIC_SATELLITE"
    dry_mass_kg: 150.0
    fuel_mass_kg: 10.0
    thruster: "BASIC_CHEMICAL_Z_BOTTOM"
    attitude_system: "BASIC_REACTION_WHEELS_3AXIS"
  initial_state:
    relative_to_target_ric:
      frame: "curv"
      state: [0.0, -2.0, 0.0, 0.0, 0.0, 0.0]
  orbit_control:
    module: "sim.control.orbit.zero_controller"
    class_name: "ZeroController"
  attitude_control:
    module: "sim.control.attitude.surrogate_snap"
    class_name: "SurrogateSnapECIController"
target:
  enabled: true
  specs:
    preset_satellite: "BASIC_SATELLITE"
    dry_mass_kg: 300.0
    fuel_mass_kg: 10.0
  initial_state:
    coes:
      a_km: 7000.0
      ecc: 0.0
      inc_deg: 35.0
      raan_deg: 5.0
      argp_deg: 0.0
      true_anomaly_deg: 0.0
  orbit_control:
    module: "sim.control.orbit.zero_controller"
    class_name: "ZeroController"
  attitude_control:
    module: "sim.control.attitude.surrogate_snap"
    class_name: "SurrogateSnapECIController"
  mission_objectives:
    - module: "sim.mission.modules"
      class_name: "SatelliteMissionModule"
      params:
        orbital_mode: "pursuit_blind"
        attitude_mode: "hold_eci"
        max_accel_km_s2: 2.0e-6
        blind_direction_eci: [1.0, 0.0, 0.0]
simulator:
  duration_s: 5.0
  dt_s: 1.0
  dynamics:
    attitude:
      enabled: false
outputs:
  output_dir: "{outdir}"
  mode: "save"
  stats:
    print_summary: false
    save_json: false
    save_full_log: true
  plots:
    enabled: false
    figure_ids: []
  animations:
    enabled: false
    types: []
monte_carlo:
  enabled: false
                """.strip().format(outdir=str(outdir).replace("\\", "\\\\")),
                encoding="utf-8",
            )
            _ = run_master_simulation(cfg_path)
            payload = json.loads((outdir / "master_run_log.json").read_text(encoding="utf-8"))

            target_truth = np.array(payload["truth_by_object"]["target"], dtype=float)
            target_torque = np.array(payload["applied_torque_by_object"]["target"], dtype=float)
            target_thrust = np.array(payload["applied_thrust_by_object"]["target"], dtype=float)

            q0 = target_truth[0, 6:10]
            w0 = target_truth[0, 10:13]
            self.assertTrue(np.allclose(target_truth[:, 6:10], q0, atol=1e-12))
            self.assertTrue(np.allclose(target_truth[:, 10:13], w0, atol=1e-12))
            self.assertTrue(np.allclose(target_torque[1:, :], 0.0, atol=1e-12))
            self.assertGreater(float(np.max(np.linalg.norm(np.nan_to_num(target_thrust, nan=0.0), axis=1))), 0.0)

    def test_satellite_out_of_fuel_cannot_maneuver(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")

        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "out_no_fuel"
            cfg_path = Path(td) / "cfg_no_fuel.yaml"
            cfg_path.write_text(
                """
scenario_name: "satellite_no_fuel_cutoff"
rocket:
  enabled: false
chaser:
  enabled: false
target:
  enabled: true
  specs:
    preset_satellite: "BASIC_SATELLITE"
    dry_mass_kg: 300.0
    fuel_mass_kg: 0.0
    thruster: "BASIC_CHEMICAL_Z_BOTTOM"
  initial_state:
    coes:
      a_km: 7000.0
      ecc: 0.0
      inc_deg: 35.0
      raan_deg: 5.0
      argp_deg: 0.0
      true_anomaly_deg: 0.0
  orbit_control:
    module: "sim.control.orbit.zero_controller"
    class_name: "ZeroController"
  attitude_control:
    module: "sim.control.attitude.surrogate_snap"
    class_name: "SurrogateSnapECIController"
  mission_objectives:
    - module: "sim.mission.modules"
      class_name: "SatelliteMissionModule"
      params:
        orbital_mode: "pursuit_blind"
        attitude_mode: "hold_eci"
        max_accel_km_s2: 2.0e-6
        blind_direction_eci: [1.0, 0.0, 0.0]
simulator:
  duration_s: 5.0
  dt_s: 1.0
outputs:
  output_dir: "{outdir}"
  mode: "save"
  stats:
    print_summary: false
    save_json: false
    save_full_log: true
  plots:
    enabled: false
    figure_ids: []
  animations:
    enabled: false
    types: []
monte_carlo:
  enabled: false
                """.strip().format(outdir=str(outdir).replace("\\", "\\\\")),
                encoding="utf-8",
            )
            _ = run_master_simulation(cfg_path)
            payload = json.loads((outdir / "master_run_log.json").read_text(encoding="utf-8"))
            target_truth = np.array(payload["truth_by_object"]["target"], dtype=float)
            target_thrust = np.array(payload["applied_thrust_by_object"]["target"], dtype=float)
            self.assertTrue(np.allclose(np.nan_to_num(target_thrust, nan=0.0), 0.0, atol=1e-12))
            self.assertTrue(np.allclose(target_truth[:, 13], target_truth[0, 13], atol=1e-12))


if __name__ == "__main__":
    unittest.main()
