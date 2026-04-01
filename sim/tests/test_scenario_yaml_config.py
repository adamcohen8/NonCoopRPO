import unittest
from pathlib import Path

from sim.config import load_simulation_yaml, scenario_config_from_dict


class TestScenarioYamlConfig(unittest.TestCase):
    def test_from_dict_parses_expected_sections(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "unit_test",
                "scenario_description": "Unit test scenario",
                "rocket": {
                    "enabled": True,
                    "base_guidance": {"module": "sim.rocket.guidance", "class_name": "OpenLoopPitchProgramGuidance"},
                    "guidance_modifiers": [{"module": "sim.rocket.guidance", "class_name": "MaxQThrottleLimiterGuidance"}],
                },
                "chaser": {"enabled": False},
                "target": {"enabled": True},
                "simulator": {"duration_s": 120.0, "dt_s": 0.5},
                "outputs": {"output_dir": "outputs/test", "mode": "both", "plots": {"enabled": True}},
                "monte_carlo": {
                    "enabled": True,
                    "iterations": 10,
                    "parallel_enabled": True,
                    "parallel_workers": 3,
                    "variations": [{"parameter_path": "simulator.dt_s", "mode": "choice", "options": [0.5, 1.0]}],
                },
            }
        )
        self.assertEqual(cfg.scenario_name, "unit_test")
        self.assertEqual(cfg.scenario_description, "Unit test scenario")
        self.assertTrue(cfg.rocket.enabled)
        self.assertIsNotNone(cfg.rocket.base_guidance)
        self.assertEqual(len(cfg.rocket.guidance_modifiers), 1)
        self.assertFalse(cfg.chaser.enabled)
        self.assertTrue(cfg.monte_carlo.enabled)
        self.assertEqual(cfg.monte_carlo.iterations, 10)
        self.assertTrue(cfg.monte_carlo.parallel_enabled)
        self.assertEqual(cfg.monte_carlo.parallel_workers, 3)
        self.assertEqual(len(cfg.monte_carlo.variations), 1)
        self.assertEqual(cfg.outputs.mode, "both")
        self.assertEqual(cfg.outputs.output_dir, "outputs/test")

    def test_invalid_outputs_mode_raises(self):
        with self.assertRaises(ValueError):
            scenario_config_from_dict(
                {
                    "simulator": {"duration_s": 100.0, "dt_s": 1.0},
                    "outputs": {"mode": "bad_mode"},
                }
            )

    def test_analysis_section_normalizes_monte_carlo_and_sensitivity(self):
        mc_cfg = scenario_config_from_dict(
            {
                "scenario_name": "analysis_mc",
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                "analysis": {
                    "enabled": True,
                    "study_type": "monte_carlo",
                    "execution": {"parallel_enabled": True, "parallel_workers": 4},
                    "monte_carlo": {
                        "iterations": 12,
                        "base_seed": 9,
                        "variations": [{"parameter_path": "simulator.dt_s", "options": [0.5, 1.0]}],
                    },
                },
            }
        )
        self.assertTrue(mc_cfg.analysis.enabled)
        self.assertEqual(mc_cfg.analysis.study_type, "monte_carlo")
        self.assertTrue(mc_cfg.monte_carlo.enabled)
        self.assertEqual(mc_cfg.monte_carlo.iterations, 12)
        self.assertTrue(mc_cfg.monte_carlo.parallel_enabled)
        self.assertEqual(mc_cfg.monte_carlo.parallel_workers, 4)

        sens_cfg = scenario_config_from_dict(
            {
                "scenario_name": "analysis_sensitivity",
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                "analysis": {
                    "enabled": True,
                    "study_type": "sensitivity",
                    "execution": {"parallel_enabled": False, "parallel_workers": 0},
                    "metrics": ["summary.duration_s", "derived.closest_approach_km"],
                    "sensitivity": {
                        "method": "one_at_a_time",
                        "parameters": [{"parameter_path": "simulator.dt_s", "values": [0.5, 1.0]}],
                    },
                },
            }
        )
        self.assertTrue(sens_cfg.analysis.enabled)
        self.assertEqual(sens_cfg.analysis.study_type, "sensitivity")
        self.assertFalse(sens_cfg.monte_carlo.enabled)
        self.assertEqual(len(sens_cfg.analysis.sensitivity.parameters), 1)
        self.assertEqual(sens_cfg.analysis.metrics, ["summary.duration_s", "derived.closest_approach_km"])

        lhs_cfg = scenario_config_from_dict(
            {
                "scenario_name": "analysis_lhs",
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                "analysis": {
                    "enabled": True,
                    "study_type": "sensitivity",
                    "sensitivity": {
                        "method": "lhs",
                        "samples": 8,
                        "seed": 13,
                        "parameters": [
                            {
                                "parameter_path": "simulator.dt_s",
                                "distribution": "uniform",
                                "low": 0.5,
                                "high": 1.5,
                            }
                        ],
                    },
                },
            }
        )
        self.assertEqual(lhs_cfg.analysis.sensitivity.method, "lhs")
        self.assertEqual(lhs_cfg.analysis.sensitivity.samples, 8)
        self.assertEqual(lhs_cfg.analysis.sensitivity.seed, 13)
        self.assertEqual(lhs_cfg.analysis.sensitivity.parameters[0].distribution, "uniform")
        self.assertAlmostEqual(float(lhs_cfg.analysis.sensitivity.parameters[0].low), 0.5)

    def test_template_yaml_loads(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")
        root = Path(__file__).resolve().parents[2]
        p = root / "configs" / "simulation_template.yaml"
        cfg = load_simulation_yaml(p)
        self.assertTrue(cfg.target.enabled)
        self.assertGreater(cfg.simulator.dt_s, 0.0)
        self.assertGreater(cfg.simulator.duration_s, 0.0)

    def test_missing_agent_sections_use_role_defaults(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "defaults_test",
                "simulator": {"duration_s": 10.0, "dt_s": 1.0},
                "outputs": {"mode": "save", "output_dir": "outputs/defaults_test"},
                "monte_carlo": {"enabled": False},
            }
        )
        self.assertFalse(cfg.rocket.enabled)
        self.assertFalse(cfg.chaser.enabled)
        self.assertTrue(cfg.target.enabled)

    def test_satellite_guidance_field_is_rejected(self):
        with self.assertRaises(ValueError):
            scenario_config_from_dict(
                {
                    "scenario_name": "stale_guidance",
                    "chaser": {
                        "enabled": True,
                        "guidance": {
                            "module": "sim.control.orbit.zero_controller",
                            "class_name": "ZeroController",
                        },
                    },
                    "simulator": {"duration_s": 10.0, "dt_s": 1.0},
                }
            )

    def test_simulator_nested_defaults_are_preserved(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "sim_defaults_test",
                "simulator": {"duration_s": 10.0, "dt_s": 1.0},
                "outputs": {"mode": "save", "output_dir": "outputs/sim_defaults_test"},
                "monte_carlo": {"enabled": False},
            }
        )
        self.assertEqual(cfg.simulator.plugin_validation.get("strict"), True)
        self.assertEqual(cfg.simulator.termination.get("earth_impact_enabled"), True)
        self.assertAlmostEqual(float(cfg.simulator.termination.get("earth_radius_km")), 6378.137, places=6)


if __name__ == "__main__":
    unittest.main()
