import unittest
from pathlib import Path

from sim.config import load_simulation_yaml, scenario_config_from_dict


class TestScenarioYamlConfig(unittest.TestCase):
    def test_from_dict_parses_expected_sections(self):
        cfg = scenario_config_from_dict(
            {
                "scenario_name": "unit_test",
                "rocket": {"enabled": True, "guidance": {"module": "sim.rocket.guidance", "class_name": "OpenLoopPitchProgramGuidance"}},
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
        self.assertTrue(cfg.rocket.enabled)
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
