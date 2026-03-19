import unittest

from sim.config import scenario_config_from_dict, validate_scenario_plugins


class TestPluginValidation(unittest.TestCase):
    def test_valid_plugins_pass(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {
                    "enabled": True,
                    "guidance": {
                        "module": "sim.rocket.guidance",
                        "class_name": "OpenLoopPitchProgramGuidance",
                        "params": {},
                    },
                    "orbit_control": {"module": "sim.control.orbit.zero_controller", "class_name": "ZeroController", "params": {}},
                    "attitude_control": {
                        "module": "sim.control.attitude.zero_torque",
                        "class_name": "ZeroTorqueController",
                        "params": {},
                    },
                },
                "chaser": {
                    "enabled": True,
                    "orbit_control": {"module": "sim.control.orbit.zero_controller", "class_name": "ZeroController", "params": {}},
                    "attitude_control": {
                        "module": "sim.control.attitude.zero_torque",
                        "class_name": "ZeroTorqueController",
                        "params": {},
                    },
                },
                "target": {"enabled": False},
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )
        errs = validate_scenario_plugins(cfg)
        self.assertEqual(errs, [])

    def test_satellite_guidance_is_rejected_at_parse_time(self):
        with self.assertRaises(ValueError):
            scenario_config_from_dict(
                {
                    "rocket": {"enabled": False},
                    "chaser": {
                        "enabled": True,
                        "guidance": {
                            "module": "sim.control.orbit.zero_controller",
                            "class_name": "ZeroController",
                            "params": {},
                        },
                    },
                    "target": {"enabled": False},
                    "simulator": {"duration_s": 20.0, "dt_s": 1.0},
                }
            )

    def test_invalid_plugins_fail(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {
                    "enabled": True,
                    "guidance": {"module": "sim.control.orbit.zero_controller", "class_name": "ZeroController", "params": {}},
                },
                "chaser": {"enabled": False},
                "target": {"enabled": False},
                "simulator": {"duration_s": 20.0, "dt_s": 1.0},
            }
        )
        errs = validate_scenario_plugins(cfg)
        self.assertTrue(any("rocket.guidance" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
