from __future__ import annotations

import unittest

from sim.config import scenario_config_from_dict
from sim.master_simulator import _create_rocket_runtime
from sim.rocket.guidance import MaxQThrottleLimiterGuidance, OpenLoopPitchProgramGuidance, OrbitInsertionCutoffGuidance


class TestRocketGuidanceStack(unittest.TestCase):
    def test_runtime_builds_guidance_stack_from_base_and_modifiers(self):
        cfg = scenario_config_from_dict(
            {
                "rocket": {
                    "enabled": True,
                    "base_guidance": {
                        "module": "sim.rocket.guidance",
                        "class_name": "OpenLoopPitchProgramGuidance",
                        "params": {},
                    },
                    "guidance_modifiers": [
                        {
                            "module": "sim.rocket.guidance",
                            "class_name": "OrbitInsertionCutoffGuidance",
                            "params": {},
                        },
                        {
                            "module": "sim.rocket.guidance",
                            "class_name": "MaxQThrottleLimiterGuidance",
                            "params": {},
                        },
                    ],
                },
                "target": {"enabled": False},
                "chaser": {"enabled": False},
                "simulator": {"duration_s": 10.0, "dt_s": 1.0},
            }
        )
        runtime = _create_rocket_runtime(cfg)
        self.assertIsInstance(runtime.rocket_guidance, MaxQThrottleLimiterGuidance)
        self.assertIsInstance(runtime.rocket_guidance.base_guidance, OrbitInsertionCutoffGuidance)
        self.assertIsInstance(runtime.rocket_guidance.base_guidance.base_guidance, OpenLoopPitchProgramGuidance)


if __name__ == "__main__":
    unittest.main()
