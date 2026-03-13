from __future__ import annotations

import unittest

import numpy as np

from sim.control.attitude.surrogate_snap import SurrogateSnapECIController
from sim.core.models import Command, StateBelief, StateTruth
from sim.dynamics.model import OrbitalAttitudeDynamics


def _quat_angle_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    a = np.array(q1, dtype=float).reshape(4)
    b = np.array(q2, dtype=float).reshape(4)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    d = float(abs(np.dot(a, b)))
    d = float(np.clip(d, -1.0, 1.0))
    return float(2.0 * np.arccos(d))


class TestSurrogateSnapController(unittest.TestCase):
    def test_eci_phase_progression_rate_cancel_then_slew(self):
        ctrl = SurrogateSnapECIController(
            desired_attitude_quat_bn=np.array([0.0, 1.0, 0.0, 0.0]),  # 180 deg about x
            cancel_rate_mag_rad_s2=1.0,
            rate_tolerance_rad_s=1e-3,
            slew_time_180_s=2.0,
            pointing_sigma_deg=0.0,
            default_dt_s=1.0,
        )
        b = StateBelief(
            state=np.hstack((np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))),
            covariance=np.eye(13),
            last_update_t_s=0.0,
        )

        c1 = ctrl.act(belief=b, t_s=1.0, budget_ms=1.0)
        o1 = dict(c1.mode_flags["attitude_state_override"])
        self.assertEqual(o1.get("phase"), "rate_cancel")
        self.assertAlmostEqual(float(np.linalg.norm(np.array(o1["w_next_body_rad_s"], dtype=float))), 1.0, places=6)

        # Next step: rates should reach zero and controller should begin slewing.
        b2 = StateBelief(
            state=np.hstack((b.state[:10], np.array(o1["w_next_body_rad_s"], dtype=float))),
            covariance=np.eye(13),
            last_update_t_s=1.0,
        )
        c2 = ctrl.act(belief=b2, t_s=2.0, budget_ms=1.0)
        o2 = dict(c2.mode_flags["attitude_state_override"])
        self.assertEqual(o2.get("phase"), "slew")
        q2 = np.array(o2["q_next_bn"], dtype=float)
        # With 2 s for 180 deg and dt=1s, first slew step should be ~90 deg from initial.
        self.assertAlmostEqual(_quat_angle_rad(np.array([1.0, 0.0, 0.0, 0.0]), q2), np.pi / 2.0, places=4)

    def test_dynamics_applies_attitude_override(self):
        dyn = OrbitalAttitudeDynamics(mu_km3_s2=398600.4418, inertia_kg_m2=np.diag([100.0, 90.0, 80.0]))
        s = StateTruth(
            position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
            velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_rate_body_rad_s=np.array([0.1, -0.2, 0.3], dtype=float),
            mass_kg=300.0,
            t_s=0.0,
        )
        q_cmd = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
        w_cmd = np.array([0.0, 0.0, 0.0], dtype=float)
        cmd = Command(
            mode_flags={
                "attitude_state_override": {
                    "q_next_bn": q_cmd.tolist(),
                    "w_next_body_rad_s": w_cmd.tolist(),
                }
            }
        )
        s_next = dyn.step(state=s, command=cmd, env={}, dt_s=1.0)
        self.assertTrue(np.allclose(s_next.attitude_quat_bn, q_cmd, atol=1e-9))
        self.assertTrue(np.allclose(s_next.angular_rate_body_rad_s, w_cmd, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
