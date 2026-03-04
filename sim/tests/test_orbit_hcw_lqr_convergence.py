import unittest

import numpy as np

from sim.control.orbit.lqr import HCWLQRController
from sim.core.models import StateBelief
from sim.utils.frames import ric_rect_to_curv


def _hcw_derivative(x_rect: np.ndarray, n_rad_s: float, accel_ric_km_s2: np.ndarray) -> np.ndarray:
    x_r, x_i, x_c, x_rdot, x_idot, x_cdot = x_rect
    ux, uy, uz = accel_ric_km_s2
    return np.array(
        [
            x_rdot,
            x_idot,
            x_cdot,
            3.0 * n_rad_s * n_rad_s * x_r + 2.0 * n_rad_s * x_idot + ux,
            -2.0 * n_rad_s * x_rdot + uy,
            -(n_rad_s * n_rad_s) * x_c + uz,
        ],
        dtype=float,
    )


def _rk4_step(x_rect: np.ndarray, dt_s: float, n_rad_s: float, accel_ric_km_s2: np.ndarray) -> np.ndarray:
    k1 = _hcw_derivative(x_rect, n_rad_s, accel_ric_km_s2)
    k2 = _hcw_derivative(x_rect + 0.5 * dt_s * k1, n_rad_s, accel_ric_km_s2)
    k3 = _hcw_derivative(x_rect + 0.5 * dt_s * k2, n_rad_s, accel_ric_km_s2)
    k4 = _hcw_derivative(x_rect + dt_s * k3, n_rad_s, accel_ric_km_s2)
    return x_rect + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class TestHCWLQRConvergence(unittest.TestCase):
    def test_converges_for_starts_within_10km_envelope(self):
        n_rad_s = 0.0011
        dt_s = 10.0
        r0_km = 7000.0
        chief_eci = np.array([r0_km, 0.0, 0.0, 0.0, 7.5, 0.0])

        ctrl = HCWLQRController(
            mean_motion_rad_s=n_rad_s,
            max_accel_km_s2=5e-5,
            design_dt_s=dt_s,
            q_weights=np.array([8.66, 8.66, 8.66, 1.33, 1.33, 1.33]) * 1e3,
            r_weights=np.ones(3) * 1.94e13,
        )

        rng = np.random.default_rng(42)
        n_cases = 60
        success = 0
        for _ in range(n_cases):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            radius = (rng.random() ** (1.0 / 3.0)) * 10.0
            rel_pos = radius * direction
            rel_vel = rng.uniform(-0.005, 0.005, size=3)
            x_rect = np.hstack((rel_pos, rel_vel))

            for k in range(360):
                x_curv = ric_rect_to_curv(x_rect, r0_km=r0_km)
                belief = StateBelief(
                    state=np.hstack((x_curv, chief_eci)),
                    covariance=np.eye(12),
                    last_update_t_s=k * dt_s,
                )
                cmd = ctrl.act(belief, t_s=k * dt_s, budget_ms=1.0)
                a_ric = np.array(cmd.mode_flags["accel_ric_km_s2"], dtype=float)
                x_rect = _rk4_step(x_rect, dt_s, n_rad_s, a_ric)

            if np.linalg.norm(x_rect[:3]) < 0.2 and np.linalg.norm(x_rect[3:]) < 2e-4:
                success += 1

        self.assertGreaterEqual(success / n_cases, 0.95)


if __name__ == "__main__":
    unittest.main()
