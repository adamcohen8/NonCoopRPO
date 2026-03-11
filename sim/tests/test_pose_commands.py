import unittest

import numpy as np

from sim.control.attitude import PoseCommandGenerator
from sim.core.models import StateTruth
from sim.dynamics.orbit.environment import EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import ecef_to_eci
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import quaternion_to_dcm_bn


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


class TestPoseCommands(unittest.TestCase):
    def setUp(self):
        self.truth = StateTruth(
            position_eci_km=np.array([7000.0, 100.0, 50.0], dtype=float),
            velocity_eci_km_s=np.array([0.0, 7.4, 1.0], dtype=float),
            attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=200.0,
            t_s=123.0,
        )

    def test_sun_track_aligns_panel_normal(self):
        sun = _unit(np.array([1.0, -0.2, 0.5], dtype=float))
        q = PoseCommandGenerator.sun_track(
            truth=self.truth,
            sun_dir_eci=sun,
            panel_normal_body=np.array([0.0, 0.0, 1.0], dtype=float),
        )
        c_bn = quaternion_to_dcm_bn(q)
        panel_eci = c_bn.T @ np.array([0.0, 0.0, 1.0], dtype=float)
        self.assertTrue(np.allclose(_unit(panel_eci), sun, atol=1e-8))

    def test_spotlight_ric_direction_aligns_boresight(self):
        d_ric = _unit(np.array([0.2, 0.9, -0.1], dtype=float))
        q = PoseCommandGenerator.spotlight_ric_direction(
            truth=self.truth,
            ric_direction=d_ric,
            boresight_body=np.array([1.0, 0.0, 0.0], dtype=float),
        )
        c_ir = ric_dcm_ir_from_rv(self.truth.position_eci_km, self.truth.velocity_eci_km_s)
        d_eci = _unit(c_ir @ d_ric)
        c_bn = quaternion_to_dcm_bn(q)
        bore_eci = _unit(c_bn.T @ np.array([1.0, 0.0, 0.0], dtype=float))
        self.assertTrue(np.allclose(bore_eci, d_eci, atol=1e-8))

    def test_spotlight_latlon_aligns_boresight(self):
        lat_deg = 15.0
        lon_deg = -30.0
        q = PoseCommandGenerator.spotlight_latlon(
            truth=self.truth,
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            altitude_km=0.0,
            boresight_body=np.array([1.0, 0.0, 0.0], dtype=float),
        )
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)
        r_ecef = np.array(
            [
                EARTH_RADIUS_KM * np.cos(lat) * np.cos(lon),
                EARTH_RADIUS_KM * np.cos(lat) * np.sin(lon),
                EARTH_RADIUS_KM * np.sin(lat),
            ],
            dtype=float,
        )
        target_eci = ecef_to_eci(r_ecef, self.truth.t_s)
        los_eci = _unit(target_eci - self.truth.position_eci_km)
        c_bn = quaternion_to_dcm_bn(q)
        bore_eci = _unit(c_bn.T @ np.array([1.0, 0.0, 0.0], dtype=float))
        self.assertTrue(np.allclose(bore_eci, los_eci, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
