import unittest

import numpy as np

from sim.core.models import StateBelief, StateTruth
from sim.mission.modules import DefensiveMissionStrategy, DefensiveRICAxisBurnMissionModule


def _truth() -> StateTruth:
    return StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=100.0,
        t_s=0.0,
    )


class DefensiveRICAxisMissionTests(unittest.TestCase):
    def test_strategy_no_knowledge_no_burn(self) -> None:
        m = DefensiveMissionStrategy(chaser_id="chaser", axis_mode="+R", burn_accel_km_s2=2e-6)
        out = m.update(truth=_truth(), own_knowledge={}, world_truth={})
        self.assertTrue(np.allclose(np.array(out["fallback_thrust_eci_km_s2"], dtype=float), np.zeros(3), atol=1e-15))
        self.assertFalse(bool(out["mission_mode"]["triggered"]))

    def test_strategy_burn_triggers_with_knowledge(self) -> None:
        m = DefensiveMissionStrategy(chaser_id="chaser", axis_mode="+I", burn_accel_km_s2=3e-6)
        kb = StateBelief(state=np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0]), covariance=np.eye(6), last_update_t_s=0.0)
        out = m.update(truth=_truth(), own_knowledge={"chaser": kb}, world_truth={})
        thrust = np.array(out["fallback_thrust_eci_km_s2"], dtype=float)
        self.assertAlmostEqual(float(np.linalg.norm(thrust)), 3e-6, places=12)
        self.assertGreater(float(thrust[1]), 0.0)
        self.assertTrue(bool(out["mission_mode"]["triggered"]))

    def test_strategy_away_from_chaser_mode_points_radially_away(self) -> None:
        m = DefensiveMissionStrategy(chaser_id="chaser", defense_mode="away_from_chaser", burn_accel_km_s2=2e-6)
        kb = StateBelief(state=np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0]), covariance=np.eye(6), last_update_t_s=0.0)
        out = m.update(truth=_truth(), own_knowledge={"chaser": kb}, world_truth={})
        thrust = np.array(out["fallback_thrust_eci_km_s2"], dtype=float)
        self.assertLess(float(thrust[0]), 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(thrust)), 2e-6, places=12)

    def test_no_knowledge_no_burn(self) -> None:
        m = DefensiveRICAxisBurnMissionModule(chaser_id="chaser", axis_mode="+R", burn_accel_km_s2=2e-6)
        out = m.update(truth=_truth(), own_knowledge={}, t_s=0.0, dt_s=1.0)
        self.assertTrue(np.allclose(np.array(out["thrust_eci_km_s2"], dtype=float), np.zeros(3), atol=1e-15))
        self.assertFalse(bool(out["mission_mode"]["triggered"]))

    def test_burn_triggers_with_knowledge(self) -> None:
        m = DefensiveRICAxisBurnMissionModule(chaser_id="chaser", axis_mode="+I", burn_accel_km_s2=3e-6)
        kb = StateBelief(state=np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0]), covariance=np.eye(6), last_update_t_s=0.0)
        out = m.update(truth=_truth(), own_knowledge={"chaser": kb}, t_s=0.0, dt_s=1.0)
        thrust = np.array(out["thrust_eci_km_s2"], dtype=float)
        self.assertAlmostEqual(float(np.linalg.norm(thrust)), 3e-6, places=12)
        self.assertGreater(float(thrust[1]), 0.0)
        self.assertTrue(bool(out["mission_mode"]["triggered"]))

    def test_negative_cross_track_direction(self) -> None:
        m = DefensiveRICAxisBurnMissionModule(chaser_id="chaser", axis_mode="-C", burn_accel_km_s2=1e-6)
        kb = StateBelief(state=np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0]), covariance=np.eye(6), last_update_t_s=0.0)
        out = m.update(truth=_truth(), own_knowledge={"chaser": kb}, t_s=0.0, dt_s=1.0)
        thrust = np.array(out["thrust_eci_km_s2"], dtype=float)
        self.assertLess(float(thrust[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
