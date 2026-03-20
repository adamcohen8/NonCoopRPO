import unittest

import numpy as np

from sim.core.models import Command, StateBelief, StateTruth
from sim.mission.modules import BudgetedEndStateExecution, DesiredStateMissionStrategy, IntegratedCommandExecution, PredictiveBurnExecution


def _truth() -> StateTruth:
    return StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0], dtype=float),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0], dtype=float),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        angular_rate_body_rad_s=np.zeros(3, dtype=float),
        mass_kg=100.0,
        t_s=0.0,
    )


class _ConstantOrbitController:
    def __init__(self, thrust_eci_km_s2: list[float]):
        self.thrust = np.array(thrust_eci_km_s2, dtype=float)
        self.target_state = None

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        return Command(thrust_eci_km_s2=self.thrust.copy(), torque_body_nm=np.zeros(3), mode_flags={"mode": "constant"})


class _ZeroAttitudeController:
    def __init__(self):
        self.target = None

    def set_target(self, q):
        self.target = np.array(q, dtype=float)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        return Command.zero()


class MissionArchitectureMigrationTests(unittest.TestCase):
    def test_desired_state_strategy_emits_explicit_state_intent(self) -> None:
        s = DesiredStateMissionStrategy(
            desired_state_source="explicit",
            desired_position_eci_km=np.array([7100.0, 1.0, 2.0]),
            desired_velocity_eci_km_s=np.array([0.1, 7.4, 0.0]),
        )
        out = s.update(own_knowledge={}, world_truth={})
        self.assertTrue(np.allclose(np.array(out["desired_state_eci_6"], dtype=float), np.array([7100.0, 1.0, 2.0, 0.1, 7.4, 0.0])))

    def test_integrated_command_execution_consumes_desired_state_intent(self) -> None:
        e = IntegratedCommandExecution(alignment_tolerance_deg=180.0)
        controller = _ConstantOrbitController([1.0e-5, 0.0, 0.0])
        att = _ZeroAttitudeController()
        truth = _truth()
        orb_belief = StateBelief(state=np.hstack((truth.position_eci_km, truth.velocity_eci_km_s)), covariance=np.eye(6), last_update_t_s=0.0)
        att_belief = StateBelief(state=np.hstack((truth.attitude_quat_bn, truth.angular_rate_body_rad_s)), covariance=np.eye(7), last_update_t_s=0.0)
        out = e.update(
            intent={"desired_state_eci_6": np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0])},
            truth=truth,
            orbit_controller=controller,
            attitude_controller=att,
            orb_belief=orb_belief,
            att_belief=att_belief,
            t_s=0.0,
            env={"attitude_disabled": False},
        )
        self.assertTrue(np.allclose(np.array(controller.target_state, dtype=float), np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0])))
        self.assertGreater(float(np.linalg.norm(np.array(out["thrust_eci_km_s2"], dtype=float))), 0.0)

    def test_budgeted_end_state_execution_reduces_delta_v_budget(self) -> None:
        e = BudgetedEndStateExecution(
            strategy="thrust_limited",
            max_thrust_n=10.0,
            min_thrust_n=0.0,
            burn_dt_s=1.0,
            available_delta_v_km_s=0.1,
            require_attitude_alignment=False,
            alignment_tolerance_deg=180.0,
        )
        truth = _truth()
        out = e.update(
            intent={"desired_state_eci_6": np.array([7000.0, 0.0, 0.0, 0.01, 7.5, 0.0])},
            truth=truth,
            env={"attitude_disabled": True},
        )
        self.assertIn("remaining_delta_v_km_s", out["mission_mode"])
        self.assertLess(float(out["mission_mode"]["remaining_delta_v_km_s"]), 0.1)

    def test_predictive_burn_execution_accepts_desired_state_intent(self) -> None:
        e = PredictiveBurnExecution(lead_time_s=0.0, alignment_tolerance_deg=180.0)
        controller = _ConstantOrbitController([2.0e-5, 0.0, 0.0])
        att = _ZeroAttitudeController()
        truth = _truth()
        out = e.update(
            intent={"desired_state_eci_6": np.array([7100.0, 0.0, 0.0, 0.0, 7.4, 0.0])},
            truth=truth,
            own_knowledge={},
            world_truth={},
            orbit_controller=controller,
            attitude_controller=att,
            att_belief=None,
            t_s=0.0,
            dt_s=1.0,
            env={"attitude_disabled": True},
        )
        self.assertGreater(float(np.linalg.norm(np.array(out["thrust_eci_km_s2"], dtype=float))), 0.0)


if __name__ == "__main__":
    unittest.main()
