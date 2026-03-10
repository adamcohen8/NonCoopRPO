from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.control.orbit.impulsive import AttitudeAgnosticImpulsiveManeuverer
from sim.control.orbit.lqr import HCWLQRController
from sim.core.models import StateBelief, StateTruth
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.utils.frames import dcm_to_euler_321, ric_rect_to_curv, ric_dcm_ir_from_rv
from sim.utils.quaternion import quaternion_to_dcm_bn


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _eci_to_ric_rect(x_host_eci: np.ndarray, x_dep_eci: np.ndarray) -> np.ndarray:
    r_host = x_host_eci[:3]
    v_host = x_host_eci[3:]
    r_dep = x_dep_eci[:3]
    v_dep = x_dep_eci[3:]
    dr = r_dep - r_host
    dv = v_dep - v_host

    h = np.cross(r_host, v_host)
    in_vec = np.cross(h, r_host)
    rsw = np.column_stack((_unit(r_host), _unit(in_vec), _unit(h)))

    rtemp = np.cross(h, v_host)
    vtemp = np.cross(h, r_host)
    drsw = np.column_stack((v_host / max(np.linalg.norm(r_host), 1e-12), rtemp / max(np.linalg.norm(vtemp), 1e-12), np.zeros(3)))

    x_r = rsw.T @ dr
    frame_mv = np.array(
        [
            x_r[0] * (r_host @ v_host) / (max(np.linalg.norm(r_host), 1e-12) ** 2),
            x_r[1] * (vtemp @ rtemp) / (max(np.linalg.norm(vtemp), 1e-12) ** 2),
            0.0,
        ]
    )
    x_v = (rsw.T @ dv) + (drsw.T @ dr) - frame_mv
    return np.hstack((x_r, x_v))


@dataclass(frozen=True)
class PredictiveBurnConfig:
    horizon_steps: int = 20
    attitude_tolerance_rad: float = np.deg2rad(5.0)
    mu_km3_s2: float = 398600.4418


@dataclass
class PredictiveBurnScheduler:
    orbit_lqr: HCWLQRController
    thruster_direction_body: np.ndarray
    config: PredictiveBurnConfig = PredictiveBurnConfig()
    _maneuverer: AttitudeAgnosticImpulsiveManeuverer = field(default_factory=AttitudeAgnosticImpulsiveManeuverer)
    _countdown: int = -1
    _planned_accel_eci_km_s2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _planned_q_target_bn: np.ndarray | None = None

    def step(
        self,
        chaser_truth: StateTruth,
        chief_truth: StateTruth,
        chaser_orbit_belief: StateBelief,
        chief_orbit_belief: StateBelief,
        dt_s: float,
    ) -> dict:
        if self._countdown < 0:
            self._plan(chaser_truth, chaser_orbit_belief, chief_orbit_belief, dt_s)

        if self._planned_q_target_bn is None:
            self._planned_q_target_bn = chaser_truth.attitude_quat_bn.copy()

        desired_ric_euler = self._target_quat_to_ric_euler(
            q_target_bn=self._planned_q_target_bn,
            r_eci_km=chaser_truth.position_eci_km,
            v_eci_km_s=chaser_truth.velocity_eci_km_s,
        )

        fire = False
        align_angle_rad = 0.0
        accel_cmd = np.zeros(3)
        if self._countdown == 0:
            align_angle_rad = self._alignment_angle(
                truth=chaser_truth,
                accel_eci_km_s2=self._planned_accel_eci_km_s2,
                thruster_direction_body=self.thruster_direction_body,
            )
            if np.linalg.norm(self._planned_accel_eci_km_s2) > 0.0 and align_angle_rad <= self.config.attitude_tolerance_rad:
                fire = True
                accel_cmd = self._planned_accel_eci_km_s2.copy()
            self._countdown = -1
        else:
            self._countdown -= 1

        return {
            "fire": fire,
            "thrust_eci_km_s2": accel_cmd,
            "desired_ric_euler_rad": desired_ric_euler,
            "countdown": self._countdown,
            "alignment_angle_rad": align_angle_rad,
            "planned_accel_eci_km_s2": self._planned_accel_eci_km_s2.copy(),
            "planned_q_target_bn": self._planned_q_target_bn.copy(),
        }

    def _plan(self, chaser_truth: StateTruth, chaser_orbit_belief: StateBelief, chief_orbit_belief: StateBelief, dt_s: float) -> None:
        x_chief_pred = chief_orbit_belief.state.copy()
        x_chaser_pred = chaser_orbit_belief.state.copy()
        for _ in range(self.config.horizon_steps):
            x_chief_pred = propagate_two_body_rk4(
                x_eci=x_chief_pred,
                dt_s=dt_s,
                mu_km3_s2=self.config.mu_km3_s2,
                accel_cmd_eci_km_s2=np.zeros(3),
            )
            x_chaser_pred = propagate_two_body_rk4(
                x_eci=x_chaser_pred,
                dt_s=dt_s,
                mu_km3_s2=self.config.mu_km3_s2,
                accel_cmd_eci_km_s2=np.zeros(3),
            )

        x_rel_rect_pred = _eci_to_ric_rect(x_chief_pred, x_chaser_pred)
        r0 = float(np.linalg.norm(x_chief_pred[:3]))
        x_rel_curv_pred = ric_rect_to_curv(x_rel_rect_pred, r0_km=r0)
        belief_for_lqr = StateBelief(
            state=np.hstack((x_rel_curv_pred, x_chief_pred)),
            covariance=np.eye(12),
            last_update_t_s=0.0,
        )
        c_orb = self.orbit_lqr.act(belief_for_lqr, t_s=0.0, budget_ms=1.0)
        self._planned_accel_eci_km_s2 = np.array(c_orb.thrust_eci_km_s2, dtype=float)

        dv = self._planned_accel_eci_km_s2 * dt_s
        if np.linalg.norm(dv) > 0.0:
            self._planned_q_target_bn = self._maneuverer.required_attitude_for_delta_v(
                truth=chaser_truth,
                delta_v_eci_km_s=dv,
                thruster_direction_body=self.thruster_direction_body,
            )
        else:
            self._planned_q_target_bn = chaser_truth.attitude_quat_bn.copy()
        self._countdown = self.config.horizon_steps

    @staticmethod
    def _target_quat_to_ric_euler(q_target_bn: np.ndarray, r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> np.ndarray:
        c_bn_des = quaternion_to_dcm_bn(q_target_bn)
        c_ir = ric_dcm_ir_from_rv(r_eci_km, v_eci_km_s)
        c_br = c_bn_des @ c_ir
        return dcm_to_euler_321(c_br)

    @staticmethod
    def _alignment_angle(truth: StateTruth, accel_eci_km_s2: np.ndarray, thruster_direction_body: np.ndarray) -> float:
        a = np.array(accel_eci_km_s2, dtype=float)
        if np.linalg.norm(a) == 0.0:
            return 0.0
        c_bn = quaternion_to_dcm_bn(truth.attitude_quat_bn)
        thrust_axis_eci = c_bn.T @ _unit(np.array(thruster_direction_body, dtype=float))
        target_axis_eci = -_unit(a)
        cosang = float(np.clip(np.dot(thrust_axis_eci, target_axis_eci), -1.0, 1.0))
        return float(np.arccos(cosang))
