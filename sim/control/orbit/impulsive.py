from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.models import StateTruth
from sim.utils.quaternion import dcm_to_quaternion_bn, quaternion_to_dcm_bn


@dataclass(frozen=True)
class ImpulsiveManeuver:
    desired_velocity_eci_km_s: np.ndarray
    require_attitude_alignment: bool = False
    thruster_position_body_m: np.ndarray | None = None
    thruster_direction_body: np.ndarray | None = None
    alignment_tolerance_rad: float = np.deg2rad(5.0)


@dataclass(frozen=True)
class DeltaVManeuver:
    delta_v_eci_km_s: np.ndarray
    require_attitude_alignment: bool = False
    thruster_position_body_m: np.ndarray | None = None
    thruster_direction_body: np.ndarray | None = None
    alignment_tolerance_rad: float = np.deg2rad(5.0)


@dataclass(frozen=True)
class ThrustLimitedDeltaVManeuver:
    delta_v_eci_km_s: np.ndarray
    max_thrust_n: float
    dt_s: float
    min_thrust_n: float = 0.0
    require_attitude_alignment: bool = False
    thruster_position_body_m: np.ndarray | None = None
    thruster_direction_body: np.ndarray | None = None
    alignment_tolerance_rad: float = np.deg2rad(5.0)


@dataclass(frozen=True)
class ImpulsiveManeuverResult:
    executed: bool
    required_delta_v_km_s: float
    remaining_delta_v_km_s: float
    required_attitude_quat_bn: np.ndarray | None = None
    alignment_ok: bool = True
    alignment_angle_rad: float | None = None


@dataclass(frozen=True)
class ThrustLimitedDeltaVManeuverResult:
    executed: bool
    commanded_delta_v_km_s: float
    commanded_thrust_n: float
    min_thrust_n: float
    thrust_limited_delta_v_km_s: float
    applied_delta_v_km_s: float
    remaining_delta_v_km_s: float
    required_attitude_quat_bn: np.ndarray | None = None
    alignment_ok: bool = True
    alignment_angle_rad: float | None = None


class AttitudeAgnosticImpulsiveManeuverer:
    """Instantly retarget velocity when delta-V budget is sufficient.

    This model ignores attitude and actuation dynamics by design.
    """

    def execute(
        self,
        truth: StateTruth,
        maneuver: ImpulsiveManeuver,
        available_delta_v_km_s: float,
    ) -> tuple[StateTruth, ImpulsiveManeuverResult]:
        desired_v = np.array(maneuver.desired_velocity_eci_km_s, dtype=float)
        if desired_v.shape != (3,):
            raise ValueError("desired_velocity_eci_km_s must be a length-3 vector.")
        if available_delta_v_km_s < 0.0:
            raise ValueError("available_delta_v_km_s must be non-negative.")

        dv_req = float(np.linalg.norm(desired_v - truth.velocity_eci_km_s))
        required_attitude = self._solve_required_attitude_for_delta_v(
            truth=truth,
            dv_eci_km_s=desired_v - truth.velocity_eci_km_s,
            thruster_direction_body=maneuver.thruster_direction_body,
        )
        align_ok, align_angle = self._check_attitude_alignment(
            truth=truth,
            dv_eci_km_s=desired_v - truth.velocity_eci_km_s,
            require_attitude_alignment=maneuver.require_attitude_alignment,
            thruster_position_body_m=maneuver.thruster_position_body_m,
            thruster_direction_body=maneuver.thruster_direction_body,
            alignment_tolerance_rad=maneuver.alignment_tolerance_rad,
        )
        if not align_ok:
            return truth.copy(), ImpulsiveManeuverResult(
                executed=False,
                required_delta_v_km_s=dv_req,
                remaining_delta_v_km_s=available_delta_v_km_s,
                required_attitude_quat_bn=required_attitude,
                alignment_ok=False,
                alignment_angle_rad=align_angle,
            )
        if dv_req <= available_delta_v_km_s:
            new_truth = truth.copy()
            new_truth.velocity_eci_km_s = desired_v
            rem = max(0.0, available_delta_v_km_s - dv_req)
            return new_truth, ImpulsiveManeuverResult(True, dv_req, rem, required_attitude, True, align_angle)

        return truth.copy(), ImpulsiveManeuverResult(
            False,
            dv_req,
            available_delta_v_km_s,
            required_attitude,
            True,
            align_angle,
        )

    def execute_delta_v(
        self,
        truth: StateTruth,
        maneuver: DeltaVManeuver,
        available_delta_v_km_s: float,
    ) -> tuple[StateTruth, ImpulsiveManeuverResult]:
        dv_vec = np.array(maneuver.delta_v_eci_km_s, dtype=float)
        if dv_vec.shape != (3,):
            raise ValueError("delta_v_eci_km_s must be a length-3 vector.")
        desired_v = truth.velocity_eci_km_s + dv_vec
        return self.execute(
            truth=truth,
            maneuver=ImpulsiveManeuver(
                desired_velocity_eci_km_s=desired_v,
                require_attitude_alignment=maneuver.require_attitude_alignment,
                thruster_position_body_m=maneuver.thruster_position_body_m,
                thruster_direction_body=maneuver.thruster_direction_body,
                alignment_tolerance_rad=maneuver.alignment_tolerance_rad,
            ),
            available_delta_v_km_s=available_delta_v_km_s,
        )

    def execute_delta_v_with_thrust_limit(
        self,
        truth: StateTruth,
        maneuver: ThrustLimitedDeltaVManeuver,
        available_delta_v_km_s: float,
    ) -> tuple[StateTruth, ThrustLimitedDeltaVManeuverResult]:
        dv_cmd = np.array(maneuver.delta_v_eci_km_s, dtype=float)
        if dv_cmd.shape != (3,):
            raise ValueError("delta_v_eci_km_s must be a length-3 vector.")
        if maneuver.max_thrust_n < 0.0:
            raise ValueError("max_thrust_n must be non-negative.")
        if maneuver.min_thrust_n < 0.0:
            raise ValueError("min_thrust_n must be non-negative.")
        if maneuver.min_thrust_n > maneuver.max_thrust_n:
            raise ValueError("min_thrust_n cannot exceed max_thrust_n.")
        if maneuver.dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        if available_delta_v_km_s < 0.0:
            raise ValueError("available_delta_v_km_s must be non-negative.")
        if truth.mass_kg <= 0.0:
            raise ValueError("truth.mass_kg must be positive.")

        cmd_mag = float(np.linalg.norm(dv_cmd))
        commanded_thrust_n = cmd_mag * 1e3 * truth.mass_kg / maneuver.dt_s
        required_attitude = self._solve_required_attitude_for_delta_v(
            truth=truth,
            dv_eci_km_s=dv_cmd,
            thruster_direction_body=maneuver.thruster_direction_body,
        )
        align_ok, align_angle = self._check_attitude_alignment(
            truth=truth,
            dv_eci_km_s=dv_cmd,
            require_attitude_alignment=maneuver.require_attitude_alignment,
            thruster_position_body_m=maneuver.thruster_position_body_m,
            thruster_direction_body=maneuver.thruster_direction_body,
            alignment_tolerance_rad=maneuver.alignment_tolerance_rad,
        )
        if not align_ok:
            max_dv_by_thrust_km_s = float((maneuver.max_thrust_n / truth.mass_kg) * maneuver.dt_s / 1e3)
            return truth.copy(), ThrustLimitedDeltaVManeuverResult(
                executed=False,
                commanded_delta_v_km_s=cmd_mag,
                commanded_thrust_n=commanded_thrust_n,
                min_thrust_n=maneuver.min_thrust_n,
                thrust_limited_delta_v_km_s=max_dv_by_thrust_km_s,
                applied_delta_v_km_s=0.0,
                remaining_delta_v_km_s=available_delta_v_km_s,
                required_attitude_quat_bn=required_attitude,
                alignment_ok=False,
                alignment_angle_rad=align_angle,
            )
        if cmd_mag == 0.0:
            return truth.copy(), ThrustLimitedDeltaVManeuverResult(
                executed=True,
                commanded_delta_v_km_s=0.0,
                commanded_thrust_n=0.0,
                min_thrust_n=maneuver.min_thrust_n,
                thrust_limited_delta_v_km_s=0.0,
                applied_delta_v_km_s=0.0,
                remaining_delta_v_km_s=available_delta_v_km_s,
                required_attitude_quat_bn=required_attitude,
                alignment_ok=True,
                alignment_angle_rad=align_angle,
            )
        if commanded_thrust_n < maneuver.min_thrust_n:
            max_dv_by_thrust_km_s = float((maneuver.max_thrust_n / truth.mass_kg) * maneuver.dt_s / 1e3)
            return truth.copy(), ThrustLimitedDeltaVManeuverResult(
                executed=False,
                commanded_delta_v_km_s=cmd_mag,
                commanded_thrust_n=commanded_thrust_n,
                min_thrust_n=maneuver.min_thrust_n,
                thrust_limited_delta_v_km_s=max_dv_by_thrust_km_s,
                applied_delta_v_km_s=0.0,
                remaining_delta_v_km_s=available_delta_v_km_s,
                required_attitude_quat_bn=required_attitude,
                alignment_ok=True,
                alignment_angle_rad=align_angle,
            )

        max_dv_by_thrust_km_s = float((maneuver.max_thrust_n / truth.mass_kg) * maneuver.dt_s / 1e3)
        applied_mag = min(cmd_mag, max_dv_by_thrust_km_s)

        if available_delta_v_km_s < applied_mag:
            return truth.copy(), ThrustLimitedDeltaVManeuverResult(
                executed=False,
                commanded_delta_v_km_s=cmd_mag,
                commanded_thrust_n=commanded_thrust_n,
                min_thrust_n=maneuver.min_thrust_n,
                thrust_limited_delta_v_km_s=max_dv_by_thrust_km_s,
                applied_delta_v_km_s=0.0,
                remaining_delta_v_km_s=available_delta_v_km_s,
                required_attitude_quat_bn=required_attitude,
                alignment_ok=True,
                alignment_angle_rad=align_angle,
            )

        dv_applied = dv_cmd * (applied_mag / cmd_mag)
        new_truth = truth.copy()
        new_truth.velocity_eci_km_s = truth.velocity_eci_km_s + dv_applied
        rem = max(0.0, available_delta_v_km_s - applied_mag)
        return new_truth, ThrustLimitedDeltaVManeuverResult(
            executed=True,
            commanded_delta_v_km_s=cmd_mag,
            commanded_thrust_n=commanded_thrust_n,
            min_thrust_n=maneuver.min_thrust_n,
            thrust_limited_delta_v_km_s=max_dv_by_thrust_km_s,
            applied_delta_v_km_s=applied_mag,
            remaining_delta_v_km_s=rem,
            required_attitude_quat_bn=required_attitude,
            alignment_ok=True,
            alignment_angle_rad=align_angle,
        )

    def required_attitude_for_delta_v(
        self,
        truth: StateTruth,
        delta_v_eci_km_s: np.ndarray,
        thruster_direction_body: np.ndarray,
    ) -> np.ndarray:
        dv = np.array(delta_v_eci_km_s, dtype=float)
        if dv.shape != (3,):
            raise ValueError("delta_v_eci_km_s must be a length-3 vector.")
        return self._solve_required_attitude_for_delta_v(
            truth=truth,
            dv_eci_km_s=dv,
            thruster_direction_body=thruster_direction_body,
        )

    @staticmethod
    def _check_attitude_alignment(
        truth: StateTruth,
        dv_eci_km_s: np.ndarray,
        require_attitude_alignment: bool,
        thruster_position_body_m: np.ndarray | None,
        thruster_direction_body: np.ndarray | None,
        alignment_tolerance_rad: float,
    ) -> tuple[bool, float | None]:
        if not require_attitude_alignment:
            return True, None
        if alignment_tolerance_rad < 0.0:
            raise ValueError("alignment_tolerance_rad must be non-negative.")
        if thruster_position_body_m is None:
            raise ValueError("thruster_position_body_m must be provided when require_attitude_alignment=True.")
        if thruster_direction_body is None:
            raise ValueError("thruster_direction_body must be provided when require_attitude_alignment=True.")
        pos = np.array(thruster_position_body_m, dtype=float)
        direction_body = np.array(thruster_direction_body, dtype=float)
        if pos.shape != (3,):
            raise ValueError("thruster_position_body_m must be length-3.")
        if direction_body.shape != (3,):
            raise ValueError("thruster_direction_body must be length-3.")
        dv_norm = float(np.linalg.norm(dv_eci_km_s))
        dir_norm = float(np.linalg.norm(direction_body))
        if dv_norm == 0.0:
            return True, 0.0
        if dir_norm == 0.0:
            raise ValueError("thruster_direction_body cannot be zero.")
        thrust_axis_body = direction_body / dir_norm
        c_bn = quaternion_to_dcm_bn(truth.attitude_quat_bn)
        thrust_axis_eci = c_bn.T @ thrust_axis_body
        target_axis_eci = -dv_eci_km_s / dv_norm
        cosang = float(np.clip(np.dot(thrust_axis_eci, target_axis_eci), -1.0, 1.0))
        angle = float(np.arccos(cosang))
        return angle <= alignment_tolerance_rad, angle

    @staticmethod
    def _solve_required_attitude_for_delta_v(
        truth: StateTruth,
        dv_eci_km_s: np.ndarray,
        thruster_direction_body: np.ndarray | None,
    ) -> np.ndarray | None:
        if thruster_direction_body is None:
            return None

        dv = np.array(dv_eci_km_s, dtype=float)
        if dv.shape != (3,):
            raise ValueError("dv_eci_km_s must be a length-3 vector.")
        dv_norm = float(np.linalg.norm(dv))
        if dv_norm == 0.0:
            return truth.attitude_quat_bn.copy()

        t_body = np.array(thruster_direction_body, dtype=float)
        if t_body.shape != (3,):
            raise ValueError("thruster_direction_body must be length-3.")
        t_body_norm = float(np.linalg.norm(t_body))
        if t_body_norm == 0.0:
            raise ValueError("thruster_direction_body cannot be zero.")
        t_body = t_body / t_body_norm

        basis = np.eye(3)
        ref_idx = int(np.argmin(np.abs(t_body)))
        ref_body = basis[:, ref_idx]
        s_body = ref_body - np.dot(ref_body, t_body) * t_body
        s_body_norm = float(np.linalg.norm(s_body))
        if s_body_norm == 0.0:
            ref_body = basis[:, (ref_idx + 1) % 3]
            s_body = ref_body - np.dot(ref_body, t_body) * t_body
            s_body_norm = float(np.linalg.norm(s_body))
            if s_body_norm == 0.0:
                raise ValueError("Could not construct an orthogonal body basis from thruster_direction_body.")
        s_body = s_body / s_body_norm
        u_body = np.cross(t_body, s_body)
        u_body = u_body / max(float(np.linalg.norm(u_body)), 1e-12)
        b_mat = np.column_stack((t_body, s_body, u_body))

        c_bn_cur = quaternion_to_dcm_bn(truth.attitude_quat_bn)
        c_nb_cur = c_bn_cur.T
        s_cur_eci = c_nb_cur @ s_body

        t_eci = -dv / dv_norm
        s_eci = s_cur_eci - np.dot(s_cur_eci, t_eci) * t_eci
        s_eci_norm = float(np.linalg.norm(s_eci))
        if s_eci_norm == 0.0:
            ref_idx_eci = int(np.argmin(np.abs(t_eci)))
            ref_eci = basis[:, ref_idx_eci]
            s_eci = ref_eci - np.dot(ref_eci, t_eci) * t_eci
            s_eci_norm = float(np.linalg.norm(s_eci))
            if s_eci_norm == 0.0:
                raise ValueError("Could not construct an orthogonal inertial basis from desired thrust axis.")
        s_eci = s_eci / s_eci_norm
        u_eci = np.cross(t_eci, s_eci)
        u_eci = u_eci / max(float(np.linalg.norm(u_eci)), 1e-12)
        s_eci = np.cross(u_eci, t_eci)
        s_eci = s_eci / max(float(np.linalg.norm(s_eci)), 1e-12)
        i_mat = np.column_stack((t_eci, s_eci, u_eci))

        c_nb_des = i_mat @ b_mat.T
        c_bn_des = c_nb_des.T
        return dcm_to_quaternion_bn(c_bn_des)
