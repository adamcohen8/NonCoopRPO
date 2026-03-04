from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from sim.control.orbit.impulsive import (
    AttitudeAgnosticImpulsiveManeuverer,
    DeltaVManeuver,
    ThrustLimitedDeltaVManeuver,
    ThrustLimitedDeltaVManeuverResult,
)
from sim.core.models import StateTruth

IntegrationAction = Literal["fire", "slew", "hold"]
ManeuverStrategy = Literal["thrust_limited", "impulsive"]


@dataclass(frozen=True)
class IntegratedManeuverCommand:
    delta_v_eci_km_s: np.ndarray
    available_delta_v_km_s: float
    strategy: ManeuverStrategy = "thrust_limited"
    max_thrust_n: float = 0.0
    dt_s: float = 1.0
    min_thrust_n: float = 0.0
    require_attitude_alignment: bool = True
    thruster_position_body_m: np.ndarray | None = None
    thruster_direction_body: np.ndarray | None = None
    alignment_tolerance_rad: float = np.deg2rad(5.0)


@dataclass(frozen=True)
class IntegratedManeuverDecision:
    action: IntegrationAction
    executed: bool
    reason: str
    should_slew: bool
    required_attitude_quat_bn: np.ndarray | None
    applied_delta_v_km_s: float
    remaining_delta_v_km_s: float
    alignment_ok: bool
    below_min_thrust: bool
    insufficient_delta_v: bool
    thrust_result: ThrustLimitedDeltaVManeuverResult


@dataclass
class OrbitalAttitudeManeuverCoordinator:
    maneuverer: AttitudeAgnosticImpulsiveManeuverer = field(default_factory=AttitudeAgnosticImpulsiveManeuverer)

    def execute(self, truth: StateTruth, command: IntegratedManeuverCommand) -> tuple[StateTruth, IntegratedManeuverDecision]:
        dv = np.array(command.delta_v_eci_km_s, dtype=float)
        if dv.shape != (3,):
            raise ValueError("delta_v_eci_km_s must be a length-3 vector.")
        if command.available_delta_v_km_s < 0.0:
            raise ValueError("available_delta_v_km_s must be non-negative.")

        if command.strategy == "thrust_limited":
            man = ThrustLimitedDeltaVManeuver(
                delta_v_eci_km_s=dv,
                max_thrust_n=command.max_thrust_n,
                dt_s=command.dt_s,
                min_thrust_n=command.min_thrust_n,
                require_attitude_alignment=command.require_attitude_alignment,
                thruster_position_body_m=command.thruster_position_body_m,
                thruster_direction_body=command.thruster_direction_body,
                alignment_tolerance_rad=command.alignment_tolerance_rad,
            )
            next_truth, result = self.maneuverer.execute_delta_v_with_thrust_limit(
                truth=truth,
                maneuver=man,
                available_delta_v_km_s=command.available_delta_v_km_s,
            )
        elif command.strategy == "impulsive":
            next_truth, imp_result = self.maneuverer.execute_delta_v(
                truth=truth,
                maneuver=DeltaVManeuver(
                    delta_v_eci_km_s=dv,
                    require_attitude_alignment=command.require_attitude_alignment,
                    thruster_position_body_m=command.thruster_position_body_m,
                    thruster_direction_body=command.thruster_direction_body,
                    alignment_tolerance_rad=command.alignment_tolerance_rad,
                ),
                available_delta_v_km_s=command.available_delta_v_km_s,
            )
            result = ThrustLimitedDeltaVManeuverResult(
                executed=imp_result.executed,
                commanded_delta_v_km_s=imp_result.required_delta_v_km_s,
                commanded_thrust_n=0.0,
                min_thrust_n=0.0,
                thrust_limited_delta_v_km_s=imp_result.required_delta_v_km_s,
                applied_delta_v_km_s=imp_result.required_delta_v_km_s if imp_result.executed else 0.0,
                remaining_delta_v_km_s=imp_result.remaining_delta_v_km_s,
                required_attitude_quat_bn=imp_result.required_attitude_quat_bn,
                alignment_ok=imp_result.alignment_ok,
                alignment_angle_rad=imp_result.alignment_angle_rad,
            )
        else:
            raise ValueError("strategy must be 'thrust_limited' or 'impulsive'.")

        below_min = (
            result.commanded_delta_v_km_s > 0.0
            and result.applied_delta_v_km_s == 0.0
            and result.commanded_thrust_n < result.min_thrust_n
        )
        insufficient_dv = (
            result.commanded_delta_v_km_s > 0.0
            and result.applied_delta_v_km_s == 0.0
            and command.available_delta_v_km_s < min(result.commanded_delta_v_km_s, result.thrust_limited_delta_v_km_s)
            and result.alignment_ok
            and not below_min
        )

        should_slew = bool((not result.alignment_ok) or below_min)
        if result.executed:
            action: IntegrationAction = "fire"
            reason = "thrust_executed"
        elif should_slew:
            action = "slew"
            reason = "attitude_alignment_required" if not result.alignment_ok else "below_min_thrust_slew_only"
        else:
            action = "hold"
            reason = "insufficient_delta_v" if insufficient_dv else "maneuver_not_executed"

        decision = IntegratedManeuverDecision(
            action=action,
            executed=result.executed,
            reason=reason,
            should_slew=should_slew,
            required_attitude_quat_bn=None if result.required_attitude_quat_bn is None else result.required_attitude_quat_bn.copy(),
            applied_delta_v_km_s=result.applied_delta_v_km_s,
            remaining_delta_v_km_s=result.remaining_delta_v_km_s,
            alignment_ok=result.alignment_ok,
            below_min_thrust=below_min,
            insufficient_delta_v=insufficient_dv,
            thrust_result=result,
        )
        return next_truth, decision
