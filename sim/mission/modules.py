from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import logging
from typing import Any

import numpy as np

from sim.control.orbit.integrated import IntegratedManeuverCommand, ManeuverStrategy, OrbitalAttitudeManeuverCoordinator
from sim.control.attitude.pose_commands import PoseCommandGenerator
from sim.core.models import Command, StateBelief, StateTruth
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.rocket.models import RocketState, RocketVehicleConfig
from sim.utils.frames import eci_relative_to_ric_rect, ric_curv_to_rect, ric_dcm_ir_from_rv, ric_rect_state_to_eci, ric_rect_to_curv
from sim.utils.quaternion import dcm_to_quaternion_bn, normalize_quaternion, quaternion_to_dcm_bn

logger = logging.getLogger(__name__)


@dataclass
class _MissionExecutiveMode:
    name: str
    strategy: Any | None
    execution: Any | None


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.array(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(x))
    if n <= eps:
        return np.zeros(3, dtype=float)
    return x / n


def _estimate_stack_delta_v_m_s(rocket_state: RocketState, vehicle_cfg: RocketVehicleConfig) -> float:
    stages = vehicle_cfg.stack.stages
    if not stages:
        return 0.0
    i0 = int(max(rocket_state.active_stage_index, 0))
    if i0 >= len(stages):
        return 0.0
    prop_rem = np.array(rocket_state.stage_prop_remaining_kg, dtype=float).reshape(-1)
    dry = np.array([s.dry_mass_kg for s in stages], dtype=float)
    isp = np.array([s.isp_s for s in stages], dtype=float)
    g0 = 9.80665
    m_cur = float(rocket_state.mass_kg)
    dv = 0.0
    for i in range(i0, len(stages)):
        mp = float(prop_rem[i]) if i < prop_rem.size else 0.0
        if mp <= 0.0:
            m_cur -= float(dry[i])
            continue
        m0 = max(m_cur, 1e-6)
        mf = max(m_cur - mp, 1e-6)
        dv += float(isp[i] * g0 * np.log(max(m0 / mf, 1.0)))
        m_cur = mf - float(dry[i])
    return float(max(dv, 0.0))


def _estimate_needed_delta_v_m_s(current_truth: StateTruth, target_truth: StateTruth | None) -> float:
    if target_truth is None:
        return np.inf
    rel_v_km_s = np.array(target_truth.velocity_eci_km_s, dtype=float) - np.array(current_truth.velocity_eci_km_s, dtype=float)
    return float(np.linalg.norm(rel_v_km_s) * 1e3)


def _resolve_angle_tolerance_rad(rad_value: float, deg_value: float | None) -> float:
    if deg_value is not None:
        return float(max(np.deg2rad(float(deg_value)), 0.0))
    return float(max(rad_value, 0.0))


def _resolve_target_state(
    *,
    target_id: str | None,
    use_knowledge_for_targeting: bool,
    own_knowledge: dict[str, StateBelief],
    world_truth: dict[str, StateTruth],
) -> tuple[np.ndarray, np.ndarray] | None:
    if target_id is None:
        return None
    if use_knowledge_for_targeting and target_id in own_knowledge:
        kb = own_knowledge[target_id]
        if kb.state.size >= 6:
            return np.array(kb.state[:3], dtype=float), np.array(kb.state[3:6], dtype=float)
    tgt = world_truth.get(target_id)
    if tgt is None:
        return None
    return np.array(tgt.position_eci_km, dtype=float), np.array(tgt.velocity_eci_km_s, dtype=float)


def _axis_unit_ric(axis_mode: str) -> np.ndarray:
    token = str(axis_mode).strip().upper().replace(" ", "")
    m = {
        "+R": np.array([1.0, 0.0, 0.0], dtype=float),
        "-R": np.array([-1.0, 0.0, 0.0], dtype=float),
        "+I": np.array([0.0, 1.0, 0.0], dtype=float),
        "-I": np.array([0.0, -1.0, 0.0], dtype=float),
        "+C": np.array([0.0, 0.0, 1.0], dtype=float),
        "-C": np.array([0.0, 0.0, -1.0], dtype=float),
    }
    if token in m:
        return m[token]
    raise ValueError("axis_mode must be one of: +R, -R, +I, -I, +C, -C")


def _set_orbit_controller_target(controller: Any | None, desired_state_eci_6: np.ndarray) -> None:
    if controller is None:
        return
    x = np.array(desired_state_eci_6, dtype=float).reshape(-1)
    if x.size != 6:
        return
    if hasattr(controller, "set_target_state"):
        try:
            controller.set_target_state(x)
            return
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Failed to set orbit target state via set_target_state: %s", exc)
    if hasattr(controller, "target_state"):
        try:
            controller.target_state = x
            return
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Failed to set orbit target state via target_state assignment: %s", exc)


def _apply_orbit_controller_intent(controller: Any | None, intent: dict[str, Any]) -> None:
    if controller is None:
        return
    rel_rect = intent.get("desired_relative_ric_rect_6")
    if rel_rect is not None and hasattr(controller, "target_rel_ric_rect"):
        try:
            controller.target_rel_ric_rect = np.array(rel_rect, dtype=float).reshape(6)
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Failed to set orbit controller relative target: %s", exc)
    desired_eci = intent.get("desired_state_eci_6")
    if desired_eci is not None:
        _set_orbit_controller_target(controller, np.array(desired_eci, dtype=float).reshape(6))


def _pointer_dict_to_obj(pointer: dict[str, Any] | None) -> Any | None:
    if not isinstance(pointer, dict):
        return None
    module_name = str(pointer.get("module", "") or "").strip()
    class_name = str(pointer.get("class_name", "") or "").strip()
    function_name = str(pointer.get("function", "") or "").strip()
    params = dict(pointer.get("params", {}) or {})
    if not module_name:
        return None
    try:
        mod = importlib.import_module(module_name)
        if class_name:
            cls = getattr(mod, class_name)
            return cls(**params)
        if function_name:
            return getattr(mod, function_name)
        return mod
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.warning("Failed to construct nested mission pointer %r: %s", pointer, exc)
        return None


def _call_plugin_method(obj: Any | None, method_names: tuple[str, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if obj is None:
        return {}
    for method_name in method_names:
        if not hasattr(obj, method_name):
            continue
        method = getattr(obj, method_name)
        try:
            ret = method(**kwargs)
        except TypeError:
            ret = method(truth=kwargs.get("truth"), t_s=kwargs.get("t_s", 0.0))
        if isinstance(ret, dict):
            return ret
        return {}
    return {}


def _relative_pd_accel_eci(
    *,
    truth: StateTruth,
    target_state_eci: tuple[np.ndarray, np.ndarray] | None,
    desired_relative_ric_rect: np.ndarray,
    kp_pos: float,
    kd_vel: float,
    max_accel_km_s2: float,
) -> np.ndarray:
    if target_state_eci is None:
        return np.zeros(3, dtype=float)
    x_self = np.hstack((np.array(truth.position_eci_km, dtype=float), np.array(truth.velocity_eci_km_s, dtype=float)))
    x_tgt = np.hstack((target_state_eci[0], target_state_eci[1]))
    rel_err = eci_relative_to_ric_rect(x_dep_eci=x_self, x_chief_eci=x_tgt) - np.array(desired_relative_ric_rect, dtype=float).reshape(6)
    a_cmd_ric = -(float(kp_pos) * rel_err[:3] + float(kd_vel) * rel_err[3:6])
    nrm = float(np.linalg.norm(a_cmd_ric))
    amax = float(max(max_accel_km_s2, 0.0))
    if nrm > amax > 0.0:
        a_cmd_ric *= amax / nrm
    c_ir = ric_dcm_ir_from_rv(target_state_eci[0], target_state_eci[1])
    return c_ir @ a_cmd_ric


def _resolve_desired_state_from_inputs(
    *,
    target_id: str | None,
    desired_state_source: str,
    use_knowledge_for_targeting: bool,
    desired_position_eci_km: np.ndarray | None,
    desired_velocity_eci_km_s: np.ndarray | None,
    own_knowledge: dict[str, StateBelief],
    world_truth: dict[str, StateTruth],
) -> tuple[np.ndarray, np.ndarray] | None:
    src = str(desired_state_source).lower()
    if src == "explicit":
        if desired_position_eci_km is None or desired_velocity_eci_km_s is None:
            return None
        return (
            np.array(desired_position_eci_km, dtype=float).reshape(3),
            np.array(desired_velocity_eci_km_s, dtype=float).reshape(3),
        )
    return _resolve_target_state(
        target_id=target_id,
        use_knowledge_for_targeting=use_knowledge_for_targeting,
        own_knowledge=own_knowledge,
        world_truth=world_truth,
    )


def _desired_attitude_for_thrust(
    *,
    truth: StateTruth,
    thrust_eci_km_s2: np.ndarray,
    thruster_direction_body: np.ndarray,
) -> np.ndarray:
    q_req = OrbitalAttitudeManeuverCoordinator().maneuverer.required_attitude_for_delta_v(
        truth=truth,
        delta_v_eci_km_s=np.array(thrust_eci_km_s2, dtype=float),
        thruster_direction_body=np.array(thruster_direction_body, dtype=float),
    )
    if q_req is None:
        return np.array(truth.attitude_quat_bn, dtype=float)
    return np.array(q_req, dtype=float)


@dataclass
class PursuitMissionStrategy:
    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    max_accel_km_s2: float = 0.0
    blind_direction_eci: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        **kwargs: Any,
    ) -> dict[str, Any]:
        tgt = _resolve_target_state(
            target_id=self.target_id,
            use_knowledge_for_targeting=self.use_knowledge_for_targeting,
            own_knowledge=own_knowledge,
            world_truth=world_truth,
        )
        if tgt is None:
            direction = _unit(np.array(self.blind_direction_eci, dtype=float))
        else:
            direction = _unit(tgt[0] - np.array(truth.position_eci_km, dtype=float))
        return {
            "strategy_name": "pursuit",
            "target_id": self.target_id,
            "fallback_thrust_eci_km_s2": float(max(self.max_accel_km_s2, 0.0)) * direction,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {"strategy": "pursuit"},
        }


@dataclass
class EvadeMissionStrategy:
    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    max_accel_km_s2: float = 0.0
    blind_direction_eci: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        **kwargs: Any,
    ) -> dict[str, Any]:
        tgt = _resolve_target_state(
            target_id=self.target_id,
            use_knowledge_for_targeting=self.use_knowledge_for_targeting,
            own_knowledge=own_knowledge,
            world_truth=world_truth,
        )
        if tgt is None:
            direction = _unit(np.array(self.blind_direction_eci, dtype=float))
        else:
            direction = -_unit(tgt[0] - np.array(truth.position_eci_km, dtype=float))
        return {
            "strategy_name": "evade",
            "target_id": self.target_id,
            "fallback_thrust_eci_km_s2": float(max(self.max_accel_km_s2, 0.0)) * direction,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {"strategy": "evade"},
        }


@dataclass
class HoldMissionStrategy:
    attitude_mode: str = "hold_eci"  # hold_eci|hold_ric|sun_track|spotlight|sensing
    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    hold_quat_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    hold_quat_br: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    boresight_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    spotlight_lat_deg: float = 0.0
    spotlight_lon_deg: float = 0.0
    spotlight_alt_km: float = 0.0
    spotlight_ric_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        env: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        mode = str(self.attitude_mode).lower()
        if mode == "hold_eci":
            q_cmd = normalize_quaternion(np.array(self.hold_quat_bn, dtype=float))
        elif mode == "hold_ric":
            c_ir = ric_dcm_ir_from_rv(truth.position_eci_km, truth.velocity_eci_km_s)
            c_br = quaternion_to_dcm_bn(np.array(self.hold_quat_br, dtype=float))
            q_cmd = dcm_to_quaternion_bn(c_br @ c_ir.T)
        elif mode == "sun_track":
            sun_dir = np.array(env.get("sun_dir_eci", np.array([1.0, 0.0, 0.0])), dtype=float)
            q_cmd = PoseCommandGenerator.sun_track(
                truth=truth,
                sun_dir_eci=sun_dir,
                panel_normal_body=np.array(self.boresight_body, dtype=float),
            )
        elif mode == "spotlight":
            q_cmd = PoseCommandGenerator.spotlight_latlon(
                truth=truth,
                latitude_deg=float(self.spotlight_lat_deg),
                longitude_deg=float(self.spotlight_lon_deg),
                altitude_km=float(self.spotlight_alt_km),
                boresight_body=np.array(self.boresight_body, dtype=float),
            )
        elif mode == "sensing":
            q_cmd = PoseCommandGenerator.spotlight_ric_direction(
                truth=truth,
                ric_direction=np.array(self.spotlight_ric_direction, dtype=float),
                boresight_body=np.array(self.boresight_body, dtype=float),
            )
        else:
            tgt = _resolve_target_state(
                target_id=self.target_id,
                use_knowledge_for_targeting=self.use_knowledge_for_targeting,
                own_knowledge=own_knowledge,
                world_truth=world_truth,
            )
            if tgt is None:
                q_cmd = normalize_quaternion(np.array(truth.attitude_quat_bn, dtype=float))
            else:
                q_cmd = PoseCommandGenerator.sun_track(
                    truth=truth,
                    sun_dir_eci=_unit(tgt[0] - np.array(truth.position_eci_km, dtype=float)),
                    panel_normal_body=np.array(self.boresight_body, dtype=float),
                )
        return {
            "strategy_name": "hold",
            "desired_attitude_quat_bn": np.array(q_cmd, dtype=float),
            "mission_mode": {"strategy": "hold", "attitude": self.attitude_mode},
        }


@dataclass
class StationKeepMissionStrategy:
    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    desired_relative_ric_rect: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    kp_pos: float = 1.0e-5
    kd_vel: float = 5.0e-4
    max_accel_km_s2: float = 5.0e-5
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        **kwargs: Any,
    ) -> dict[str, Any]:
        desired_rel = np.array(self.desired_relative_ric_rect, dtype=float).reshape(6)
        tgt = _resolve_target_state(
            target_id=self.target_id,
            use_knowledge_for_targeting=self.use_knowledge_for_targeting,
            own_knowledge=own_knowledge,
            world_truth=world_truth,
        )
        desired_state_eci = None
        fallback_accel = np.zeros(3, dtype=float)
        if tgt is not None:
            desired_state_eci = ric_rect_state_to_eci(desired_rel, tgt[0], tgt[1])
            fallback_accel = _relative_pd_accel_eci(
                truth=truth,
                target_state_eci=tgt,
                desired_relative_ric_rect=desired_rel,
                kp_pos=float(self.kp_pos),
                kd_vel=float(self.kd_vel),
                max_accel_km_s2=float(self.max_accel_km_s2),
            )
        return {
            "strategy_name": "stationkeep",
            "target_id": self.target_id,
            "use_knowledge_for_targeting": bool(self.use_knowledge_for_targeting),
            "desired_relative_ric_rect_6": desired_rel,
            "desired_state_eci_6": desired_state_eci,
            "fallback_thrust_eci_km_s2": fallback_accel,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {"strategy": "stationkeep"},
        }


@dataclass
class InspectMissionStrategy:
    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    desired_relative_ric_rect: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=float))
    boresight_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    kp_pos: float = 1.0e-5
    kd_vel: float = 5.0e-4
    max_accel_km_s2: float = 5.0e-5
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        **kwargs: Any,
    ) -> dict[str, Any]:
        desired_rel = np.array(self.desired_relative_ric_rect, dtype=float).reshape(6)
        tgt = _resolve_target_state(
            target_id=self.target_id,
            use_knowledge_for_targeting=self.use_knowledge_for_targeting,
            own_knowledge=own_knowledge,
            world_truth=world_truth,
        )
        desired_state_eci = None
        desired_attitude = np.array(truth.attitude_quat_bn, dtype=float)
        fallback_accel = np.zeros(3, dtype=float)
        if tgt is not None:
            desired_state_eci = ric_rect_state_to_eci(desired_rel, tgt[0], tgt[1])
            fallback_accel = _relative_pd_accel_eci(
                truth=truth,
                target_state_eci=tgt,
                desired_relative_ric_rect=desired_rel,
                kp_pos=float(self.kp_pos),
                kd_vel=float(self.kd_vel),
                max_accel_km_s2=float(self.max_accel_km_s2),
            )
            los_eci = _unit(tgt[0] - np.array(truth.position_eci_km, dtype=float))
            if float(np.linalg.norm(los_eci)) > 0.0:
                desired_attitude = PoseCommandGenerator.sun_track(
                    truth=truth,
                    sun_dir_eci=los_eci,
                    panel_normal_body=np.array(self.boresight_body, dtype=float),
                )
        return {
            "strategy_name": "inspect",
            "target_id": self.target_id,
            "use_knowledge_for_targeting": bool(self.use_knowledge_for_targeting),
            "desired_relative_ric_rect_6": desired_rel,
            "desired_state_eci_6": desired_state_eci,
            "desired_attitude_quat_bn": np.array(desired_attitude, dtype=float),
            "fallback_thrust_eci_km_s2": fallback_accel,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {"strategy": "inspect"},
        }


@dataclass
class SafeHoldMissionStrategy:
    attitude_mode: str = "hold_current"  # hold_current|hold_eci|sun_track
    hold_quat_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    boresight_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))

    def update(
        self,
        *,
        truth: StateTruth,
        env: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        mode = str(self.attitude_mode).lower()
        if mode == "hold_eci":
            q_cmd = normalize_quaternion(np.array(self.hold_quat_bn, dtype=float))
        elif mode == "sun_track":
            sun_dir = np.array(env.get("sun_dir_eci", np.array([1.0, 0.0, 0.0])), dtype=float)
            q_cmd = PoseCommandGenerator.sun_track(
                truth=truth,
                sun_dir_eci=sun_dir,
                panel_normal_body=np.array(self.boresight_body, dtype=float),
            )
        else:
            q_cmd = normalize_quaternion(np.array(truth.attitude_quat_bn, dtype=float))
        return {
            "strategy_name": "safe_hold",
            "desired_attitude_quat_bn": np.array(q_cmd, dtype=float),
            "fallback_thrust_eci_km_s2": np.zeros(3, dtype=float),
            "command_torque_body_nm": np.zeros(3, dtype=float),
            "mission_mode": {"strategy": "safe_hold", "attitude": mode},
        }


@dataclass
class DesiredStateMissionStrategy:
    target_id: str | None = None
    desired_state_source: str = "target"  # target|explicit
    use_knowledge_for_targeting: bool = True
    desired_position_eci_km: np.ndarray | None = None
    desired_velocity_eci_km_s: np.ndarray | None = None
    align_to_thrust: bool = True

    def update(
        self,
        *,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        **kwargs: Any,
    ) -> dict[str, Any]:
        desired = _resolve_desired_state_from_inputs(
            target_id=self.target_id,
            desired_state_source=self.desired_state_source,
            use_knowledge_for_targeting=self.use_knowledge_for_targeting,
            desired_position_eci_km=self.desired_position_eci_km,
            desired_velocity_eci_km_s=self.desired_velocity_eci_km_s,
            own_knowledge=own_knowledge,
            world_truth=world_truth,
        )
        if desired is None:
            return {
                "strategy_name": "desired_state",
                "mission_mode": {"strategy": "desired_state", "phase": "hold_no_target"},
            }
        x_des = np.hstack((desired[0], desired[1]))
        return {
            "strategy_name": "desired_state",
            "target_id": self.target_id,
            "desired_state_eci_6": x_des,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {"strategy": "desired_state", "source": str(self.desired_state_source)},
        }


@dataclass
class DefensiveMissionStrategy:
    chaser_id: str = "chaser"
    defense_mode: str = "fixed_ric_axis"  # fixed_ric_axis|away_from_chaser
    axis_mode: str = "+R"  # +R|-R|+I|-I|+C|-C
    burn_accel_km_s2: float = 2e-6
    require_finite_knowledge: bool = True
    allow_truth_fallback: bool = False
    align_to_thrust: bool = True

    def _resolve_chaser_state(
        self,
        *,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        kb = own_knowledge.get(self.chaser_id)
        if kb is not None and kb.state.size >= 6:
            x = np.array(kb.state[:6], dtype=float)
            if (not self.require_finite_knowledge) or bool(np.all(np.isfinite(x))):
                return np.array(x[:3], dtype=float), np.array(x[3:6], dtype=float)
        if not self.allow_truth_fallback:
            return None
        tgt = world_truth.get(self.chaser_id)
        if tgt is None:
            return None
        return np.array(tgt.position_eci_km, dtype=float), np.array(tgt.velocity_eci_km_s, dtype=float)

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        **kwargs: Any,
    ) -> dict[str, Any]:
        chaser_state = self._resolve_chaser_state(own_knowledge=own_knowledge, world_truth=world_truth)
        thrust_cmd = np.zeros(3, dtype=float)
        direction_source = "none"
        if chaser_state is not None and float(max(self.burn_accel_km_s2, 0.0)) > 0.0:
            mode = str(self.defense_mode).strip().lower()
            if mode == "away_from_chaser":
                direction_eci = -_unit(chaser_state[0] - np.array(truth.position_eci_km, dtype=float))
                direction_source = "away_from_chaser"
            else:
                direction_eci = ric_dcm_ir_from_rv(
                    np.array(truth.position_eci_km, dtype=float),
                    np.array(truth.velocity_eci_km_s, dtype=float),
                ) @ _axis_unit_ric(self.axis_mode)
                direction_source = "fixed_ric_axis"
            thrust_cmd = float(max(self.burn_accel_km_s2, 0.0)) * _unit(direction_eci)
        return {
            "strategy_name": "defensive",
            "target_id": self.chaser_id,
            "fallback_thrust_eci_km_s2": thrust_cmd,
            "align_to_thrust": bool(self.align_to_thrust),
            "mission_mode": {
                "strategy": "defensive",
                "defense_mode": str(self.defense_mode),
                "axis_mode": str(self.axis_mode),
                "has_chaser_knowledge": bool(chaser_state is not None),
                "direction_source": direction_source,
                "triggered": bool(float(np.linalg.norm(thrust_cmd)) > 0.0),
            },
        }


@dataclass
class MissionExecutiveStrategy:
    initial_mode: str | None = None
    modes: list[dict[str, Any]] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    _modes: dict[str, _MissionExecutiveMode] = field(default_factory=dict, init=False, repr=False)
    _active_mode: str | None = field(default=None, init=False, repr=False)
    _last_transition: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _active_mode_enter_t_s: float | None = field(default=None, init=False, repr=False)
    _transition_armed: dict[int, bool] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        built: dict[str, _MissionExecutiveMode] = {}
        for raw in list(self.modes or []):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "") or "").strip()
            if not name:
                continue
            built[name] = _MissionExecutiveMode(
                name=name,
                strategy=_pointer_dict_to_obj(dict(raw.get("mission_strategy", {}) or {})),
                execution=_pointer_dict_to_obj(dict(raw.get("mission_execution", {}) or {})),
            )
        self._modes = built
        if self.initial_mode is not None and str(self.initial_mode).strip() in self._modes:
            self._active_mode = str(self.initial_mode).strip()
        elif self._modes:
            self._active_mode = next(iter(self._modes.keys()))
        self._transition_armed = {i: True for i, _ in enumerate(list(self.transitions or []))}

    @staticmethod
    def _metric_suffix(trigger: str) -> str | None:
        t = str(trigger).strip().lower()
        if t.startswith("range_"):
            return "km"
        if t == "fuel_below_kg":
            return "kg"
        if t == "fuel_below_fraction":
            return "fraction"
        return None

    def _fuel_metrics(
        self,
        *,
        truth: StateTruth,
        dry_mass_kg: float | None,
        fuel_capacity_kg: float | None,
        rocket_state: RocketState | None,
        rocket_vehicle_cfg: RocketVehicleConfig | None,
    ) -> tuple[float | None, float | None]:
        if rocket_state is not None:
            fuel_kg = float(np.sum(np.clip(np.array(rocket_state.stage_prop_remaining_kg, dtype=float), 0.0, np.inf)))
            fuel0_kg = None
            if rocket_vehicle_cfg is not None:
                fuel0_kg = float(sum(float(s.propellant_mass_kg) for s in rocket_vehicle_cfg.stack.stages))
            fuel_frac = None
            if fuel0_kg is not None and fuel0_kg > 0.0:
                fuel_frac = float(np.clip(fuel_kg / fuel0_kg, 0.0, 1.0))
            return fuel_kg, fuel_frac
        if dry_mass_kg is None or not np.isfinite(float(dry_mass_kg)):
            return None, None
        fuel_kg = float(max(float(truth.mass_kg) - float(dry_mass_kg), 0.0))
        fuel_frac = None
        if fuel_capacity_kg is not None and np.isfinite(float(fuel_capacity_kg)) and float(fuel_capacity_kg) > 0.0:
            fuel_frac = float(np.clip(fuel_kg / float(fuel_capacity_kg), 0.0, 1.0))
        elif fuel_kg <= 0.0:
            fuel_frac = 0.0
        return fuel_kg, fuel_frac

    def _range_km(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        target_id: str | None,
        use_knowledge: bool,
    ) -> float | None:
        tgt = _resolve_target_state(
            target_id=target_id,
            use_knowledge_for_targeting=use_knowledge,
            own_knowledge=own_knowledge,
            world_truth=world_truth,
        )
        if tgt is None:
            return None
        return float(np.linalg.norm(np.array(tgt[0], dtype=float) - np.array(truth.position_eci_km, dtype=float)))

    @staticmethod
    def _transition_applies_to_mode(transition: dict[str, Any], active_mode: str) -> bool:
        raw = transition.get("from_mode", "*")
        if raw is None:
            return True
        if isinstance(raw, (list, tuple)):
            return active_mode in {str(x).strip() for x in raw}
        token = str(raw).strip()
        return token in {"", "*"} or token == active_mode

    def _evaluate_transition(
        self,
        *,
        transition: dict[str, Any],
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        dry_mass_kg: float | None,
        rocket_state: RocketState | None,
        rocket_vehicle_cfg: RocketVehicleConfig | None,
        fuel_capacity_kg: float | None,
    ) -> tuple[bool, str]:
        trigger = str(transition.get("trigger", "") or "").strip().lower()
        if trigger in {"range_lt", "range_gt"}:
            range_km = self._range_km(
                truth=truth,
                own_knowledge=own_knowledge,
                world_truth=world_truth,
                target_id=(None if transition.get("target_id") is None else str(transition.get("target_id"))),
                use_knowledge=bool(transition.get("use_knowledge_for_targeting", True)),
            )
            if range_km is None:
                return False, "range_unavailable"
            threshold_km = float(transition.get("threshold_km", transition.get("threshold", 0.0)) or 0.0)
            if trigger == "range_lt":
                return bool(range_km < threshold_km), f"range_km={range_km:.6f}<threshold_km={threshold_km:.6f}"
            return bool(range_km > threshold_km), f"range_km={range_km:.6f}>threshold_km={threshold_km:.6f}"
        fuel_kg, fuel_frac = self._fuel_metrics(
            truth=truth,
            dry_mass_kg=dry_mass_kg,
            fuel_capacity_kg=fuel_capacity_kg,
            rocket_state=rocket_state,
            rocket_vehicle_cfg=rocket_vehicle_cfg,
        )
        if trigger == "fuel_below_kg":
            if fuel_kg is None:
                return False, "fuel_kg_unavailable"
            threshold_kg = float(transition.get("threshold_kg", transition.get("threshold", 0.0)) or 0.0)
            return bool(fuel_kg < threshold_kg), f"fuel_kg={fuel_kg:.6f}<threshold_kg={threshold_kg:.6f}"
        if trigger == "fuel_below_fraction":
            if fuel_frac is None:
                return False, "fuel_fraction_unavailable"
            threshold = float(transition.get("threshold_fraction", transition.get("threshold", 0.0)) or 0.0)
            return bool(fuel_frac < threshold), f"fuel_fraction={fuel_frac:.6f}<threshold={threshold:.6f}"
        return False, f"unsupported_trigger={trigger}"

    def _metric_value_for_transition(
        self,
        *,
        transition: dict[str, Any],
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        dry_mass_kg: float | None,
        rocket_state: RocketState | None,
        rocket_vehicle_cfg: RocketVehicleConfig | None,
        fuel_capacity_kg: float | None,
    ) -> float | None:
        trigger = str(transition.get("trigger", "") or "").strip().lower()
        if trigger in {"range_lt", "range_gt"}:
            return self._range_km(
                truth=truth,
                own_knowledge=own_knowledge,
                world_truth=world_truth,
                target_id=(None if transition.get("target_id") is None else str(transition.get("target_id"))),
                use_knowledge=bool(transition.get("use_knowledge_for_targeting", True)),
            )
        fuel_kg, fuel_frac = self._fuel_metrics(
            truth=truth,
            dry_mass_kg=dry_mass_kg,
            fuel_capacity_kg=fuel_capacity_kg,
            rocket_state=rocket_state,
            rocket_vehicle_cfg=rocket_vehicle_cfg,
        )
        if trigger == "fuel_below_kg":
            return fuel_kg
        if trigger == "fuel_below_fraction":
            return fuel_frac
        return None

    def _rearm_condition_met(self, *, transition: dict[str, Any], metric_value: float | None) -> bool:
        if metric_value is None or not np.isfinite(float(metric_value)):
            return False
        trigger = str(transition.get("trigger", "") or "").strip().lower()
        suffix = self._metric_suffix(trigger)
        reset_value = None
        if suffix is not None:
            reset_value = transition.get(f"reset_threshold_{suffix}")
        if reset_value is None:
            reset_value = transition.get("reset_threshold")
        if reset_value is None:
            return True
        threshold = float(reset_value)
        if trigger == "range_lt":
            return bool(metric_value > threshold)
        if trigger == "range_gt":
            return bool(metric_value < threshold)
        if trigger in {"fuel_below_kg", "fuel_below_fraction"}:
            return bool(metric_value > threshold)
        return True

    def _maybe_transition(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        dry_mass_kg: float | None,
        rocket_state: RocketState | None,
        rocket_vehicle_cfg: RocketVehicleConfig | None,
        fuel_capacity_kg: float | None,
        t_s: float | None,
    ) -> None:
        active_mode = self._active_mode
        if active_mode is None:
            return
        self._last_transition = None
        for idx, transition in enumerate(list(self.transitions or [])):
            if not isinstance(transition, dict):
                continue
            if not self._transition_applies_to_mode(transition, active_mode):
                continue
            to_mode = str(transition.get("to_mode", "") or "").strip()
            if not to_mode or to_mode not in self._modes:
                continue
            metric_value = self._metric_value_for_transition(
                transition=transition,
                truth=truth,
                own_knowledge=own_knowledge,
                world_truth=world_truth,
                dry_mass_kg=dry_mass_kg,
                rocket_state=rocket_state,
                rocket_vehicle_cfg=rocket_vehicle_cfg,
                fuel_capacity_kg=fuel_capacity_kg,
            )
            if not self._transition_armed.get(idx, True):
                if self._rearm_condition_met(transition=transition, metric_value=metric_value):
                    self._transition_armed[idx] = True
                else:
                    continue
            min_mode_duration_s = float(max(float(transition.get("min_mode_duration_s", 0.0) or 0.0), 0.0))
            if (
                min_mode_duration_s > 0.0
                and self._active_mode_enter_t_s is not None
                and t_s is not None
                and (float(t_s) - float(self._active_mode_enter_t_s)) < (min_mode_duration_s - 1e-12)
            ):
                continue
            fired, detail = self._evaluate_transition(
                transition=transition,
                truth=truth,
                own_knowledge=own_knowledge,
                world_truth=world_truth,
                dry_mass_kg=dry_mass_kg,
                rocket_state=rocket_state,
                rocket_vehicle_cfg=rocket_vehicle_cfg,
                fuel_capacity_kg=fuel_capacity_kg,
            )
            if not fired:
                continue
            self._active_mode = to_mode
            self._active_mode_enter_t_s = None if t_s is None else float(t_s)
            self._transition_armed[idx] = False
            self._last_transition = {
                "from_mode": active_mode,
                "to_mode": to_mode,
                "trigger": str(transition.get("trigger", "") or ""),
                "detail": detail,
                "min_mode_duration_s": float(max(float(transition.get("min_mode_duration_s", 0.0) or 0.0), 0.0)),
            }
            return

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        dry_mass_kg: float | None = None,
        fuel_capacity_kg: float | None = None,
        rocket_state: RocketState | None = None,
        rocket_vehicle_cfg: RocketVehicleConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._modes:
            return {}
        if self._active_mode not in self._modes:
            self._active_mode = next(iter(self._modes.keys()))
        if self._active_mode_enter_t_s is None:
            t_now = kwargs.get("t_s")
            self._active_mode_enter_t_s = None if t_now is None else float(t_now)
        self._maybe_transition(
            truth=truth,
            own_knowledge=own_knowledge,
            world_truth=world_truth,
            dry_mass_kg=dry_mass_kg,
            rocket_state=rocket_state,
            rocket_vehicle_cfg=rocket_vehicle_cfg,
            fuel_capacity_kg=fuel_capacity_kg,
            t_s=(None if kwargs.get("t_s") is None else float(kwargs.get("t_s"))),
        )
        mode = self._modes.get(self._active_mode or "")
        if mode is None:
            return {}
        strategy_out = _call_plugin_method(
            mode.strategy,
            ("update", "plan", "decide"),
            {
                "truth": truth,
                "own_knowledge": own_knowledge,
                "world_truth": world_truth,
                "env": dict(kwargs.get("env", {}) or {}),
                "dry_mass_kg": dry_mass_kg,
                "fuel_capacity_kg": fuel_capacity_kg,
                "rocket_state": rocket_state,
                "rocket_vehicle_cfg": rocket_vehicle_cfg,
                **dict(kwargs or {}),
            },
        )
        out = dict(strategy_out or {})
        if mode.execution is not None:
            out["_mission_execution_override"] = mode.execution
        mission_mode = dict(out.get("mission_mode", {}) or {})
        mission_mode["executive_mode"] = str(mode.name)
        if self._last_transition is not None:
            mission_mode["executive_transition"] = dict(self._last_transition)
        out["mission_mode"] = mission_mode
        return out


@dataclass
class RocketPursuitMissionStrategy:
    target_id: str | None = None
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        **kwargs: Any,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "strategy_name": "rocket_pursuit",
            "orbital_goal": "pursuit",
            "mission_mode": {"strategy": "rocket_pursuit", "orbital_goal": "pursuit"},
            "align_to_thrust": bool(self.align_to_thrust),
        }
        if self.target_id:
            out["target_id"] = str(self.target_id)
        return out


@dataclass
class RocketPredefinedOrbitMissionStrategy:
    predef_target_alt_km: float = 400.0
    predef_target_ecc: float = 0.02
    align_to_thrust: bool = True

    def update(
        self,
        *,
        truth: StateTruth,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "strategy_name": "rocket_predefined_orbit",
            "orbital_goal": "predefined_orbit",
            "predefined_orbit_goal": {
                "target_alt_km": float(self.predef_target_alt_km),
                "target_ecc": float(self.predef_target_ecc),
            },
            "mission_mode": {"strategy": "rocket_predefined_orbit", "orbital_goal": "predefined_orbit"},
            "align_to_thrust": bool(self.align_to_thrust),
        }


@dataclass
class RocketGoNowExecution:
    def update(
        self,
        *,
        intent: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        mission_mode = dict(intent.get("mission_mode", {}) or {})
        mission_mode["launch"] = "go_now"
        return {
            "launch_authorized": True,
            "mission_mode": mission_mode,
        }


@dataclass
class RocketGoWhenPossibleExecution:
    go_when_possible_margin_m_s: float = 0.0
    target_id: str | None = None

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        world_truth: dict[str, StateTruth],
        rocket_state: RocketState | None = None,
        rocket_vehicle_cfg: RocketVehicleConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        target_id = self.target_id or intent.get("target_id")
        target_truth = world_truth.get(str(target_id)) if target_id else None
        dv_needed = _estimate_needed_delta_v_m_s(truth, target_truth)
        dv_avail = _estimate_stack_delta_v_m_s(rocket_state, rocket_vehicle_cfg) if (rocket_state is not None and rocket_vehicle_cfg is not None) else np.inf
        launch_authorized = bool(np.isfinite(dv_avail) and dv_avail >= (dv_needed + float(self.go_when_possible_margin_m_s)))
        mission_mode = dict(intent.get("mission_mode", {}) or {})
        mission_mode["launch"] = "go_when_possible"
        return {
            "launch_authorized": launch_authorized,
            "mission_mode": mission_mode,
        }


@dataclass
class RocketWaitOptimalExecution:
    window_period_s: float = 5400.0
    window_open_duration_s: float = 300.0

    def update(
        self,
        *,
        intent: dict[str, Any],
        t_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        period = max(float(self.window_period_s), 1.0)
        open_dt = float(np.clip(self.window_open_duration_s, 0.0, period))
        launch_authorized = (float(t_s) % period) <= open_dt
        mission_mode = dict(intent.get("mission_mode", {}) or {})
        mission_mode["launch"] = "wait_optimal_window"
        return {
            "launch_authorized": bool(launch_authorized),
            "mission_mode": mission_mode,
        }


@dataclass
class RocketMissionStrategy:
    launch_mode: str = "go_now"  # go_now|go_when_possible|wait_optimal_window
    orbital_goal: str = "pursuit"  # pursuit|predefined_orbit
    target_id: str | None = None
    go_when_possible_margin_m_s: float = 0.0
    window_period_s: float = 5400.0
    window_open_duration_s: float = 300.0
    predef_target_alt_km: float = 400.0
    predef_target_ecc: float = 0.02

    def update(
        self,
        *,
        truth: StateTruth,
        world_truth: dict[str, StateTruth],
        t_s: float,
        rocket_state: RocketState | None = None,
        rocket_vehicle_cfg: RocketVehicleConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Compatibility wrapper for older configs that combined goal selection and launch timing.
        if self.orbital_goal == "predefined_orbit":
            out = RocketPredefinedOrbitMissionStrategy(
                predef_target_alt_km=self.predef_target_alt_km,
                predef_target_ecc=self.predef_target_ecc,
            ).update(truth=truth)
        else:
            out = RocketPursuitMissionStrategy(target_id=self.target_id).update(truth=truth)
        if self.launch_mode == "wait_optimal_window":
            out.update(
                RocketWaitOptimalExecution(
                    window_period_s=self.window_period_s,
                    window_open_duration_s=self.window_open_duration_s,
                ).update(intent=out, t_s=t_s)
            )
        elif self.launch_mode == "go_when_possible":
            out.update(
                RocketGoWhenPossibleExecution(
                    go_when_possible_margin_m_s=self.go_when_possible_margin_m_s,
                    target_id=self.target_id,
                ).update(
                    intent=out,
                    truth=truth,
                    world_truth=world_truth,
                    rocket_state=rocket_state,
                    rocket_vehicle_cfg=rocket_vehicle_cfg,
                )
            )
        else:
            out.update(RocketGoNowExecution().update(intent=out))
        return out


@dataclass
class ControllerPointingExecution:
    align_thruster_to_thrust: bool = True
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    require_attitude_alignment: bool = True
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    use_strategy_fallback_thrust: bool = True
    detumble_enter_rate_rad_s: float | None = None
    detumble_exit_rate_rad_s: float | None = None
    detumble_mode_name: str = "detumble"
    nominal_mode_name: str = "nominal"
    _detumble_latched: bool = False

    def _maybe_update_mode(self, truth: StateTruth, att_belief: StateBelief | None, attitude_controller: Any | None) -> None:
        if self.detumble_enter_rate_rad_s is None or attitude_controller is None or not hasattr(attitude_controller, "set_mode"):
            return
        if att_belief is not None and att_belief.state.size >= 13:
            w = np.array(att_belief.state[10:13], dtype=float)
        else:
            w = np.array(truth.angular_rate_body_rad_s, dtype=float)
        w_norm = float(np.linalg.norm(w))
        enter = float(max(self.detumble_enter_rate_rad_s, 0.0))
        exit_rate = float(max(self.detumble_exit_rate_rad_s if self.detumble_exit_rate_rad_s is not None else enter, 0.0))
        if self._detumble_latched:
            if w_norm <= exit_rate:
                self._detumble_latched = False
        elif w_norm >= enter:
            self._detumble_latched = True
        mode = self.detumble_mode_name if self._detumble_latched else self.nominal_mode_name
        try:
            attitude_controller.set_mode(mode)
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Unable to set attitude controller mode '%s': %s", mode, exc)

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        t_s: float,
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        orb_belief: StateBelief | None = None,
        att_belief: StateBelief | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._maybe_update_mode(truth=truth, att_belief=att_belief, attitude_controller=attitude_controller)
        _apply_orbit_controller_intent(orbit_controller, intent)
        c_orb = orbit_controller.act(orb_belief, t_s, 2.0) if (orbit_controller is not None and orb_belief is not None) else Command.zero()
        thrust_cmd = np.array(c_orb.thrust_eci_km_s2, dtype=float).reshape(3)
        if self.use_strategy_fallback_thrust and float(np.linalg.norm(thrust_cmd)) <= 1e-15 and "fallback_thrust_eci_km_s2" in intent:
            thrust_cmd = np.array(intent.get("fallback_thrust_eci_km_s2"), dtype=float).reshape(3)

        q_des = None
        if "desired_attitude_quat_bn" in intent:
            q_des = np.array(intent.get("desired_attitude_quat_bn"), dtype=float).reshape(-1)
        elif bool(intent.get("align_to_thrust", self.align_thruster_to_thrust)) and float(np.linalg.norm(thrust_cmd)) > 1e-15:
            q_des = PoseCommandGenerator.sun_track(
                truth=truth,
                sun_dir_eci=_unit(thrust_cmd),
                panel_normal_body=np.array(self.thruster_direction_body, dtype=float),
            )
        if q_des is not None and q_des.size == 4 and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(q_des)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set mission execution target quaternion: %s", exc)

        c_att = attitude_controller.act(att_belief, t_s, 2.0) if (attitude_controller is not None and att_belief is not None) else Command.zero()

        tol_rad = _resolve_angle_tolerance_rad(self.alignment_tolerance_rad, self.alignment_tolerance_deg)
        alignment_error_rad = float("nan")
        if float(np.linalg.norm(thrust_cmd)) > 1e-15:
            b_dir = _unit(np.array(self.thruster_direction_body, dtype=float))
            b_to_eci = quaternion_to_dcm_bn(np.array(truth.attitude_quat_bn, dtype=float)).T
            thrust_axis_eci = _unit(b_to_eci @ b_dir)
            thrust_dir = _unit(thrust_cmd)
            alignment_error_rad = float(np.arccos(np.clip(np.dot(thrust_axis_eci, thrust_dir), -1.0, 1.0)))
            if self.require_attitude_alignment and alignment_error_rad > tol_rad:
                thrust_cmd = np.zeros(3, dtype=float)

        mode_flags = dict(c_orb.mode_flags or {})
        if np.isfinite(alignment_error_rad):
            mode_flags["alignment_error_rad"] = float(alignment_error_rad)
            mode_flags["attitude_alignment_satisfied"] = bool(alignment_error_rad <= tol_rad)
        mode_flags["execution"] = "controller_pointing"
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": thrust_cmd,
            "torque_body_nm": np.array(c_att.torque_body_nm, dtype=float).reshape(3),
            "command_mode_flags": mode_flags,
            "desired_attitude_quat_bn": (np.array(q_des, dtype=float).reshape(4) if q_des is not None and q_des.size == 4 else intent.get("desired_attitude_quat_bn")),
            "mission_mode": {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "controller_pointing",
            },
        }


@dataclass
class PredictiveBurnExecution:
    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    lead_time_s: float = 30.0
    predict_dt_s: float = 1.0
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    min_burn_accel_km_s2: float = 1e-12
    mu_km3_s2: float = 398600.4418
    orbit_controller_budget_ms: float = 2.0
    attitude_controller_budget_ms: float = 2.0
    planning_period_s: float | None = None
    skip_orbit_planning_in_detumble_mode: bool = True
    attitude_mode_attr: str = "mode"
    detumble_mode_tokens: tuple[str, ...] = ("detumble",)
    detumble_enter_rate_rad_s: float | None = None
    detumble_exit_rate_rad_s: float | None = None
    detumble_mode_name: str = "detumble"
    nominal_mode_name: str = "nominal"
    _detumble_latched: bool = field(default=False, init=False, repr=False)
    _countdown_s: float = field(default=-1.0, init=False, repr=False)
    _last_plan_t_s: float | None = field(default=None, init=False, repr=False)
    _planned_accel_eci_km_s2: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float), init=False, repr=False)
    _planned_attitude_quat_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float), init=False, repr=False)

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(self.alignment_tolerance_rad, self.alignment_tolerance_deg)

    def _target_state(
        self,
        *,
        intent: dict[str, Any],
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        desired_state = intent.get("desired_state_eci_6")
        if desired_state is not None:
            x_des = np.array(desired_state, dtype=float).reshape(-1)
            if x_des.size >= 6 and np.all(np.isfinite(x_des[:6])):
                return np.array(x_des[:3], dtype=float), np.array(x_des[3:6], dtype=float)
        target_id = intent.get("target_id", self.target_id)
        use_knowledge = bool(intent.get("use_knowledge_for_targeting", self.use_knowledge_for_targeting))
        return _resolve_target_state(
            target_id=(None if target_id is None else str(target_id)),
            use_knowledge_for_targeting=use_knowledge,
            own_knowledge=own_knowledge,
            world_truth=world_truth,
        )

    def _predict_eci(self, x_eci: np.ndarray, horizon_s: float, dt_s: float) -> np.ndarray:
        x = np.array(x_eci, dtype=float).reshape(6)
        n_steps = int(max(np.floor(horizon_s / dt_s), 0))
        rem = float(max(horizon_s - n_steps * dt_s, 0.0))
        for _ in range(n_steps):
            x = propagate_two_body_rk4(x_eci=x, dt_s=dt_s, mu_km3_s2=self.mu_km3_s2, accel_cmd_eci_km_s2=np.zeros(3))
        if rem > 1e-9:
            x = propagate_two_body_rk4(x_eci=x, dt_s=rem, mu_km3_s2=self.mu_km3_s2, accel_cmd_eci_km_s2=np.zeros(3))
        return x

    def _predict_orb_belief_for_controller(
        self,
        *,
        orbit_controller: Any | None,
        self_truth: StateTruth,
        target_state_eci: tuple[np.ndarray, np.ndarray] | None,
        lead_time_s: float,
    ) -> StateBelief:
        x_self = np.hstack((np.array(self_truth.position_eci_km, dtype=float), np.array(self_truth.velocity_eci_km_s, dtype=float)))
        horizon = float(max(lead_time_s, 0.0))
        hdt = float(max(min(self.predict_dt_s, max(horizon, 1e-6)), 1e-6))
        x_self_p = self._predict_eci(x_self, horizon_s=horizon, dt_s=hdt)
        if target_state_eci is None:
            return StateBelief(state=x_self_p, covariance=np.eye(6) * 1e-4, last_update_t_s=float(self_truth.t_s))
        x_tgt = np.hstack((target_state_eci[0], target_state_eci[1]))
        x_tgt_p = self._predict_eci(x_tgt, horizon_s=horizon, dt_s=hdt)
        if orbit_controller is not None and hasattr(orbit_controller, "ric_curv_state_slice"):
            r_c = x_tgt_p[:3]
            v_c = x_tgt_p[3:6]
            r_s = x_self_p[:3]
            v_s = x_self_p[3:6]
            x_rect = eci_relative_to_ric_rect(x_dep_eci=np.hstack((r_s, v_s)), x_chief_eci=np.hstack((r_c, v_c)))
            x_curv = ric_rect_to_curv(x_rect, r0_km=float(np.linalg.norm(r_c)))
            x = np.hstack((x_curv, np.hstack((r_c, v_c))))
            return StateBelief(state=x, covariance=np.eye(12) * 1e-4, last_update_t_s=float(self_truth.t_s))
        return StateBelief(state=x_self_p, covariance=np.eye(6) * 1e-4, last_update_t_s=float(self_truth.t_s))

    def _alignment(self, truth: StateTruth, accel_eci_km_s2: np.ndarray) -> tuple[bool, float]:
        a = np.array(accel_eci_km_s2, dtype=float).reshape(3)
        if float(np.linalg.norm(a)) <= 0.0:
            return True, 0.0
        c_bn = quaternion_to_dcm_bn(truth.attitude_quat_bn)
        t_body = _unit(np.array(self.thruster_direction_body, dtype=float))
        if float(np.linalg.norm(t_body)) <= 0.0:
            return False, float(np.pi)
        thrust_axis_eci = c_bn.T @ t_body
        target_axis_eci = -_unit(a)
        cosang = float(np.clip(np.dot(thrust_axis_eci, target_axis_eci), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        return ang <= float(max(self.alignment_tolerance_rad, 0.0)), ang

    def _attitude_controller_in_detumble_mode(self, attitude_controller: Any | None) -> tuple[bool, str]:
        if attitude_controller is None:
            return False, ""
        attr = str(self.attitude_mode_attr).strip()
        if not attr:
            return False, ""
        try:
            mode_obj = getattr(attitude_controller, attr, "")
        except AttributeError:
            mode_obj = ""
        mode_str = str(mode_obj).strip().lower()
        if not mode_str:
            return False, ""
        tokens = [str(t).strip().lower() for t in tuple(self.detumble_mode_tokens or ()) if str(t).strip()]
        return any(tok in mode_str for tok in tokens), mode_str

    def _effective_planning_period_s(self, dt_s: float) -> float:
        if self.planning_period_s is not None:
            return float(max(self.planning_period_s, 1e-9))
        return float(max(dt_s, 1e-9))

    def _maybe_update_mode(self, truth: StateTruth, att_belief: StateBelief | None, attitude_controller: Any | None) -> None:
        if self.detumble_enter_rate_rad_s is None or attitude_controller is None or not hasattr(attitude_controller, "set_mode"):
            return
        if att_belief is not None and att_belief.state.size >= 13:
            w = np.array(att_belief.state[10:13], dtype=float)
        else:
            w = np.array(truth.angular_rate_body_rad_s, dtype=float)
        w_norm = float(np.linalg.norm(w))
        enter = float(max(self.detumble_enter_rate_rad_s, 0.0))
        exit_rate = float(max(self.detumble_exit_rate_rad_s if self.detumble_exit_rate_rad_s is not None else enter, 0.0))
        if self._detumble_latched:
            if w_norm <= exit_rate:
                self._detumble_latched = False
        elif w_norm >= enter:
            self._detumble_latched = True
        mode = self.detumble_mode_name if self._detumble_latched else self.nominal_mode_name
        try:
            attitude_controller.set_mode(mode)
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("Unable to set attitude controller mode '%s': %s", mode, exc)

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        att_belief: StateBelief | None = None,
        t_s: float,
        dt_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}

        self._maybe_update_mode(truth=truth, att_belief=att_belief, attitude_controller=attitude_controller)
        in_detumble_mode, mode_str = self._attitude_controller_in_detumble_mode(attitude_controller)
        planning_blocked_by_detumble = bool(self.skip_orbit_planning_in_detumble_mode and in_detumble_mode)
        plan_period_s = self._effective_planning_period_s(float(dt_s))
        target_state = self._target_state(intent=intent, own_knowledge=own_knowledge, world_truth=world_truth)
        _apply_orbit_controller_intent(orbit_controller, intent)
        plan_due = bool(self._last_plan_t_s is None or (float(t_s) - float(self._last_plan_t_s)) >= (plan_period_s - 1e-12))
        planned_this_step = False
        lead_time_s = float(max(intent.get("lead_time_s", self.lead_time_s), 0.0))

        if planning_blocked_by_detumble:
            self._countdown_s = -1.0
            self._planned_accel_eci_km_s2 = np.zeros(3, dtype=float)
            self._planned_attitude_quat_bn = np.array(truth.attitude_quat_bn, dtype=float)
        elif self._countdown_s < 0.0 and plan_due:
            b_pred = self._predict_orb_belief_for_controller(
                orbit_controller=orbit_controller,
                self_truth=truth,
                target_state_eci=target_state,
                lead_time_s=lead_time_s,
            )
            c_orb_pred = (
                orbit_controller.act(b_pred, float(t_s), float(max(self.orbit_controller_budget_ms, 1e-9)))
                if orbit_controller is not None
                else Command.zero()
            )
            self._planned_accel_eci_km_s2 = np.array(c_orb_pred.thrust_eci_km_s2, dtype=float).reshape(3)
            if not np.all(np.isfinite(self._planned_accel_eci_km_s2)):
                self._planned_accel_eci_km_s2 = np.zeros(3, dtype=float)
            if float(np.linalg.norm(self._planned_accel_eci_km_s2)) <= 1e-15 and "fallback_thrust_eci_km_s2" in intent:
                self._planned_accel_eci_km_s2 = np.array(intent.get("fallback_thrust_eci_km_s2"), dtype=float).reshape(3)
            dv_pred = self._planned_accel_eci_km_s2 * float(max(self.predict_dt_s, 1e-6))
            q_req = OrbitalAttitudeManeuverCoordinator().maneuverer.required_attitude_for_delta_v(
                truth=truth,
                delta_v_eci_km_s=dv_pred,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
            )
            self._planned_attitude_quat_bn = np.array(q_req if q_req is not None else truth.attitude_quat_bn, dtype=float)
            self._countdown_s = lead_time_s
            self._last_plan_t_s = float(t_s)
            planned_this_step = True

        if (not attitude_disabled) and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(np.array(self._planned_attitude_quat_bn, dtype=float))
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set predictive burn attitude target: %s", exc)
        att_belief_eff = att_belief
        if (not attitude_disabled) and att_belief_eff is None and attitude_controller is not None:
            att_belief_eff = StateBelief(
                state=np.hstack((np.array(truth.attitude_quat_bn, dtype=float), np.array(truth.angular_rate_body_rad_s, dtype=float))),
                covariance=np.eye(7) * 1e-6,
                last_update_t_s=float(truth.t_s),
            )
        c_att = (
            attitude_controller.act(att_belief_eff, float(t_s), float(max(self.attitude_controller_budget_ms, 1e-9)))
            if (not attitude_disabled) and attitude_controller is not None and att_belief_eff is not None
            else Command.zero()
        )

        fire = False
        align_ok, align_angle = self._alignment(truth=truth, accel_eci_km_s2=self._planned_accel_eci_km_s2)
        if attitude_disabled:
            align_ok = True
            align_angle = 0.0
        if planning_blocked_by_detumble:
            fire = False
        else:
            if lead_time_s <= 0.0:
                fire = bool(align_ok and float(np.linalg.norm(self._planned_accel_eci_km_s2)) > float(max(self.min_burn_accel_km_s2, 0.0)))
                self._countdown_s = 0.0
            elif self._countdown_s < 0.0:
                fire = False
            elif self._countdown_s <= float(max(dt_s, 1e-9)):
                if align_ok and float(np.linalg.norm(self._planned_accel_eci_km_s2)) > float(max(self.min_burn_accel_km_s2, 0.0)):
                    fire = True
                self._countdown_s = -1.0
            else:
                self._countdown_s -= float(max(dt_s, 1e-9))

        out["mission_use_integrated_command"] = True
        out["torque_body_nm"] = np.array(c_att.torque_body_nm, dtype=float).reshape(3)
        out["command_mode_flags"] = dict(c_att.mode_flags or {})
        out["desired_attitude_quat_bn"] = np.array(self._planned_attitude_quat_bn, dtype=float)
        out["thrust_eci_km_s2"] = self._planned_accel_eci_km_s2.copy() if fire else np.zeros(3, dtype=float)
        out["mission_mode"] = {
            **dict(intent.get("mission_mode", {}) or {}),
            "execution": "predictive_burn",
            "countdown_s": float(self._countdown_s),
            "fire": bool(fire),
            "alignment_ok": bool(align_ok),
            "alignment_angle_rad": float(align_angle),
            "orbit_controller_budget_ms": float(self.orbit_controller_budget_ms),
            "attitude_controller_budget_ms": float(self.attitude_controller_budget_ms),
            "planning_blocked_by_detumble": bool(planning_blocked_by_detumble),
            "attitude_controller_mode": str(mode_str),
            "plan_due": bool(plan_due),
            "planned_this_step": bool(planned_this_step),
            "planning_period_s": float(plan_period_s),
        }
        return out


@dataclass
class DirectIntegratedExecution:
    align_thruster_to_thrust: bool = True
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    use_strategy_fallback_thrust: bool = True
    use_orbit_controller: bool = False
    orbit_controller_budget_ms: float = 2.0
    attitude_controller_budget_ms: float = 2.0

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        t_s: float,
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        orb_belief: StateBelief | None = None,
        att_belief: StateBelief | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        _apply_orbit_controller_intent(orbit_controller, intent)
        thrust_cmd = np.array(intent.get("command_thrust_eci_km_s2", np.zeros(3)), dtype=float).reshape(3)
        if float(np.linalg.norm(thrust_cmd)) <= 1e-15 and self.use_orbit_controller and orbit_controller is not None and orb_belief is not None:
            c_orb = orbit_controller.act(orb_belief, float(t_s), float(max(self.orbit_controller_budget_ms, 1e-9)))
            thrust_cmd = np.array(c_orb.thrust_eci_km_s2, dtype=float).reshape(3)
        if float(np.linalg.norm(thrust_cmd)) <= 1e-15 and self.use_strategy_fallback_thrust and "fallback_thrust_eci_km_s2" in intent:
            thrust_cmd = np.array(intent.get("fallback_thrust_eci_km_s2"), dtype=float).reshape(3)

        q_des = intent.get("desired_attitude_quat_bn")
        if q_des is None and bool(intent.get("align_to_thrust", self.align_thruster_to_thrust)) and float(np.linalg.norm(thrust_cmd)) > 1e-15:
            q_des = _desired_attitude_for_thrust(
                truth=truth,
                thrust_eci_km_s2=thrust_cmd,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
            )
        q_des_arr = None if q_des is None else np.array(q_des, dtype=float).reshape(4)

        if q_des_arr is not None and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(q_des_arr)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set direct execution attitude target: %s", exc)
        c_att = (
            attitude_controller.act(att_belief, float(t_s), float(max(self.attitude_controller_budget_ms, 1e-9)))
            if (attitude_controller is not None and att_belief is not None)
            else Command.zero()
        )

        torque_cmd = np.array(intent.get("command_torque_body_nm", c_att.torque_body_nm), dtype=float).reshape(3)
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": thrust_cmd,
            "torque_body_nm": torque_cmd,
            "desired_attitude_quat_bn": q_des_arr,
            "command_mode_flags": {"execution": "direct_integrated"},
            "mission_mode": {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "direct_integrated",
            },
        }


@dataclass
class IntegratedCommandExecution:
    require_attitude_alignment: bool = True
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    min_burn_accel_km_s2: float = 1e-12
    orbit_controller_budget_ms: float = 2.0
    attitude_controller_budget_ms: float = 2.0

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(self.alignment_tolerance_rad, self.alignment_tolerance_deg)

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        orb_belief: StateBelief | None = None,
        att_belief: StateBelief | None = None,
        t_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}
        _apply_orbit_controller_intent(orbit_controller, intent)

        c_orb = (
            orbit_controller.act(orb_belief, float(t_s), float(max(self.orbit_controller_budget_ms, 1e-9)))
            if (orbit_controller is not None and orb_belief is not None)
            else Command.zero()
        )
        thrust_cmd = np.array(c_orb.thrust_eci_km_s2, dtype=float).reshape(3)
        burn_requested = float(np.linalg.norm(thrust_cmd)) > float(max(self.min_burn_accel_km_s2, 0.0))

        align_ok = True
        align_angle = 0.0
        required_q = np.array(truth.attitude_quat_bn, dtype=float)
        if burn_requested and self.require_attitude_alignment and (not attitude_disabled):
            align_ok, align_angle = PredictiveBurnExecution._alignment(self, truth=truth, accel_eci_km_s2=thrust_cmd)
            required_q = _desired_attitude_for_thrust(
                truth=truth,
                thrust_eci_km_s2=thrust_cmd,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
            )

        if (not attitude_disabled) and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(np.array(required_q, dtype=float))
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set attitude target in IntegratedCommandExecution: %s", exc)
        c_att = (
            attitude_controller.act(att_belief, float(t_s), float(max(self.attitude_controller_budget_ms, 1e-9)))
            if (not attitude_disabled) and attitude_controller is not None and att_belief is not None
            else Command.zero()
        )

        if burn_requested and align_ok:
            out["thrust_eci_km_s2"] = thrust_cmd
            phase = "burn"
        else:
            out["thrust_eci_km_s2"] = np.zeros(3, dtype=float)
            phase = "slew" if burn_requested else "hold"
        out["torque_body_nm"] = np.array(c_att.torque_body_nm, dtype=float).reshape(3)
        out["desired_attitude_quat_bn"] = np.array(required_q, dtype=float)
        out["mission_use_integrated_command"] = True
        out["mission_mode"] = {
            **dict(intent.get("mission_mode", {}) or {}),
            "execution": "integrated_command",
            "phase": phase,
            "burn_requested": bool(burn_requested),
            "alignment_ok": bool(align_ok),
            "alignment_angle_rad": float(align_angle),
        }
        return out


@dataclass
class ImpulsiveExecution:
    align_thruster_to_thrust: bool = True
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    require_attitude_alignment: bool = True
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    use_strategy_fallback_thrust: bool = True
    pulse_period_s: float = 60.0
    pulse_width_s: float = 5.0
    pulse_phase_s: float = 0.0
    min_burn_accel_km_s2: float = 1e-12
    orbit_controller_budget_ms: float = 2.0
    attitude_controller_budget_ms: float = 2.0

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(self.alignment_tolerance_rad, self.alignment_tolerance_deg)

    def _pulse_active(self, t_s: float) -> bool:
        period = float(max(self.pulse_period_s, 1e-9))
        width = float(np.clip(self.pulse_width_s, 0.0, period))
        phase = float(self.pulse_phase_s)
        tau = (float(t_s) - phase) % period
        return tau <= width

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        t_s: float,
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        orb_belief: StateBelief | None = None,
        att_belief: StateBelief | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        _apply_orbit_controller_intent(orbit_controller, intent)
        c_orb = (
            orbit_controller.act(orb_belief, float(t_s), float(max(self.orbit_controller_budget_ms, 1e-9)))
            if (orbit_controller is not None and orb_belief is not None)
            else Command.zero()
        )
        thrust_cmd = np.array(c_orb.thrust_eci_km_s2, dtype=float).reshape(3)
        if float(np.linalg.norm(thrust_cmd)) <= 1e-15 and self.use_strategy_fallback_thrust and "fallback_thrust_eci_km_s2" in intent:
            thrust_cmd = np.array(intent.get("fallback_thrust_eci_km_s2"), dtype=float).reshape(3)

        q_des = intent.get("desired_attitude_quat_bn")
        if q_des is None and bool(intent.get("align_to_thrust", self.align_thruster_to_thrust)) and float(np.linalg.norm(thrust_cmd)) > 1e-15:
            q_des = _desired_attitude_for_thrust(
                truth=truth,
                thrust_eci_km_s2=thrust_cmd,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
            )
        q_des_arr = None if q_des is None else np.array(q_des, dtype=float).reshape(4)
        if q_des_arr is not None and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(q_des_arr)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set impulsive execution attitude target: %s", exc)
        c_att = (
            attitude_controller.act(att_belief, float(t_s), float(max(self.attitude_controller_budget_ms, 1e-9)))
            if (attitude_controller is not None and att_belief is not None)
            else Command.zero()
        )

        tol_rad = float(max(self.alignment_tolerance_rad, 0.0))
        alignment_ok = True
        alignment_angle_rad = 0.0
        if float(np.linalg.norm(thrust_cmd)) > float(max(self.min_burn_accel_km_s2, 0.0)):
            alignment_ok, alignment_angle_rad = PredictiveBurnExecution._alignment(self, truth=truth, accel_eci_km_s2=thrust_cmd)
        pulse_active = self._pulse_active(float(t_s))
        fire = bool(
            pulse_active
            and float(np.linalg.norm(thrust_cmd)) > float(max(self.min_burn_accel_km_s2, 0.0))
            and ((not self.require_attitude_alignment) or alignment_ok)
        )
        if self.require_attitude_alignment and alignment_angle_rad > tol_rad:
            fire = False

        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": thrust_cmd if fire else np.zeros(3, dtype=float),
            "torque_body_nm": np.array(c_att.torque_body_nm, dtype=float).reshape(3),
            "desired_attitude_quat_bn": q_des_arr,
            "command_mode_flags": {
                **dict(c_att.mode_flags or {}),
                "execution": "impulsive",
                "pulse_active": bool(pulse_active),
                "alignment_ok": bool(alignment_ok),
            },
            "mission_mode": {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "impulsive",
                "pulse_active": bool(pulse_active),
                "fire": bool(fire),
                "alignment_ok": bool(alignment_ok),
                "alignment_angle_rad": float(alignment_angle_rad),
            },
        }


@dataclass
class BudgetedEndStateExecution:
    strategy: ManeuverStrategy = "thrust_limited"
    max_thrust_n: float = 0.2
    min_thrust_n: float = 0.0
    burn_dt_s: float = 1.0
    available_delta_v_km_s: float = 0.5
    require_attitude_alignment: bool = True
    thruster_position_body_m: np.ndarray | None = None
    thruster_direction_body: np.ndarray | None = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    terminate_on_velocity_tolerance_km_s: float = 1e-5
    _coordinator: OrbitalAttitudeManeuverCoordinator = field(default_factory=OrbitalAttitudeManeuverCoordinator, init=False, repr=False)

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(self.alignment_tolerance_rad, self.alignment_tolerance_deg)

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}
        x_des = intent.get("desired_state_eci_6")
        if x_des is None:
            out["mission_mode"] = {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "budgeted_end_state",
                "phase": "hold_no_target",
            }
            return out
        x_des_arr = np.array(x_des, dtype=float).reshape(-1)
        if x_des_arr.size != 6:
            out["mission_mode"] = {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "budgeted_end_state",
                "phase": "hold_no_target",
            }
            return out
        dv_eci = x_des_arr[3:6] - np.array(truth.velocity_eci_km_s, dtype=float)
        if float(np.linalg.norm(dv_eci)) <= max(float(self.terminate_on_velocity_tolerance_km_s), 0.0):
            out["mission_mode"] = {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "budgeted_end_state",
                "phase": "on_target",
            }
            return out

        cmd = IntegratedManeuverCommand(
            delta_v_eci_km_s=dv_eci,
            available_delta_v_km_s=float(max(self.available_delta_v_km_s, 0.0)),
            strategy=str(self.strategy),  # type: ignore[arg-type]
            max_thrust_n=float(max(self.max_thrust_n, 0.0)),
            dt_s=float(max(self.burn_dt_s, 1e-6)),
            min_thrust_n=float(max(self.min_thrust_n, 0.0)),
            require_attitude_alignment=(bool(self.require_attitude_alignment) and (not attitude_disabled)),
            thruster_position_body_m=None if self.thruster_position_body_m is None else np.array(self.thruster_position_body_m, dtype=float),
            thruster_direction_body=None if self.thruster_direction_body is None else np.array(self.thruster_direction_body, dtype=float),
            alignment_tolerance_rad=float(max(self.alignment_tolerance_rad, 0.0)),
        )
        _, decision = self._coordinator.execute(truth=truth, command=cmd)
        self.available_delta_v_km_s = float(max(decision.remaining_delta_v_km_s, 0.0))

        if decision.required_attitude_quat_bn is not None:
            out["desired_attitude_quat_bn"] = np.array(decision.required_attitude_quat_bn, dtype=float)
        if decision.executed and decision.applied_delta_v_km_s > 0.0:
            out["thrust_eci_km_s2"] = _unit(dv_eci) * (float(decision.applied_delta_v_km_s) / float(max(self.burn_dt_s, 1e-6)))

        out["mission_mode"] = {
            **dict(intent.get("mission_mode", {}) or {}),
            "execution": "budgeted_end_state",
            "phase": decision.action,
            "reason": decision.reason,
            "alignment_ok": bool(decision.alignment_ok),
            "remaining_delta_v_km_s": float(self.available_delta_v_km_s),
            "applied_delta_v_km_s": float(decision.applied_delta_v_km_s),
        }
        return out


@dataclass
class SafeHoldExecution:
    attitude_controller_budget_ms: float = 2.0

    def update(
        self,
        *,
        intent: dict[str, Any],
        truth: StateTruth,
        t_s: float,
        attitude_controller: Any | None = None,
        att_belief: StateBelief | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        q_des = intent.get("desired_attitude_quat_bn", np.array(truth.attitude_quat_bn, dtype=float))
        q_des_arr = np.array(q_des, dtype=float).reshape(4)
        if attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(q_des_arr)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set safe-hold attitude target: %s", exc)
        c_att = (
            attitude_controller.act(att_belief, float(t_s), float(max(self.attitude_controller_budget_ms, 1e-9)))
            if (attitude_controller is not None and att_belief is not None)
            else Command.zero()
        )
        return {
            "mission_use_integrated_command": True,
            "thrust_eci_km_s2": np.zeros(3, dtype=float),
            "torque_body_nm": np.array(c_att.torque_body_nm, dtype=float).reshape(3),
            "desired_attitude_quat_bn": q_des_arr,
            "command_mode_flags": {
                **dict(c_att.mode_flags or {}),
                "execution": "safe_hold",
            },
            "mission_mode": {
                **dict(intent.get("mission_mode", {}) or {}),
                "execution": "safe_hold",
            },
        }


@dataclass
class AttitudeDetumbleGateMissionModule:
    """
    Mission-side attitude mode gate:
    - Use detumble mode when |w_body| exceeds enter threshold.
    - Return to nominal mode when |w_body| falls below exit threshold.
    Requires attitude controller to implement `set_mode("detumble"|"nominal")`.
    """

    enter_rate_rad_s: float = 0.03
    exit_rate_rad_s: float = 0.015
    prefer_attitude_belief_rate: bool = True
    detumble_mode_name: str = "detumble"
    nominal_mode_name: str = "nominal"
    _detumble_latched: bool = False

    def __post_init__(self) -> None:
        self.enter_rate_rad_s = float(max(self.enter_rate_rad_s, 0.0))
        self.exit_rate_rad_s = float(max(self.exit_rate_rad_s, 0.0))
        if self.exit_rate_rad_s > self.enter_rate_rad_s:
            self.exit_rate_rad_s = self.enter_rate_rad_s

    def _rate_norm_rad_s(self, truth: StateTruth, att_belief: StateBelief | None) -> float:
        if self.prefer_attitude_belief_rate and att_belief is not None and att_belief.state.size >= 13:
            w = np.array(att_belief.state[10:13], dtype=float)
            if np.all(np.isfinite(w)):
                return float(np.linalg.norm(w))
        w_t = np.array(truth.angular_rate_body_rad_s, dtype=float)
        return float(np.linalg.norm(w_t))

    def update(
        self,
        *,
        truth: StateTruth,
        attitude_controller: Any | None = None,
        att_belief: StateBelief | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        w_norm = self._rate_norm_rad_s(truth=truth, att_belief=att_belief)
        if self._detumble_latched:
            if w_norm <= self.exit_rate_rad_s:
                self._detumble_latched = False
        else:
            if w_norm >= self.enter_rate_rad_s:
                self._detumble_latched = True

        desired_mode = self.detumble_mode_name if self._detumble_latched else self.nominal_mode_name
        controller_accepts_mode = bool(attitude_controller is not None and hasattr(attitude_controller, "set_mode"))
        if controller_accepts_mode:
            try:
                attitude_controller.set_mode(desired_mode)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Unable to set attitude controller mode '%s': %s", desired_mode, exc)
                controller_accepts_mode = False

        return {
            "attitude_gate": {
                "mode": str(desired_mode),
                "rate_norm_rad_s": float(w_norm),
                "enter_rate_rad_s": float(self.enter_rate_rad_s),
                "exit_rate_rad_s": float(self.exit_rate_rad_s),
                "controller_accepts_mode": bool(controller_accepts_mode),
            }
        }


@dataclass
class SatelliteMissionModule:
    orbital_mode: str = "coast"  # coast|pursuit_knowledge|evade_knowledge|pursuit_blind|evade_blind
    attitude_mode: str = "hold_eci"  # hold_eci|hold_ric|spotlight|sun_track|pursuit|evade|sensing
    target_id: str | None = None
    max_accel_km_s2: float = 0.0
    blind_direction_eci: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    hold_quat_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    hold_quat_br: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    boresight_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    spotlight_lat_deg: float = 0.0
    spotlight_lon_deg: float = 0.0
    spotlight_alt_km: float = 0.0
    spotlight_ric_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    use_knowledge_for_targeting: bool = True

    def _target_state(self, own_knowledge: dict[str, StateBelief], world_truth: dict[str, StateTruth]) -> tuple[np.ndarray, np.ndarray] | None:
        if self.target_id is None:
            return None
        if self.use_knowledge_for_targeting and self.target_id in own_knowledge:
            kb = own_knowledge[self.target_id]
            if kb.state.size >= 6:
                return np.array(kb.state[:3], dtype=float), np.array(kb.state[3:6], dtype=float)
        tgt = world_truth.get(self.target_id)
        if tgt is None:
            return None
        return np.array(tgt.position_eci_km, dtype=float), np.array(tgt.velocity_eci_km_s, dtype=float)

    def _orbital_command(self, truth: StateTruth, own_knowledge: dict[str, StateBelief], world_truth: dict[str, StateTruth]) -> np.ndarray:
        amax = float(max(self.max_accel_km_s2, 0.0))
        if self.orbital_mode == "coast" or amax <= 0.0:
            return np.zeros(3, dtype=float)
        if self.orbital_mode in ("pursuit_knowledge", "evade_knowledge", "pursuit_blind", "evade_blind"):
            tgt = self._target_state(own_knowledge=own_knowledge, world_truth=world_truth)
            if tgt is None:
                d = _unit(np.array(self.blind_direction_eci, dtype=float))
            else:
                d = _unit(tgt[0] - np.array(truth.position_eci_km, dtype=float))
            if self.orbital_mode.startswith("evade"):
                d = -d
            return amax * d
        return np.zeros(3, dtype=float)

    def _attitude_command(
        self,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        env: dict[str, Any],
        orbital_accel_cmd: np.ndarray,
    ) -> np.ndarray:
        mode = str(self.attitude_mode).lower()
        if mode == "hold_eci":
            return normalize_quaternion(np.array(self.hold_quat_bn, dtype=float))
        if mode == "hold_ric":
            c_ir = ric_dcm_ir_from_rv(truth.position_eci_km, truth.velocity_eci_km_s)
            c_br = quaternion_to_dcm_bn(np.array(self.hold_quat_br, dtype=float))
            c_bn = c_br @ c_ir.T
            return dcm_to_quaternion_bn(c_bn)
        if mode == "sun_track":
            sun_dir = np.array(env.get("sun_dir_eci", np.array([1.0, 0.0, 0.0])), dtype=float)
            return PoseCommandGenerator.sun_track(
                truth=truth,
                sun_dir_eci=sun_dir,
                panel_normal_body=np.array(self.boresight_body, dtype=float),
            )
        if mode == "spotlight":
            return PoseCommandGenerator.spotlight_latlon(
                truth=truth,
                latitude_deg=float(self.spotlight_lat_deg),
                longitude_deg=float(self.spotlight_lon_deg),
                altitude_km=float(self.spotlight_alt_km),
                boresight_body=np.array(self.boresight_body, dtype=float),
            )
        if mode == "sensing":
            return PoseCommandGenerator.spotlight_ric_direction(
                truth=truth,
                ric_direction=np.array(self.spotlight_ric_direction, dtype=float),
                boresight_body=np.array(self.boresight_body, dtype=float),
            )
        if mode in ("pursuit", "evade"):
            d = _unit(np.array(orbital_accel_cmd, dtype=float))
            if np.linalg.norm(d) <= 0.0:
                return normalize_quaternion(np.array(truth.attitude_quat_bn, dtype=float))
            if mode == "evade":
                d = -d
            return PoseCommandGenerator.sun_track(
                truth=truth,
                sun_dir_eci=d,
                panel_normal_body=np.array(self.boresight_body, dtype=float),
            )
        tgt = self._target_state(own_knowledge=own_knowledge, world_truth=world_truth)
        if tgt is not None:
            d = _unit(tgt[0] - np.array(truth.position_eci_km, dtype=float))
            return PoseCommandGenerator.sun_track(
                truth=truth,
                sun_dir_eci=d,
                panel_normal_body=np.array(self.boresight_body, dtype=float),
            )
        return normalize_quaternion(np.array(truth.attitude_quat_bn, dtype=float))

    def update(
        self,
        *,
        object_id: str,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        env: dict[str, Any],
        t_s: float,
        dt_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        a_cmd = self._orbital_command(truth=truth, own_knowledge=own_knowledge, world_truth=world_truth)
        q_cmd = self._attitude_command(
            truth=truth,
            own_knowledge=own_knowledge,
            world_truth=world_truth,
            env=env,
            orbital_accel_cmd=a_cmd,
        )
        return {
            "thrust_eci_km_s2": np.array(a_cmd, dtype=float),
            "desired_attitude_quat_bn": np.array(q_cmd, dtype=float),
            "mission_mode": {"orbital": self.orbital_mode, "attitude": self.attitude_mode},
        }


@dataclass
class DefensiveRICAxisBurnMissionModule:
    """
    Basic defensive maneuver:
    - Select one fixed burn direction in the RIC frame: +R/-R/+I/-I/+C/-C.
    - Burn only when valid knowledge of the chaser is available.
    """

    chaser_id: str = "chaser"
    axis_mode: str = "+R"  # +R|-R|+I|-I|+C|-C
    burn_accel_km_s2: float = 2e-6
    require_finite_knowledge: bool = True
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    min_burn_accel_km_s2: float = 1e-12

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(self.alignment_tolerance_rad, self.alignment_tolerance_deg)

    def _has_chaser_knowledge(self, own_knowledge: dict[str, StateBelief]) -> bool:
        kb = own_knowledge.get(self.chaser_id)
        if kb is None or kb.state.size < 6:
            return False
        if not self.require_finite_knowledge:
            return True
        x = np.array(kb.state[:6], dtype=float)
        return bool(np.all(np.isfinite(x)))

    def _alignment(self, truth: StateTruth, accel_eci_km_s2: np.ndarray) -> tuple[bool, float]:
        a = np.array(accel_eci_km_s2, dtype=float).reshape(3)
        if float(np.linalg.norm(a)) <= 0.0:
            return True, 0.0
        c_bn = quaternion_to_dcm_bn(truth.attitude_quat_bn)
        t_body = _unit(np.array(self.thruster_direction_body, dtype=float))
        if float(np.linalg.norm(t_body)) <= 0.0:
            return False, float(np.pi)
        thrust_axis_eci = c_bn.T @ t_body
        target_axis_eci = -_unit(a)
        cosang = float(np.clip(np.dot(thrust_axis_eci, target_axis_eci), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        return ang <= float(max(self.alignment_tolerance_rad, 0.0)), ang

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        attitude_controller: Any | None = None,
        att_belief: StateBelief | None = None,
        t_s: float,
        dt_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}
        know = self._has_chaser_knowledge(own_knowledge)
        if not know:
            out["mission_use_integrated_command"] = True
            out["thrust_eci_km_s2"] = np.zeros(3, dtype=float)
            out["torque_body_nm"] = np.zeros(3, dtype=float)
            out["mission_mode"] = {
                "type": "defensive_ric_axis_burn",
                "axis_mode": str(self.axis_mode),
                "triggered": False,
                "has_chaser_knowledge": False,
                "alignment_ok": False,
            }
            return out

        a_mag = float(max(self.burn_accel_km_s2, 0.0))
        if a_mag <= 0.0:
            out["mission_use_integrated_command"] = True
            out["thrust_eci_km_s2"] = np.zeros(3, dtype=float)
            out["torque_body_nm"] = np.zeros(3, dtype=float)
            out["mission_mode"] = {
                "type": "defensive_ric_axis_burn",
                "axis_mode": str(self.axis_mode),
                "triggered": False,
                "has_chaser_knowledge": True,
                "alignment_ok": False,
                "reason": "zero_burn_accel",
            }
            return out

        dir_ric = _axis_unit_ric(self.axis_mode)
        c_ir = ric_dcm_ir_from_rv(np.array(truth.position_eci_km, dtype=float), np.array(truth.velocity_eci_km_s, dtype=float))
        dir_eci = c_ir @ dir_ric
        thrust_cmd = a_mag * _unit(dir_eci)
        q_req = OrbitalAttitudeManeuverCoordinator().maneuverer.required_attitude_for_delta_v(
            truth=truth,
            delta_v_eci_km_s=np.array(thrust_cmd, dtype=float),
            thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
        )
        q_des = np.array(q_req if q_req is not None else truth.attitude_quat_bn, dtype=float)

        if (not attitude_disabled) and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(q_des)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set defensive burn attitude target: %s", exc)

        att_belief_eff = att_belief
        if (not attitude_disabled) and att_belief_eff is None and attitude_controller is not None:
            att_belief_eff = StateBelief(
                state=np.hstack((np.array(truth.attitude_quat_bn, dtype=float), np.array(truth.angular_rate_body_rad_s, dtype=float))),
                covariance=np.eye(7) * 1e-6,
                last_update_t_s=float(truth.t_s),
            )
        c_att = (
            attitude_controller.act(att_belief_eff, float(t_s), 2.0)
            if (not attitude_disabled) and attitude_controller is not None and att_belief_eff is not None
            else Command.zero()
        )

        align_ok, align_angle = self._alignment(truth=truth, accel_eci_km_s2=np.array(thrust_cmd, dtype=float))
        if attitude_disabled or attitude_controller is None:
            # Permit immediate burns when no attitude loop is present to execute slews.
            align_ok = True
        fire = bool(align_ok and float(np.linalg.norm(thrust_cmd)) > float(max(self.min_burn_accel_km_s2, 0.0)))

        out["mission_use_integrated_command"] = True
        out["torque_body_nm"] = np.array(c_att.torque_body_nm, dtype=float).reshape(3)
        out["command_mode_flags"] = dict(c_att.mode_flags or {})
        out["desired_attitude_quat_bn"] = q_des
        out["thrust_eci_km_s2"] = np.array(thrust_cmd, dtype=float) if fire else np.zeros(3, dtype=float)
        out["mission_mode"] = {
            "type": "defensive_ric_axis_burn",
            "axis_mode": str(self.axis_mode),
            "triggered": True,
            "has_chaser_knowledge": True,
            "alignment_ok": bool(align_ok),
            "alignment_angle_rad": float(align_angle),
            "fire": bool(fire),
        }
        return out


@dataclass
class RocketMissionModule:
    launch_mode: str = "go_now"  # go_now|go_when_possible|wait_optimal_window
    orbital_goal: str = "pursuit"  # pursuit|predefined_orbit
    target_id: str | None = None
    go_when_possible_margin_m_s: float = 100.0
    window_period_s: float = 5400.0
    window_open_duration_s: float = 300.0
    predef_target_alt_km: float = 500.0
    predef_target_ecc: float = 0.0

    def _in_window(self, t_s: float) -> bool:
        p = max(float(self.window_period_s), 1.0)
        w = max(float(self.window_open_duration_s), 0.0)
        tau = float(t_s % p)
        return tau <= w

    def update(
        self,
        *,
        object_id: str,
        truth: StateTruth,
        world_truth: dict[str, StateTruth],
        t_s: float,
        rocket_state: RocketState | None = None,
        rocket_vehicle_cfg: RocketVehicleConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        launch_authorized = True
        if self.launch_mode == "go_now":
            launch_authorized = True
        elif self.launch_mode == "wait_optimal_window":
            launch_authorized = self._in_window(float(t_s))
        elif self.launch_mode == "go_when_possible":
            if rocket_state is None or rocket_vehicle_cfg is None:
                launch_authorized = True
            else:
                tgt = world_truth.get(str(self.target_id)) if self.target_id else None
                dv_avail = _estimate_stack_delta_v_m_s(rocket_state=rocket_state, vehicle_cfg=rocket_vehicle_cfg)
                dv_need = _estimate_needed_delta_v_m_s(current_truth=truth, target_truth=tgt)
                launch_authorized = dv_need <= (dv_avail - float(self.go_when_possible_margin_m_s))
        out: dict[str, Any] = {"launch_authorized": bool(launch_authorized)}
        out["mission_mode"] = {"launch": self.launch_mode, "goal": self.orbital_goal}
        return out


@dataclass
class EndStateManeuverMissionModule:
    """
    Mission-level orbital/attitude coupling module.

    Flow:
    1) Build desired end state from explicit target or object knowledge.
    2) Compute required delta-v (current v -> desired v).
    3) Ask integrated maneuver coordinator for fire/slew/hold decision.
    4) Emit attitude target for alignment; emit thrust only when burn is allowed.
    """

    target_id: str | None = None
    use_knowledge_for_targeting: bool = True
    desired_position_eci_km: np.ndarray | None = None
    desired_velocity_eci_km_s: np.ndarray | None = None
    desired_state_source: str = "target"  # target|explicit
    strategy: ManeuverStrategy = "thrust_limited"
    max_thrust_n: float = 0.2
    min_thrust_n: float = 0.0
    burn_dt_s: float = 1.0
    available_delta_v_km_s: float = 0.5
    require_attitude_alignment: bool = True
    thruster_position_body_m: np.ndarray | None = None
    thruster_direction_body: np.ndarray | None = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    terminate_on_velocity_tolerance_km_s: float = 1e-5
    _coordinator: OrbitalAttitudeManeuverCoordinator = field(default_factory=OrbitalAttitudeManeuverCoordinator, init=False, repr=False)

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(self.alignment_tolerance_rad, self.alignment_tolerance_deg)

    def _resolve_desired_state(
        self,
        *,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        src = str(self.desired_state_source).lower()
        if src == "explicit":
            if self.desired_position_eci_km is None or self.desired_velocity_eci_km_s is None:
                return None
            return (
                np.array(self.desired_position_eci_km, dtype=float).reshape(3),
                np.array(self.desired_velocity_eci_km_s, dtype=float).reshape(3),
            )
        if self.target_id is None:
            return None
        if self.use_knowledge_for_targeting:
            kb = own_knowledge.get(self.target_id)
            if kb is not None and kb.state.size >= 6:
                return np.array(kb.state[:3], dtype=float), np.array(kb.state[3:6], dtype=float)
        tgt = world_truth.get(self.target_id)
        if tgt is None:
            return None
        return np.array(tgt.position_eci_km, dtype=float), np.array(tgt.velocity_eci_km_s, dtype=float)

    def update(
        self,
        *,
        object_id: str,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        t_s: float,
        dt_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}
        desired = self._resolve_desired_state(own_knowledge=own_knowledge, world_truth=world_truth)
        if desired is None:
            out["mission_mode"] = {"type": "end_state", "phase": "hold_no_target"}
            return out

        _, v_des = desired
        dv_eci = np.array(v_des, dtype=float) - np.array(truth.velocity_eci_km_s, dtype=float)
        dv_mag = float(np.linalg.norm(dv_eci))
        if dv_mag <= max(float(self.terminate_on_velocity_tolerance_km_s), 0.0):
            out["mission_mode"] = {"type": "end_state", "phase": "on_target"}
            return out

        cmd = IntegratedManeuverCommand(
            delta_v_eci_km_s=dv_eci,
            available_delta_v_km_s=float(max(self.available_delta_v_km_s, 0.0)),
            strategy=str(self.strategy),  # type: ignore[arg-type]
            max_thrust_n=float(max(self.max_thrust_n, 0.0)),
            dt_s=float(max(self.burn_dt_s, 1e-6)),
            min_thrust_n=float(max(self.min_thrust_n, 0.0)),
            require_attitude_alignment=(bool(self.require_attitude_alignment) and (not attitude_disabled)),
            thruster_position_body_m=None if self.thruster_position_body_m is None else np.array(self.thruster_position_body_m, dtype=float),
            thruster_direction_body=None if self.thruster_direction_body is None else np.array(self.thruster_direction_body, dtype=float),
            alignment_tolerance_rad=float(max(self.alignment_tolerance_rad, 0.0)),
        )
        _, decision = self._coordinator.execute(truth=truth, command=cmd)
        self.available_delta_v_km_s = float(max(decision.remaining_delta_v_km_s, 0.0))

        if decision.required_attitude_quat_bn is not None:
            out["desired_attitude_quat_bn"] = np.array(decision.required_attitude_quat_bn, dtype=float)

        if decision.executed and decision.applied_delta_v_km_s > 0.0:
            d = _unit(dv_eci)
            a_cmd = d * (float(decision.applied_delta_v_km_s) / float(max(self.burn_dt_s, 1e-6)))
            out["thrust_eci_km_s2"] = a_cmd

        out["mission_mode"] = {
            "type": "end_state",
            "phase": decision.action,
            "reason": decision.reason,
            "alignment_ok": bool(decision.alignment_ok),
            "remaining_delta_v_km_s": float(self.available_delta_v_km_s),
            "applied_delta_v_km_s": float(decision.applied_delta_v_km_s),
        }
        return out


@dataclass
class IntegratedCommandMissionModule:
    """
    Base mission brain for integrated orbital+attitude command arbitration.

    Workflow each step:
    1) Determine desired end state from knowledge/world/explicit input.
    2) Update orbital controller target if supported.
    3) Ask orbital controller for burn command.
    4) Check alignment for that burn.
    5) If aligned -> burn (and optional attitude hold command).
       If not aligned -> zero burn and use attitude controller to slew.
    6) Output final actuator command for this timestep.
    """

    target_id: str | None = None
    desired_state_source: str = "target"  # target|explicit
    use_knowledge_for_targeting: bool = True
    desired_position_eci_km: np.ndarray | None = None
    desired_velocity_eci_km_s: np.ndarray | None = None
    require_attitude_alignment: bool = True
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    min_burn_accel_km_s2: float = 1e-12

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(self.alignment_tolerance_rad, self.alignment_tolerance_deg)

    def _resolve_desired_state(
        self,
        *,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        src = str(self.desired_state_source).lower()
        if src == "explicit":
            if self.desired_position_eci_km is None or self.desired_velocity_eci_km_s is None:
                return None
            return (
                np.array(self.desired_position_eci_km, dtype=float).reshape(3),
                np.array(self.desired_velocity_eci_km_s, dtype=float).reshape(3),
            )
        if self.target_id is None:
            return None
        if self.use_knowledge_for_targeting:
            kb = own_knowledge.get(self.target_id)
            if kb is not None and kb.state.size >= 6:
                return np.array(kb.state[:3], dtype=float), np.array(kb.state[3:6], dtype=float)
        tgt = world_truth.get(self.target_id)
        if tgt is None:
            return None
        return np.array(tgt.position_eci_km, dtype=float), np.array(tgt.velocity_eci_km_s, dtype=float)

    @staticmethod
    def _set_orbit_controller_target(controller: Any, desired_state_eci_6: np.ndarray) -> None:
        if controller is None:
            return
        x = np.array(desired_state_eci_6, dtype=float).reshape(-1)
        if x.size != 6:
            return
        if hasattr(controller, "set_target_state"):
            try:
                controller.set_target_state(x)
                return
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set orbit target state via set_target_state: %s", exc)
        if hasattr(controller, "target_state"):
            try:
                controller.target_state = x
                return
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set orbit target state via target_state assignment: %s", exc)

    @staticmethod
    def _burn_alignment(
        *,
        truth: StateTruth,
        thrust_eci_km_s2: np.ndarray,
        thruster_direction_body: np.ndarray,
        alignment_tolerance_rad: float,
    ) -> tuple[bool, float, np.ndarray]:
        a = np.array(thrust_eci_km_s2, dtype=float).reshape(3)
        an = float(np.linalg.norm(a))
        if an <= 0.0:
            return True, 0.0, np.array(truth.attitude_quat_bn, dtype=float)
        t_body = _unit(np.array(thruster_direction_body, dtype=float))
        if np.linalg.norm(t_body) <= 0.0:
            return False, float(np.pi), np.array(truth.attitude_quat_bn, dtype=float)
        c_bn = quaternion_to_dcm_bn(truth.attitude_quat_bn)
        thrust_axis_eci = c_bn.T @ t_body
        target_axis_eci = -a / an
        cosang = float(np.clip(np.dot(thrust_axis_eci, target_axis_eci), -1.0, 1.0))
        angle = float(np.arccos(cosang))
        required_q = OrbitalAttitudeManeuverCoordinator().maneuverer.required_attitude_for_delta_v(
            truth=truth,
            delta_v_eci_km_s=a,  # direction-only usage
            thruster_direction_body=t_body,
        )
        return angle <= float(max(alignment_tolerance_rad, 0.0)), angle, required_q

    def update(
        self,
        *,
        object_id: str,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        orb_belief: StateBelief | None = None,
        att_belief: StateBelief | None = None,
        t_s: float,
        dt_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}
        desired = self._resolve_desired_state(own_knowledge=own_knowledge, world_truth=world_truth)
        if desired is not None:
            x_des = np.hstack((desired[0], desired[1]))
            self._set_orbit_controller_target(orbit_controller, x_des)

        c_orb = Command.zero()
        if orbit_controller is not None and orb_belief is not None:
            c_orb = orbit_controller.act(orb_belief, float(t_s), 2.0)
        thrust_cmd = np.array(c_orb.thrust_eci_km_s2, dtype=float).reshape(3)
        burn_mag = float(np.linalg.norm(thrust_cmd))
        burn_requested = burn_mag > float(max(self.min_burn_accel_km_s2, 0.0))

        align_ok = True
        align_angle = 0.0
        required_q = np.array(truth.attitude_quat_bn, dtype=float)
        if burn_requested and self.require_attitude_alignment and (not attitude_disabled):
            align_ok, align_angle, required_q = self._burn_alignment(
                truth=truth,
                thrust_eci_km_s2=thrust_cmd,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
                alignment_tolerance_rad=float(self.alignment_tolerance_rad),
            )

        if (not attitude_disabled) and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(np.array(required_q, dtype=float))
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set attitude target in IntegratedCommandMissionModule: %s", exc)
        c_att = Command.zero()
        if (not attitude_disabled) and attitude_controller is not None and att_belief is not None:
            c_att = attitude_controller.act(att_belief, float(t_s), 2.0)

        if burn_requested and align_ok:
            out["thrust_eci_km_s2"] = thrust_cmd
            phase = "burn"
        else:
            out["thrust_eci_km_s2"] = np.zeros(3, dtype=float)
            phase = "slew" if burn_requested else "hold"
        out["torque_body_nm"] = np.array(c_att.torque_body_nm, dtype=float).reshape(3)
        out["desired_attitude_quat_bn"] = np.array(required_q, dtype=float)
        out["mission_use_integrated_command"] = True
        out["mission_mode"] = {
            "type": "integrated_brain",
            "phase": phase,
            "burn_requested": bool(burn_requested),
            "alignment_ok": bool(align_ok),
            "alignment_angle_rad": float(align_angle),
        }
        return out


@dataclass
class PredictiveIntegratedCommandMissionModule:
    """
    Predictive integrated mission brain.

    - Predicts forward by lead time.
    - Uses orbital controller at future state to determine thrust direction.
    - Commands attitude controller toward required burn attitude.
    - Fires exactly when burn time arrives if angular tolerance is met.
    """

    target_id: str = "target"
    use_knowledge_for_targeting: bool = True
    lead_time_s: float = 30.0
    predict_dt_s: float = 1.0
    alignment_tolerance_rad: float = np.deg2rad(5.0)
    alignment_tolerance_deg: float | None = None
    thruster_direction_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    min_burn_accel_km_s2: float = 1e-12
    mu_km3_s2: float = 398600.4418
    orbit_controller_budget_ms: float = 2.0
    attitude_controller_budget_ms: float = 2.0
    planning_period_s: float | None = None
    skip_orbit_planning_in_detumble_mode: bool = True
    attitude_mode_attr: str = "mode"
    detumble_mode_tokens: tuple[str, ...] = ("detumble",)
    _countdown_s: float = field(default=-1.0, init=False, repr=False)
    _last_plan_t_s: float | None = field(default=None, init=False, repr=False)
    _planned_accel_eci_km_s2: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float), init=False, repr=False)
    _planned_attitude_quat_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float), init=False, repr=False)

    def __post_init__(self) -> None:
        self.alignment_tolerance_rad = _resolve_angle_tolerance_rad(self.alignment_tolerance_rad, self.alignment_tolerance_deg)

    def _target_state(self, own_knowledge: dict[str, StateBelief], world_truth: dict[str, StateTruth]) -> tuple[np.ndarray, np.ndarray] | None:
        if self.use_knowledge_for_targeting:
            kb = own_knowledge.get(self.target_id)
            if kb is not None and kb.state.size >= 6:
                r_k = np.array(kb.state[:3], dtype=float)
                v_k = np.array(kb.state[3:6], dtype=float)
                if np.all(np.isfinite(r_k)) and np.all(np.isfinite(v_k)):
                    return r_k, v_k
        tgt = world_truth.get(self.target_id)
        if tgt is None:
            return None
        return np.array(tgt.position_eci_km, dtype=float), np.array(tgt.velocity_eci_km_s, dtype=float)

    def _predict_eci(self, x_eci: np.ndarray, horizon_s: float, dt_s: float) -> np.ndarray:
        x = np.array(x_eci, dtype=float).reshape(6)
        n_steps = int(max(np.floor(horizon_s / dt_s), 0))
        rem = float(max(horizon_s - n_steps * dt_s, 0.0))
        for _ in range(n_steps):
            x = propagate_two_body_rk4(x_eci=x, dt_s=dt_s, mu_km3_s2=self.mu_km3_s2, accel_cmd_eci_km_s2=np.zeros(3))
        if rem > 1e-9:
            x = propagate_two_body_rk4(x_eci=x, dt_s=rem, mu_km3_s2=self.mu_km3_s2, accel_cmd_eci_km_s2=np.zeros(3))
        return x

    def _predict_orb_belief_for_controller(
        self,
        orbit_controller: Any | None,
        self_truth: StateTruth,
        target_state_eci: tuple[np.ndarray, np.ndarray] | None,
    ) -> StateBelief:
        x_self = np.hstack((np.array(self_truth.position_eci_km, dtype=float), np.array(self_truth.velocity_eci_km_s, dtype=float)))
        horizon = float(max(self.lead_time_s, 0.0))
        hdt = float(max(min(self.predict_dt_s, max(horizon, 1e-6)), 1e-6))
        x_self_p = self._predict_eci(x_self, horizon_s=horizon, dt_s=hdt)
        if target_state_eci is None:
            return StateBelief(state=x_self_p, covariance=np.eye(6) * 1e-4, last_update_t_s=float(self_truth.t_s))
        x_tgt = np.hstack((target_state_eci[0], target_state_eci[1]))
        x_tgt_p = self._predict_eci(x_tgt, horizon_s=horizon, dt_s=hdt)

        if orbit_controller is not None and hasattr(orbit_controller, "ric_curv_state_slice"):
            r_c = x_tgt_p[:3]
            v_c = x_tgt_p[3:6]
            r_s = x_self_p[:3]
            v_s = x_self_p[3:6]
            x_rect = eci_relative_to_ric_rect(
                x_dep_eci=np.hstack((r_s, v_s)),
                x_chief_eci=np.hstack((r_c, v_c)),
            )
            x_curv = ric_rect_to_curv(x_rect, r0_km=float(np.linalg.norm(r_c)))
            x = np.hstack((x_curv, np.hstack((r_c, v_c))))
            return StateBelief(state=x, covariance=np.eye(12) * 1e-4, last_update_t_s=float(self_truth.t_s))
        return StateBelief(state=x_self_p, covariance=np.eye(6) * 1e-4, last_update_t_s=float(self_truth.t_s))

    def _alignment(self, truth: StateTruth, accel_eci_km_s2: np.ndarray) -> tuple[bool, float]:
        a = np.array(accel_eci_km_s2, dtype=float).reshape(3)
        if float(np.linalg.norm(a)) <= 0.0:
            return True, 0.0
        c_bn = quaternion_to_dcm_bn(truth.attitude_quat_bn)
        t_body = _unit(np.array(self.thruster_direction_body, dtype=float))
        if float(np.linalg.norm(t_body)) <= 0.0:
            return False, float(np.pi)
        thrust_axis_eci = c_bn.T @ t_body
        target_axis_eci = -_unit(a)
        cosang = float(np.clip(np.dot(thrust_axis_eci, target_axis_eci), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        return ang <= float(max(self.alignment_tolerance_rad, 0.0)), ang

    def _attitude_controller_in_detumble_mode(self, attitude_controller: Any | None) -> tuple[bool, str]:
        if attitude_controller is None:
            return False, ""
        attr = str(self.attitude_mode_attr).strip()
        if not attr:
            return False, ""
        try:
            mode_obj = getattr(attitude_controller, attr, "")
        except AttributeError:
            mode_obj = ""
        mode_str = str(mode_obj).strip().lower()
        if not mode_str:
            return False, ""
        tokens = [str(t).strip().lower() for t in tuple(self.detumble_mode_tokens or ()) if str(t).strip()]
        if any(tok in mode_str for tok in tokens):
            return True, mode_str
        return False, mode_str

    def _effective_planning_period_s(self, orbit_controller: Any | None, dt_s: float) -> float:
        if self.planning_period_s is not None:
            return float(max(self.planning_period_s, 1e-9))
        # Default to mission update cadence; explicit planning_period_s can be used
        # when slower planning is desired.
        return float(max(dt_s, 1e-9))

    def update(
        self,
        *,
        truth: StateTruth,
        own_knowledge: dict[str, StateBelief],
        world_truth: dict[str, StateTruth],
        orbit_controller: Any | None = None,
        attitude_controller: Any | None = None,
        orb_belief: StateBelief | None = None,
        att_belief: StateBelief | None = None,
        t_s: float,
        dt_s: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        env = dict(kwargs.get("env", {}) or {})
        attitude_disabled = bool(env.get("attitude_disabled", False))
        out: dict[str, Any] = {}
        in_detumble_mode, mode_str = self._attitude_controller_in_detumble_mode(attitude_controller)
        planning_blocked_by_detumble = bool(self.skip_orbit_planning_in_detumble_mode and in_detumble_mode)
        plan_period_s = self._effective_planning_period_s(orbit_controller=orbit_controller, dt_s=float(dt_s))
        target_state = self._target_state(own_knowledge=own_knowledge, world_truth=world_truth)
        plan_due = bool(
            self._last_plan_t_s is None
            or (float(t_s) - float(self._last_plan_t_s)) >= (plan_period_s - 1e-12)
        )
        planned_this_step = False

        if planning_blocked_by_detumble:
            # Cancel pending planned burn while detumbling.
            self._countdown_s = -1.0
            self._planned_accel_eci_km_s2 = np.zeros(3, dtype=float)
            self._planned_attitude_quat_bn = np.array(truth.attitude_quat_bn, dtype=float)
        elif self._countdown_s < 0.0 and plan_due:
            b_pred = self._predict_orb_belief_for_controller(
                orbit_controller=orbit_controller,
                self_truth=truth,
                target_state_eci=target_state,
            )
            c_orb_pred = (
                orbit_controller.act(b_pred, float(t_s), float(max(self.orbit_controller_budget_ms, 1e-9)))
                if orbit_controller is not None
                else Command.zero()
            )
            self._planned_accel_eci_km_s2 = np.array(c_orb_pred.thrust_eci_km_s2, dtype=float).reshape(3)
            if not np.all(np.isfinite(self._planned_accel_eci_km_s2)):
                self._planned_accel_eci_km_s2 = np.zeros(3, dtype=float)
            dv_pred = self._planned_accel_eci_km_s2 * float(max(self.predict_dt_s, 1e-6))
            q_req = OrbitalAttitudeManeuverCoordinator().maneuverer.required_attitude_for_delta_v(
                truth=truth,
                delta_v_eci_km_s=dv_pred,
                thruster_direction_body=np.array(self.thruster_direction_body, dtype=float),
            )
            self._planned_attitude_quat_bn = np.array(q_req if q_req is not None else truth.attitude_quat_bn, dtype=float)
            self._countdown_s = float(max(self.lead_time_s, 0.0))
            self._last_plan_t_s = float(t_s)
            planned_this_step = True

        # Slew/hold phase before gate
        if (not attitude_disabled) and attitude_controller is not None and hasattr(attitude_controller, "set_target"):
            try:
                attitude_controller.set_target(np.array(self._planned_attitude_quat_bn, dtype=float))
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Failed to set predictive attitude target: %s", exc)
        att_belief_eff = att_belief
        if (not attitude_disabled) and att_belief_eff is None and attitude_controller is not None:
            # Ensure integrated attitude logic can still run even if self-knowledge is not configured.
            att_belief_eff = StateBelief(
                state=np.hstack((np.array(truth.attitude_quat_bn, dtype=float), np.array(truth.angular_rate_body_rad_s, dtype=float))),
                covariance=np.eye(7) * 1e-6,
                last_update_t_s=float(truth.t_s),
            )
        c_att = (
            attitude_controller.act(att_belief_eff, float(t_s), float(max(self.attitude_controller_budget_ms, 1e-9)))
            if (not attitude_disabled) and attitude_controller is not None and att_belief_eff is not None
            else Command.zero()
        )

        fire = False
        align_ok, align_angle = self._alignment(truth=truth, accel_eci_km_s2=self._planned_accel_eci_km_s2)
        if attitude_disabled:
            align_ok = True
            align_angle = 0.0
        if planning_blocked_by_detumble:
            fire = False
        else:
            if float(max(self.lead_time_s, 0.0)) <= 0.0:
                # Zero-lead mode: continuous closed-loop burn eligibility each step.
                fire = bool(
                    align_ok
                    and float(np.linalg.norm(self._planned_accel_eci_km_s2)) > float(max(self.min_burn_accel_km_s2, 0.0))
                )
                self._countdown_s = 0.0
            elif self._countdown_s < 0.0:
                fire = False
            elif self._countdown_s <= float(max(dt_s, 1e-9)):
                if align_ok and float(np.linalg.norm(self._planned_accel_eci_km_s2)) > float(max(self.min_burn_accel_km_s2, 0.0)):
                    fire = True
                self._countdown_s = -1.0
            else:
                self._countdown_s -= float(max(dt_s, 1e-9))

        out["mission_use_integrated_command"] = True
        out["torque_body_nm"] = np.array(c_att.torque_body_nm, dtype=float).reshape(3)
        out["command_mode_flags"] = dict(c_att.mode_flags or {})
        out["desired_attitude_quat_bn"] = np.array(self._planned_attitude_quat_bn, dtype=float)
        out["thrust_eci_km_s2"] = self._planned_accel_eci_km_s2.copy() if fire else np.zeros(3, dtype=float)
        out["mission_mode"] = {
            "type": "predictive_integrated_brain",
            "countdown_s": float(self._countdown_s),
            "fire": bool(fire),
            "alignment_ok": bool(align_ok),
            "alignment_angle_rad": float(align_angle),
            "orbit_controller_budget_ms": float(self.orbit_controller_budget_ms),
            "attitude_controller_budget_ms": float(self.attitude_controller_budget_ms),
            "planning_blocked_by_detumble": bool(planning_blocked_by_detumble),
            "attitude_controller_mode": str(mode_str),
            "plan_due": bool(plan_due),
            "planned_this_step": bool(planned_this_step),
            "planning_period_s": float(plan_period_s),
        }
        return out
