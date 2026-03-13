from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.control.attitude.pose_commands import PoseCommandGenerator
from sim.core.models import StateBelief, StateTruth
from sim.rocket.models import RocketState, RocketVehicleConfig
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.quaternion import dcm_to_quaternion_bn, normalize_quaternion, quaternion_to_dcm_bn


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
