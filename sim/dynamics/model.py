from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import DynamicsModel
from sim.core.models import Command, StateTruth
from sim.dynamics.attitude.disturbances import DisturbanceTorqueModel
from sim.dynamics.attitude.rigid_body import propagate_attitude_exponential_map
from sim.dynamics.orbit.atmosphere import density_from_model
from sim.dynamics.orbit.eclipse import srp_shadow_factor
from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.dynamics.spacecraft_geometry import RectangularPrismGeometry
from sim.utils.quaternion import normalize_quaternion, quaternion_to_dcm_bn


@dataclass(frozen=True)
class OrbitalAttitudeDynamics(DynamicsModel):
    mu_km3_s2: float
    inertia_kg_m2: np.ndarray
    disturbance_model: DisturbanceTorqueModel | None = None
    area_m2: float = 1.0
    cd: float = 2.2
    cr: float = 1.2
    use_rectangular_prism_for_aero_srp: bool = False
    rectangular_prism_dims_m: tuple[float, float, float] | None = None
    orbit_substep_s: float | None = None
    attitude_substep_s: float | None = None
    propagate_attitude: bool = True
    orbit_propagator: OrbitPropagator = field(default_factory=lambda: OrbitPropagator(integrator="rk4"))

    def __post_init__(self) -> None:
        if self.use_rectangular_prism_for_aero_srp:
            if self.rectangular_prism_dims_m is None:
                raise ValueError(
                    "rectangular_prism_dims_m must be provided when use_rectangular_prism_for_aero_srp=True."
                )
            if self.disturbance_model is None:
                raise ValueError(
                    "Rectangular prism aero/SRP mode requires coupled orbit+attitude disturbance simulation "
                    "(disturbance_model must be set)."
                )

    def step(self, state: StateTruth, command: Command, env: dict, dt_s: float) -> StateTruth:
        env_local = dict(env)
        geom = self._rectangular_prism_geometry()
        if self.use_rectangular_prism_for_aero_srp and geom is not None and self.disturbance_model is not None:
            c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn)
            omega_earth = np.array([0.0, 0.0, EARTH_ROT_RATE_RAD_S], dtype=float)
            v_atm_eci_km_s = np.array(
                [
                    -EARTH_ROT_RATE_RAD_S * float(state.position_eci_km[1]),
                    EARTH_ROT_RATE_RAD_S * float(state.position_eci_km[0]),
                    0.0,
                ],
                dtype=float,
            )
            v_rel_eci_km_s = state.velocity_eci_km_s - v_atm_eci_km_s
            v_rel_body = c_bn @ v_rel_eci_km_s
            env_local["drag_area_m2"] = geom.projected_area_m2(-v_rel_body)

            sun_dir_eci = np.array(env_local.get("sun_dir_eci", np.array([1.0, 0.0, 0.0])), dtype=float)
            s_norm = float(np.linalg.norm(sun_dir_eci))
            if s_norm > 0.0:
                sun_dir_body = c_bn @ (sun_dir_eci / s_norm)
                env_local["srp_area_m2"] = geom.projected_area_m2(-sun_dir_body)

        x_orbit = np.hstack((state.position_eci_km, state.velocity_eci_km_s))
        orbit_ctx = OrbitContext(
            mu_km3_s2=self.mu_km3_s2,
            mass_kg=state.mass_kg,
            area_m2=self.area_m2,
            cd=self.cd,
            cr=self.cr,
        )
        orbit_dt = self._effective_substep(self.orbit_substep_s, dt_s)
        x_orbit_next = x_orbit.copy()
        t_local = state.t_s
        for h in self._substep_sequence(dt_s, orbit_dt):
            x_orbit_next = self.orbit_propagator.propagate(
                x_eci=x_orbit_next,
                dt_s=h,
                t_s=t_local,
                command_accel_eci_km_s2=command.thrust_eci_km_s2,
                env=env_local,
                ctx=orbit_ctx,
            )
            t_local += h

        q_next = state.attitude_quat_bn.copy()
        w_next = state.angular_rate_body_rad_s.copy()
        if self.propagate_attitude:
            disturbance_cfg = getattr(self.disturbance_model, "config", None)
            if self.disturbance_model is not None and bool(getattr(disturbance_cfg, "use_drag", False)):
                if "density_kg_m3" not in env_local:
                    env_local["density_kg_m3"] = density_from_model(
                        str(env_local.get("atmosphere_model", "exponential")).lower(),
                        state.position_eci_km,
                        state.t_s,
                        env=env_local,
                    )
                v_atm_eci_km_s = np.array(
                    [
                        -EARTH_ROT_RATE_RAD_S * float(state.position_eci_km[1]),
                        EARTH_ROT_RATE_RAD_S * float(state.position_eci_km[0]),
                        0.0,
                    ],
                    dtype=float,
                )
                v_rel_eci_m_s = (state.velocity_eci_km_s - v_atm_eci_km_s) * 1e3
                env_local["drag_v_rel_eci_m_s"] = v_rel_eci_m_s
                env_local["drag_v_rel_norm_m_s"] = float(np.linalg.norm(v_rel_eci_m_s))

            if self.disturbance_model is not None and bool(getattr(disturbance_cfg, "use_srp", False)):
                sun_dir_eci = env_local.get("sun_dir_eci")
                if sun_dir_eci is not None:
                    sun_dir_eci = np.asarray(sun_dir_eci, dtype=float).reshape(3)
                    sun_norm = float(np.linalg.norm(sun_dir_eci))
                    if sun_norm > 0.0:
                        env_local["sun_dir_eci_unit"] = sun_dir_eci / sun_norm
                env_local["srp_shadow_factor"] = srp_shadow_factor(
                    r_sc_eci_km=state.position_eci_km,
                    t_s=state.t_s,
                    env=env_local,
                )
            att_dt = self._effective_substep(self.attitude_substep_s, dt_s)
            t_att = state.t_s
            for h in self._substep_sequence(dt_s, att_dt):
                att_state = StateTruth(
                    position_eci_km=np.array(state.position_eci_km, dtype=float),
                    velocity_eci_km_s=np.array(state.velocity_eci_km_s, dtype=float),
                    attitude_quat_bn=np.array(q_next, dtype=float),
                    angular_rate_body_rad_s=np.array(w_next, dtype=float),
                    mass_kg=float(state.mass_kg),
                    t_s=float(t_att),
                )
                disturbance_torque = (
                    np.zeros(3) if self.disturbance_model is None else self.disturbance_model.total_torque_body_nm(att_state, env_local)
                )
                total_torque = command.torque_body_nm + disturbance_torque
                q_next, w_next = propagate_attitude_exponential_map(
                    quat_bn=q_next,
                    omega_body_rad_s=w_next,
                    inertia_kg_m2=self.inertia_kg_m2,
                    torque_body_nm=total_torque,
                    dt_s=h,
                )
                t_att += h

        # Optional direct attitude state override for surrogate controller testing.
        if self.propagate_attitude:
            att_override = dict(command.mode_flags.get("attitude_state_override", {}) or {})
            if att_override:
                q_cmd = np.array(att_override.get("q_next_bn", q_next), dtype=float).reshape(-1)
                w_cmd = np.array(att_override.get("w_next_body_rad_s", w_next), dtype=float).reshape(-1)
                if q_cmd.size == 4:
                    q_next = normalize_quaternion(q_cmd)
                if w_cmd.size == 3:
                    w_next = w_cmd
        delta_mass_kg = float(command.mode_flags.get("delta_mass_kg", 0.0))
        min_mass_kg = float(command.mode_flags.get("min_mass_kg", 0.0))
        if not np.isfinite(min_mass_kg):
            min_mass_kg = 0.0
        min_mass_kg = max(min_mass_kg, 0.0)
        mass_next = max(min_mass_kg, state.mass_kg - delta_mass_kg)

        return StateTruth(
            position_eci_km=x_orbit_next[:3],
            velocity_eci_km_s=x_orbit_next[3:],
            attitude_quat_bn=q_next,
            angular_rate_body_rad_s=w_next,
            mass_kg=mass_next,
            t_s=state.t_s + dt_s,
        )

    def _rectangular_prism_geometry(self) -> RectangularPrismGeometry | None:
        dims = self.rectangular_prism_dims_m
        if dims is None:
            return None
        return RectangularPrismGeometry(lx_m=float(dims[0]), ly_m=float(dims[1]), lz_m=float(dims[2]))

    @staticmethod
    def _effective_substep(substep_s: float | None, dt_s: float) -> float:
        if substep_s is None:
            return dt_s
        return max(min(float(substep_s), dt_s), 1e-9)

    @staticmethod
    def _substep_sequence(total_dt_s: float, h_s: float) -> list[float]:
        if h_s >= total_dt_s:
            return [float(total_dt_s)]
        n = int(np.floor(total_dt_s / h_s))
        steps = [float(h_s)] * n
        rem = float(total_dt_s - n * h_s)
        if rem > 1e-12:
            steps.append(rem)
        return steps
