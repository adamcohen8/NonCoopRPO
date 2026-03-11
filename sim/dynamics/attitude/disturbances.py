from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.models import StateTruth
from sim.dynamics.orbit.atmosphere import density_from_model
from sim.dynamics.orbit.eclipse import srp_shadow_factor
from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.spacecraft_geometry import RectangularPrismGeometry
from sim.utils.quaternion import quaternion_to_dcm_bn

EARTH_MAGNETIC_DIPOLE_T_M3 = 7.94e15
MU_0_4PI = 1e-7
SOLAR_PRESSURE_N_M2 = 4.56e-6


@dataclass(frozen=True)
class DisturbanceTorqueConfig:
    use_gravity_gradient: bool = True
    use_magnetic: bool = True
    use_drag: bool = True
    use_srp: bool = True
    magnetic_dipole_body_a_m2: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.0, 0.0]))
    drag_area_m2: float = 1.5
    drag_cd: float = 2.2
    drag_cp_offset_body_m: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.02, -0.01]))
    srp_area_m2: float = 1.0
    srp_cr: float = 1.3
    srp_cp_offset_body_m: np.ndarray = field(default_factory=lambda: np.array([-0.02, 0.03, 0.01]))
    sun_dir_eci: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    use_rectangular_prism_faces: bool = False
    rectangular_prism_dims_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class DisturbanceTorqueModel:
    mu_km3_s2: float
    inertia_kg_m2: np.ndarray
    config: DisturbanceTorqueConfig = field(default_factory=DisturbanceTorqueConfig)

    def total_torque_body_nm(self, state: StateTruth, env: dict | None = None) -> np.ndarray:
        env = env or {}
        tau = np.zeros(3)

        if self.config.use_gravity_gradient:
            tau += self._gravity_gradient_torque(state)
        if self.config.use_magnetic:
            tau += self._magnetic_torque(state)
        if self.config.use_drag:
            tau += self._drag_torque(state, env)
        if self.config.use_srp:
            tau += self._srp_torque(state, env)

        return tau

    def _gravity_gradient_torque(self, state: StateTruth) -> np.ndarray:
        r_i_m = state.position_eci_km * 1e3
        r_norm_m = np.linalg.norm(r_i_m)
        if r_norm_m == 0.0:
            return np.zeros(3)
        c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn)
        r_hat_b = c_bn @ (r_i_m / r_norm_m)
        mu_m3_s2 = self.mu_km3_s2 * 1e9
        return 3.0 * mu_m3_s2 / (r_norm_m**3) * np.cross(r_hat_b, self.inertia_kg_m2 @ r_hat_b)

    def _magnetic_torque(self, state: StateTruth) -> np.ndarray:
        r_i_m = state.position_eci_km * 1e3
        r_norm_m = np.linalg.norm(r_i_m)
        if r_norm_m == 0.0:
            return np.zeros(3)

        m_eci = np.array([0.0, 0.0, EARTH_MAGNETIC_DIPOLE_T_M3])
        r_hat = r_i_m / r_norm_m
        b_eci = MU_0_4PI * (3.0 * r_hat * np.dot(m_eci, r_hat) - m_eci) / (r_norm_m**3)
        c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn)
        b_body = c_bn @ b_eci
        return np.cross(self.config.magnetic_dipole_body_a_m2, b_body)

    def _drag_torque(self, state: StateTruth, env: dict) -> np.ndarray:
        if "density_kg_m3" in env:
            rho = float(env["density_kg_m3"])
        else:
            rho = density_from_model(
                str(env.get("atmosphere_model", "exponential")).lower(),
                state.position_eci_km,
                state.t_s,
                env=env,
            )
        omega_earth = np.array([0.0, 0.0, EARTH_ROT_RATE_RAD_S], dtype=float)
        v_atm_eci_km_s = np.cross(omega_earth, state.position_eci_km)
        v_rel_eci_m_s = (state.velocity_eci_km_s - v_atm_eci_km_s) * 1e3
        v_norm = np.linalg.norm(v_rel_eci_m_s)
        if v_norm == 0.0 or rho <= 0.0:
            return np.zeros(3)

        c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn)
        v_rel_body = c_bn @ v_rel_eci_m_s
        if self._rect_prism_geometry is not None and self.config.use_rectangular_prism_faces:
            q_dyn = 0.5 * rho * (v_norm**2) * self.config.drag_cd
            return self._rect_prism_geometry.face_torque_sum_body_nm(v_rel_body, q_dyn)

        f_drag_mag = 0.5 * rho * (v_norm**2) * self.config.drag_cd * self.config.drag_area_m2
        f_drag_body = -f_drag_mag * (v_rel_body / v_norm)
        return np.cross(self.config.drag_cp_offset_body_m, f_drag_body)

    def _srp_torque(self, state: StateTruth, env: dict) -> np.ndarray:
        sun_dir_eci = np.array(env.get("sun_dir_eci", self.config.sun_dir_eci), dtype=float)
        n = np.linalg.norm(sun_dir_eci)
        if n == 0.0:
            return np.zeros(3)
        sun_dir_eci = sun_dir_eci / n
        shadow = srp_shadow_factor(r_sc_eci_km=state.position_eci_km, t_s=state.t_s, env=env)
        if shadow <= 0.0:
            return np.zeros(3)

        c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn)
        sun_dir_body = c_bn @ sun_dir_eci
        if self._rect_prism_geometry is not None and self.config.use_rectangular_prism_faces:
            p_srp = SOLAR_PRESSURE_N_M2 * self.config.srp_cr * shadow
            return self._rect_prism_geometry.face_torque_sum_body_nm(sun_dir_body, p_srp)

        force_mag = SOLAR_PRESSURE_N_M2 * self.config.srp_cr * self.config.srp_area_m2 * shadow
        f_srp_body = -force_mag * sun_dir_body
        return np.cross(self.config.srp_cp_offset_body_m, f_srp_body)

    @property
    def _rect_prism_geometry(self) -> RectangularPrismGeometry | None:
        dims = self.config.rectangular_prism_dims_m
        if dims is None:
            return None
        return RectangularPrismGeometry(lx_m=float(dims[0]), ly_m=float(dims[1]), lz_m=float(dims[2]))
