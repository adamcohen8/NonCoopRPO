from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


@dataclass(frozen=True)
class RocketAeroConfig:
    enabled: bool = True
    reference_area_m2: float = 10.0
    reference_length_m: float = 30.0
    cp_offset_body_m: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    cd_base: float = 0.20
    cd_alpha2: float = 0.10
    cd_supersonic: float = 0.28
    transonic_peak_cd: float = 0.22
    transonic_mach: float = 1.0
    transonic_width: float = 0.22
    cl_alpha_per_rad: float = 0.15
    cy_beta_per_rad: float = 0.15
    cm_alpha_per_rad: float = -0.02
    cn_beta_per_rad: float = -0.02
    cl_roll_per_rad: float = -0.01
    alpha_limit_deg: float = 20.0
    beta_limit_deg: float = 20.0


@dataclass(frozen=True)
class RocketAeroState:
    rho_kg_m3: float
    pressure_pa: float
    temperature_k: float
    sound_speed_m_s: float
    dynamic_pressure_pa: float
    speed_m_s: float
    mach: float
    alpha_rad: float
    beta_rad: float


@dataclass(frozen=True)
class RocketAeroLoads:
    force_body_n: np.ndarray
    moment_body_nm: np.ndarray
    coeff_force_body: np.ndarray
    coeff_moment_body: np.ndarray
    state: RocketAeroState


def compute_aero_state(
    *,
    rho_kg_m3: float,
    pressure_pa: float,
    temperature_k: float,
    sound_speed_m_s: float,
    v_rel_body_m_s: np.ndarray,
    alpha_limit_deg: float,
    beta_limit_deg: float,
) -> RocketAeroState:
    v = np.array(v_rel_body_m_s, dtype=float).reshape(3)
    speed = float(np.linalg.norm(v))
    q = 0.5 * float(rho_kg_m3) * speed * speed
    u, v_lat, w = float(v[0]), float(v[1]), float(v[2])
    if speed <= 1e-12:
        alpha = 0.0
        beta = 0.0
        mach = 0.0
    else:
        alpha = float(np.arctan2(w, max(u, 1e-12)))
        beta = float(np.arcsin(np.clip(v_lat / speed, -1.0, 1.0)))
        mach = speed / max(float(sound_speed_m_s), 1e-6)
    alpha_lim = np.deg2rad(float(alpha_limit_deg))
    beta_lim = np.deg2rad(float(beta_limit_deg))
    alpha = float(np.clip(alpha, -alpha_lim, alpha_lim))
    beta = float(np.clip(beta, -beta_lim, beta_lim))
    return RocketAeroState(
        rho_kg_m3=float(max(rho_kg_m3, 0.0)),
        pressure_pa=float(max(pressure_pa, 0.0)),
        temperature_k=float(max(temperature_k, 1.0)),
        sound_speed_m_s=float(max(sound_speed_m_s, 1e-3)),
        dynamic_pressure_pa=float(max(q, 0.0)),
        speed_m_s=float(max(speed, 0.0)),
        mach=float(max(mach, 0.0)),
        alpha_rad=alpha,
        beta_rad=beta,
    )


def _cd_model(mach: float, alpha_rad: float, cfg: RocketAeroConfig) -> float:
    m = float(max(mach, 0.0))
    cd_m = cfg.cd_base if m <= 1.0 else (cfg.cd_base + (cfg.cd_supersonic - cfg.cd_base) * (1.0 - np.exp(-(m - 1.0))))
    w = max(float(cfg.transonic_width), 1e-3)
    bump = float(cfg.transonic_peak_cd) * np.exp(-0.5 * ((m - float(cfg.transonic_mach)) / w) ** 2)
    cd_alpha = float(cfg.cd_alpha2) * float(alpha_rad) * float(alpha_rad)
    return float(max(cd_m + bump + cd_alpha, 0.0))


def compute_aero_loads(
    v_rel_body_m_s: np.ndarray,
    atmos: RocketAeroState,
    cfg: RocketAeroConfig,
) -> RocketAeroLoads:
    q = atmos.dynamic_pressure_pa
    if q <= 0.0 or not cfg.enabled:
        z3 = np.zeros(3)
        return RocketAeroLoads(
            force_body_n=z3.copy(),
            moment_body_nm=z3.copy(),
            coeff_force_body=z3.copy(),
            coeff_moment_body=z3.copy(),
            state=atmos,
        )

    cd = _cd_model(atmos.mach, atmos.alpha_rad, cfg)
    cl = float(cfg.cl_alpha_per_rad) * atmos.alpha_rad
    cy = float(cfg.cy_beta_per_rad) * atmos.beta_rad
    c_force = np.array([-cd, cy, -cl], dtype=float)
    f_body = q * float(cfg.reference_area_m2) * c_force

    cm = float(cfg.cm_alpha_per_rad) * atmos.alpha_rad
    cn = float(cfg.cn_beta_per_rad) * atmos.beta_rad
    cl_roll = float(cfg.cl_roll_per_rad) * atmos.beta_rad
    c_moment = np.array([cl_roll, cm, cn], dtype=float)
    m_coeff = q * float(cfg.reference_area_m2) * float(cfg.reference_length_m) * c_moment
    r_cp_cg = np.array(cfg.cp_offset_body_m, dtype=float).reshape(3)
    m_offset = np.cross(r_cp_cg, f_body)
    m_body = m_coeff + m_offset

    return RocketAeroLoads(
        force_body_n=f_body,
        moment_body_nm=m_body,
        coeff_force_body=c_force,
        coeff_moment_body=c_moment,
        state=atmos,
    )
