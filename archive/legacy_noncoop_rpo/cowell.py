from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .atmosphere import EARTH_ROT_RATE_RAD_S, atmos76_density_from_eci
from .constants import MU_EARTH_KM3_S2


PerturbationFn = Callable[[float, np.ndarray], np.ndarray]
BurnFn = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DragConfig:
    cd: float
    area_m2: float
    mass_kg: float
    omega_earth_rad_s: float = EARTH_ROT_RATE_RAD_S

    @property
    def area_km2(self) -> float:
        return self.area_m2 * 1e-6


@dataclass
class CowellResult:
    t_s: np.ndarray
    x_eci: np.ndarray
    a_pert_total_eci: np.ndarray
    a_burn_eci: np.ndarray
    terminated_early: bool = False
    termination_reason: str = ""


def _ensure_accel(vec: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{label} must return a 3-vector acceleration in km/s^2.")
    return arr


def rk4_step_time(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    x: np.ndarray,
    dt: float,
) -> np.ndarray:
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def cowell_total_accel_eci(
    t_s: float,
    x_eci: np.ndarray,
    perturbations: Optional[Sequence[PerturbationFn]] = None,
    burn_fn: Optional[BurnFn] = None,
) -> tuple[np.ndarray, np.ndarray]:
    a_pert_total = np.zeros(3, dtype=float)
    if perturbations is not None:
        for fn in perturbations:
            a_pert_total += _ensure_accel(fn(t_s, x_eci), "Perturbation")

    a_burn = np.zeros(3, dtype=float)
    if burn_fn is not None:
        a_burn = _ensure_accel(burn_fn(t_s, x_eci), "Burn")

    return a_pert_total, a_burn


def cowell_deriv(
    t_s: float,
    x_eci: np.ndarray,
    mu: float = MU_EARTH_KM3_S2,
    perturbations: Optional[Sequence[PerturbationFn]] = None,
    burn_fn: Optional[BurnFn] = None,
) -> np.ndarray:
    r = x_eci[0:3]
    v = x_eci[3:6]
    r_norm = np.linalg.norm(r)
    a_grav = -mu * r / (r_norm**3)
    a_pert, a_burn = cowell_total_accel_eci(t_s, x_eci, perturbations=perturbations, burn_fn=burn_fn)
    a_total = a_grav + a_pert + a_burn
    return np.hstack((v, a_total))


def propagate_cowell(
    x0_eci: np.ndarray,
    dt_s: float,
    steps: int,
    t0_s: float = 0.0,
    mu: float = MU_EARTH_KM3_S2,
    perturbations: Optional[Sequence[PerturbationFn]] = None,
    burn_fn: Optional[BurnFn] = None,
    terminate_below_radius_km: Optional[float] = None,
) -> CowellResult:
    x0 = np.asarray(x0_eci, dtype=float)
    if x0.shape != (6,):
        raise ValueError("x0_eci must be a 6-vector [rx,ry,rz,vx,vy,vz].")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive.")
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    if terminate_below_radius_km is not None and terminate_below_radius_km <= 0.0:
        raise ValueError("terminate_below_radius_km must be positive when provided.")

    n = steps + 1
    t_log = np.zeros(n, dtype=float)
    x_log = np.zeros((n, 6), dtype=float)
    a_pert_log = np.zeros((n, 3), dtype=float)
    a_burn_log = np.zeros((n, 3), dtype=float)

    x = x0.copy()
    t = float(t0_s)
    terminated_early = False
    termination_reason = ""
    t_log[0] = t
    x_log[0, :] = x
    a_pert_log[0, :], a_burn_log[0, :] = cowell_total_accel_eci(t, x, perturbations=perturbations, burn_fn=burn_fn)

    if terminate_below_radius_km is not None:
        if np.linalg.norm(x[0:3]) <= terminate_below_radius_km:
            terminated_early = True
            termination_reason = "radius_below_termination_threshold"
            return CowellResult(
                t_s=t_log[:1].copy(),
                x_eci=x_log[:1, :].copy(),
                a_pert_total_eci=a_pert_log[:1, :].copy(),
                a_burn_eci=a_burn_log[:1, :].copy(),
                terminated_early=terminated_early,
                termination_reason=termination_reason,
            )

    def dyn(local_t: float, local_x: np.ndarray) -> np.ndarray:
        return cowell_deriv(local_t, local_x, mu=mu, perturbations=perturbations, burn_fn=burn_fn)

    for k in range(steps):
        x = rk4_step_time(dyn, t, x, dt_s)
        t += dt_s
        t_log[k + 1] = t
        x_log[k + 1, :] = x
        a_pert_log[k + 1, :], a_burn_log[k + 1, :] = cowell_total_accel_eci(
            t, x, perturbations=perturbations, burn_fn=burn_fn
        )
        if terminate_below_radius_km is not None and np.linalg.norm(x[0:3]) <= terminate_below_radius_km:
            terminated_early = True
            termination_reason = "radius_below_termination_threshold"
            end = k + 2
            return CowellResult(
                t_s=t_log[:end].copy(),
                x_eci=x_log[:end, :].copy(),
                a_pert_total_eci=a_pert_log[:end, :].copy(),
                a_burn_eci=a_burn_log[:end, :].copy(),
                terminated_early=terminated_early,
                termination_reason=termination_reason,
            )

    return CowellResult(
        t_s=t_log,
        x_eci=x_log,
        a_pert_total_eci=a_pert_log,
        a_burn_eci=a_burn_log,
        terminated_early=terminated_early,
        termination_reason=termination_reason,
    )


def make_drag_perturbation(
    config: DragConfig,
    density_model: Callable[[np.ndarray], float] = atmos76_density_from_eci,
) -> PerturbationFn:
    if config.mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive.")
    if config.area_m2 < 0.0:
        raise ValueError("area_m2 must be non-negative.")
    if config.cd < 0.0:
        raise ValueError("cd must be non-negative.")

    omega_vec = np.array([0.0, 0.0, config.omega_earth_rad_s], dtype=float)
    area_over_mass_km2_per_kg = config.area_km2 / config.mass_kg

    def drag_accel_eci(_t_s: float, x_eci: np.ndarray) -> np.ndarray:
        r = x_eci[0:3]
        v = x_eci[3:6]

        rho = density_model(r)
        if rho <= 0.0:
            return np.zeros(3, dtype=float)

        v_atm = np.cross(omega_vec, r)
        v_rel = v - v_atm
        speed = np.linalg.norm(v_rel)
        if speed <= 0.0:
            return np.zeros(3, dtype=float)

        return -0.5 * config.cd * area_over_mass_km2_per_kg * rho * speed * v_rel

    return drag_accel_eci


def make_constant_burn(
    accel_eci_km_s2: np.ndarray,
    start_s: float,
    stop_s: float,
) -> BurnFn:
    accel_cmd = np.asarray(accel_eci_km_s2, dtype=float)
    if accel_cmd.shape != (3,):
        raise ValueError("accel_eci_km_s2 must be a 3-vector.")
    if stop_s < start_s:
        raise ValueError("stop_s must be >= start_s.")

    def burn_fn(t_s: float, _x_eci: np.ndarray) -> np.ndarray:
        if start_s <= t_s <= stop_s:
            return accel_cmd
        return np.zeros(3, dtype=float)

    return burn_fn
