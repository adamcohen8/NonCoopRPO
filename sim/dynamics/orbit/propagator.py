from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from sim.dynamics.orbit.accelerations import (
    OrbitContext,
    accel_drag,
    accel_j2,
    accel_j3,
    accel_j4,
    accel_srp,
    accel_third_body,
    accel_two_body,
)
from sim.dynamics.orbit.atmosphere import density_from_model
from sim.dynamics.orbit.environment import MOON_MU_KM3_S2, SUN_MU_KM3_S2
from sim.dynamics.orbit.integrators import integrate_adaptive, rk4_step_state
from sim.dynamics.orbit.spherical_harmonics import accel_spherical_harmonics_terms, parse_spherical_harmonic_terms


AccelerationPlugin = Callable[[float, np.ndarray, dict, OrbitContext], np.ndarray]


def j2_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    return accel_j2(x_eci[:3], ctx.mu_km3_s2)


def j3_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    return accel_j3(x_eci[:3], ctx.mu_km3_s2)


def j4_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    return accel_j4(x_eci[:3], ctx.mu_km3_s2)


def spherical_harmonics_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    """
    Generic spherical-harmonics perturbation plugin.

    Expects `env["spherical_harmonics_terms"]` as list[dict], each with:
    - n: degree
    - m: order
    - c_nm (or c): cosine coefficient
    - s_nm (or s): sine coefficient (optional)

    Optional env fields:
    - spherical_harmonics_fd_step_km
    """
    terms = parse_spherical_harmonic_terms(env.get("spherical_harmonics_terms"))
    if not terms:
        return np.zeros(3)
    fd_step_km = float(env.get("spherical_harmonics_fd_step_km", 1e-3))
    return accel_spherical_harmonics_terms(
        r_eci_km=x_eci[:3],
        t_s=t_s,
        terms=terms,
        mu_km3_s2=ctx.mu_km3_s2,
        fd_step_km=fd_step_km,
    )


def drag_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    env_local = dict(env)
    if "density_kg_m3" not in env_local:
        atmo_model = str(env_local.get("atmosphere_model", "exponential")).lower()
        env_local["density_kg_m3"] = density_from_model(
            atmo_model,
            x_eci[:3],
            t_s,
            env=env_local,
        )
    return accel_drag(x_eci[:3], x_eci[3:], t_s, ctx.mass_kg, ctx.area_m2, ctx.cd, env_local)


def srp_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    return accel_srp(ctx.mass_kg, ctx.area_m2, ctx.cr, env)


def third_body_moon_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    moon = np.array(env.get("moon_pos_eci_km", np.array([384400.0, 0.0, 0.0])), dtype=float)
    return accel_third_body(x_eci[:3], moon, MOON_MU_KM3_S2)


def third_body_sun_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    sun = np.array(env.get("sun_pos_eci_km", np.array([149597870.7, 0.0, 0.0])), dtype=float)
    return accel_third_body(x_eci[:3], sun, SUN_MU_KM3_S2)


@dataclass
class OrbitPropagator:
    integrator: str = "rk4"
    plugins: list[AccelerationPlugin] = field(default_factory=list)
    adaptive_atol: float = 1e-9
    adaptive_rtol: float = 1e-7

    def propagate(
        self,
        x_eci: np.ndarray,
        dt_s: float,
        t_s: float,
        command_accel_eci_km_s2: np.ndarray,
        env: dict,
        ctx: OrbitContext,
    ) -> np.ndarray:
        def deriv(t_local: float, x_local: np.ndarray) -> np.ndarray:
            a = accel_two_body(x_local[:3], ctx.mu_km3_s2) + command_accel_eci_km_s2
            for plugin in self.plugins:
                a += plugin(t_local, x_local, env, ctx)
            return np.hstack((x_local[3:], a))

        if self.integrator in ("rkf78", "dopri5", "adaptive"):
            return integrate_adaptive(
                deriv_fn=deriv,
                t_s=t_s,
                x=x_eci,
                dt_s=dt_s,
                atol=self.adaptive_atol,
                rtol=self.adaptive_rtol,
            )
        return rk4_step_state(deriv_fn=deriv, t_s=t_s, x=x_eci, dt_s=dt_s)
