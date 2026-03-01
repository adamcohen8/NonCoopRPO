from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from sim.dynamics.orbit.accelerations import (
    OrbitContext,
    accel_drag,
    accel_j2,
    accel_srp,
    accel_third_body,
    accel_two_body,
    default_density_model,
)
from sim.dynamics.orbit.environment import MOON_MU_KM3_S2, SUN_MU_KM3_S2
from sim.dynamics.orbit.integrators import integrate_adaptive, rk4_step_state


AccelerationPlugin = Callable[[float, np.ndarray, dict, OrbitContext], np.ndarray]


def j2_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    return accel_j2(x_eci[:3], ctx.mu_km3_s2)


def drag_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    env_local = dict(env)
    if "density_kg_m3" not in env_local:
        env_local["density_kg_m3"] = default_density_model(x_eci[:3], t_s)
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
