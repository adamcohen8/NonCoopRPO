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
from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.epoch import resolve_body_position_eci_km, resolve_sun_moon_positions
from sim.dynamics.orbit.environment import (
    JUPITER_MU_KM3_S2,
    MARS_MU_KM3_S2,
    MERCURY_MU_KM3_S2,
    MOON_MU_KM3_S2,
    NEPTUNE_MU_KM3_S2,
    PLUTO_MU_KM3_S2,
    SATURN_MU_KM3_S2,
    SUN_MU_KM3_S2,
    URANUS_MU_KM3_S2,
    VENUS_MU_KM3_S2,
)
from sim.dynamics.orbit.integrators import integrate_adaptive, rk4_step_state
from sim.dynamics.orbit.spherical_harmonics import (
    accel_spherical_harmonics_terms,
    load_real_earth_gravity_terms,
    parse_spherical_harmonic_terms,
)


AccelerationPlugin = Callable[[float, np.ndarray, dict, OrbitContext], np.ndarray]
PLANETARY_MU_KM3_S2 = {
    "mercury": MERCURY_MU_KM3_S2,
    "venus": VENUS_MU_KM3_S2,
    "mars": MARS_MU_KM3_S2,
    "jupiter": JUPITER_MU_KM3_S2,
    "saturn": SATURN_MU_KM3_S2,
    "uranus": URANUS_MU_KM3_S2,
    "neptune": NEPTUNE_MU_KM3_S2,
    "pluto": PLUTO_MU_KM3_S2,
}
_ZERO3 = np.zeros(3, dtype=float)


def j2_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    return accel_j2(x_eci[:3], ctx.mu_km3_s2)


def j3_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    return accel_j3(x_eci[:3], ctx.mu_km3_s2)


def j4_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    return accel_j4(x_eci[:3], ctx.mu_km3_s2)


def spherical_harmonics_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    """
    Generic spherical-harmonics perturbation plugin.

    Expects one of:
    1) `env["spherical_harmonics_terms"]` as list[dict], each with:
    - n: degree
    - m: order
    - c_nm (or c): cosine coefficient
    - s_nm (or s): sine coefficient (optional)
    - normalized: whether coefficients are fully normalized (optional; default False)

    2) Real-coefficient mode:
    - spherical_harmonics_use_real_coefficients: bool (True)
    - spherical_harmonics_model: e.g., "EGM96" (optional; default EGM96)
    - spherical_harmonics_coeff_path: local .gfc path (optional)
    - spherical_harmonics_max_degree: int (optional; default 8)
    - spherical_harmonics_max_order: int (optional; default max_degree)
    - spherical_harmonics_allow_download: bool (optional; default True)

    Optional env fields:
    - spherical_harmonics_fd_step_km
    """
    terms = env.get("_parsed_spherical_harmonics_terms")
    if terms is None:
        terms = parse_spherical_harmonic_terms(env.get("spherical_harmonics_terms"))
        if terms:
            env["_parsed_spherical_harmonics_terms"] = terms
    if not terms and bool(env.get("spherical_harmonics_use_real_coefficients", False)):
        n_max = int(env.get("spherical_harmonics_max_degree", 8))
        m_max = int(env.get("spherical_harmonics_max_order", n_max))
        model = str(env.get("spherical_harmonics_model", "EGM96"))
        coeff_path = env.get("spherical_harmonics_coeff_path")
        allow_download = bool(env.get("spherical_harmonics_allow_download", True))
        cache_key = (
            n_max,
            m_max,
            model,
            None if coeff_path is None else str(coeff_path),
            allow_download,
        )
        cached_terms = env.get("_real_spherical_harmonics_cache")
        if cached_terms is None or cached_terms[0] != cache_key:
            terms = load_real_earth_gravity_terms(
                max_degree=n_max,
                max_order=m_max,
                model=model,
                coeff_path=None if coeff_path is None else str(coeff_path),
                allow_download=allow_download,
            )
            env["_real_spherical_harmonics_cache"] = (cache_key, terms)
        else:
            terms = cached_terms[1]
    if not terms:
        return np.zeros(3)
    fd_step_km = float(env.get("spherical_harmonics_fd_step_km", 1e-3))
    jd_utc_start = env.get("jd_utc_start")
    if jd_utc_start is None and "jd_utc" in env:
        jd_utc_start = float(env["jd_utc"]) - float(t_s) / 86400.0
    return accel_spherical_harmonics_terms(
        r_eci_km=x_eci[:3],
        t_s=t_s,
        terms=terms,
        mu_km3_s2=ctx.mu_km3_s2,
        fd_step_km=fd_step_km,
        jd_utc_start=None if jd_utc_start is None else float(jd_utc_start),
    )


def drag_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    density = env.get("density_kg_m3")
    if density is None:
        atmo_model = str(env.get("atmosphere_model", "exponential")).lower()
        density = density_from_model(
            atmo_model,
            x_eci[:3],
            t_s,
            env=env,
        )
    return accel_drag(
        x_eci[:3],
        x_eci[3:],
        t_s,
        ctx.mass_kg,
        ctx.area_m2,
        ctx.cd,
        {
            "density_kg_m3": density,
            "drag_area_m2": env.get("drag_area_m2", ctx.area_m2),
        },
    )


def srp_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    srp_geometry = resolve_srp_geometry(x_eci[:3], t_s, env)
    return accel_srp(
        x_eci[:3],
        ctx.mass_kg,
        ctx.area_m2,
        ctx.cr,
        t_s,
        {
            "srp_geometry": srp_geometry,
            "srp_sun_dir_eci": srp_geometry["sun_dir_sc_eci"],
            "srp_distance_scale": srp_geometry["distance_scale"],
            "srp_shadow_factor": srp_shadow_factor(
                r_sc_eci_km=x_eci[:3],
                t_s=t_s,
                env=env,
                srp_geometry=srp_geometry,
            ),
            "srp_area_m2": env.get("srp_area_m2", ctx.area_m2),
            "srp_shadow_model": env.get("srp_shadow_model", "conical"),
        },
    )


def third_body_moon_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    moon = env.get("moon_pos_eci_km")
    if moon is None:
        _, moon = resolve_sun_moon_positions(env, t_s)
    return accel_third_body(x_eci[:3], moon, MOON_MU_KM3_S2)


def third_body_sun_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    sun = env.get("sun_pos_eci_km")
    if sun is None:
        sun, _ = resolve_sun_moon_positions(env, t_s)
    return accel_third_body(x_eci[:3], sun, SUN_MU_KM3_S2)


def third_body_planets_plugin(t_s: float, x_eci: np.ndarray, env: dict, ctx: OrbitContext) -> np.ndarray:
    selected = env.get("third_body_planets", [])
    if isinstance(selected, str):
        selected_names = [selected.strip().lower()]
    else:
        selected_names = [str(v).strip().lower() for v in selected]
    if any(v in ("all", "*") for v in selected_names):
        selected_names = list(PLANETARY_MU_KM3_S2.keys())

    acc = np.zeros(3)
    for name in selected_names:
        if name not in PLANETARY_MU_KM3_S2:
            continue
        pos = resolve_body_position_eci_km(name, env=env, t_s=t_s)
        mu = float(env.get(f"{name}_mu_km3_s2", PLANETARY_MU_KM3_S2[name]))
        acc += accel_third_body(x_eci[:3], pos, mu)
    return acc


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
            dx = np.empty(6, dtype=float)
            dx[:3] = x_local[3:]
            a = accel_two_body(x_local[:3], ctx.mu_km3_s2) + command_accel_eci_km_s2
            for plugin in self.plugins:
                a += plugin(t_local, x_local, env, ctx)
            dx[3:] = a
            return dx

        if self.integrator in ("rkf78", "dopri5", "adaptive"):
            adaptive_method = "rkf78" if self.integrator in ("rkf78", "adaptive") else "dopri5"
            return integrate_adaptive(
                deriv_fn=deriv,
                t_s=t_s,
                x=x_eci,
                dt_s=dt_s,
                atol=self.adaptive_atol,
                rtol=self.adaptive_rtol,
                method=adaptive_method,
            )
        return rk4_step_state(deriv_fn=deriv, t_s=t_s, x=x_eci, dt_s=dt_s)
