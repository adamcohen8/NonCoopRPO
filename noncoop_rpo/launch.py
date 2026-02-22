from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .atmosphere import (
    EARTH_EQUATORIAL_RADIUS_KM,
    EARTH_FLATTENING,
    EARTH_ROT_RATE_RAD_S,
    atmos76_density_from_eci,
)
from .attitude import AttitudeConstraint, AttitudeRateState, apply_attitude_rate_constraint
from .constants import MU_EARTH_KM3_S2
from .cowell import propagate_cowell
from .orbital_elements import coe2rv
from .sat_params import SatParams
from .satellite import Satellite


G0_M_S2 = 9.80665


class LaunchTimingMode(str, Enum):
    GO_NOW = "go_now"
    WHEN_FEASIBLE = "when_feasible"
    OPTIMAL = "optimal"


@dataclass(frozen=True)
class LaunchSite:
    lat_deg: float
    lon_deg: float
    altitude_km: float = 0.0


@dataclass(frozen=True)
class OrbitTarget:
    a_km: float
    ecc: float
    incl_deg: float
    raan_deg: float
    argp_deg: float = 0.0
    nu_deg: float = 0.0

    def to_eci(self, mu: float = MU_EARTH_KM3_S2) -> np.ndarray:
        r0, v0 = coe2rv(
            p=self.a_km * (1.0 - self.ecc * self.ecc),
            ecc=self.ecc,
            incl=np.deg2rad(self.incl_deg),
            raan=np.deg2rad(self.raan_deg),
            argp=np.deg2rad(self.argp_deg),
            nu=np.deg2rad(self.nu_deg),
            mu=mu,
        )
        return np.hstack((r0, v0))


@dataclass(frozen=True)
class Rocket:
    isp_s: float
    dry_mass_kg: float
    fuel_mass_kg: float
    thrust_newton: float
    cd: float
    area_m2: float
    vertical_ascent_time_s: float
    pitch_over_time_s: float
    guidance_mode: str = "feedback"
    use_feedback_guidance: bool = True
    guidance_gain_a: float = 2.0
    guidance_gain_energy: float = 4.0
    guidance_gain_vr: float = 3.0
    guidance_gain_plane: float = 2.0
    guidance_gain_nominal: float = 0.5
    throttle_down_near_insertion: bool = True
    throttle_min_fraction: float = 0.2
    insertion_hold_time_s: float = 5.0
    max_flight_time_s: float = 2400.0
    dt_s: float = 0.5
    predictive_update_period_s: float = 10.0
    predictive_horizon_s: float = 180.0
    attitude_control_enabled: bool = False
    inertia_body_kg_m2: Optional[np.ndarray] = None
    max_torque_nm: Optional[np.ndarray] = None

    def initial_mass_kg(self) -> float:
        return self.dry_mass_kg + self.fuel_mass_kg

    def mdot_kg_s(self) -> float:
        if self.isp_s <= 0.0:
            return 0.0
        return self.thrust_newton / (self.isp_s * G0_M_S2)

    def max_ideal_delta_v_km_s(self) -> float:
        m0 = self.initial_mass_kg()
        if self.dry_mass_kg <= 0.0 or m0 <= self.dry_mass_kg:
            return 0.0
        return (self.isp_s * G0_M_S2 / 1000.0) * np.log(m0 / self.dry_mass_kg)


@dataclass(frozen=True)
class InsertionCriteria:
    target_orbit: OrbitTarget
    semimajor_axis_tolerance_km: Optional[float] = None
    periapsis_tolerance_km: float = 25.0
    apoapsis_tolerance_km: float = 25.0
    inclination_tolerance_deg: float = 1.5
    eccentricity_tolerance: float = 0.02
    flight_path_angle_target_deg: float = 90.0
    flight_path_angle_tolerance_deg: Optional[float] = None
    min_periapsis_altitude_km: Optional[float] = None


@dataclass
class LaunchResult:
    t_s: np.ndarray
    x_eci: np.ndarray
    fuel_mass_kg: np.ndarray
    thrust_newton: np.ndarray
    inserted: bool
    insertion_index: int
    insertion_reason: str
    satellite: Satellite
    target_state_at_insertion_eci: np.ndarray
    achieved_a_km: float
    achieved_ecc: float
    achieved_incl_deg: float
    launch_delay_s: float
    plane_error_deg_at_launch: float
    feasible_by_delta_v_check: bool
    estimated_required_delta_v_km_s: float
    estimated_available_delta_v_km_s: float


def _unit(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < eps:
        return np.zeros(3, dtype=float)
    return vec / n


def _rot3(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, h_km: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    a = EARTH_EQUATORIAL_RADIUS_KM
    f = EARTH_FLATTENING
    e2 = 2.0 * f - f * f

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    n = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + h_km) * cos_lat * cos_lon
    y = (n + h_km) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + h_km) * sin_lat
    return np.array([x, y, z], dtype=float)


def _site_eci_at_time(site: LaunchSite, t_s: float) -> np.ndarray:
    r_ecef = _geodetic_to_ecef(site.lat_deg, site.lon_deg, site.altitude_km)
    return _rot3(EARTH_ROT_RATE_RAD_S * t_s) @ r_ecef


def _target_eci_at_time(target0: np.ndarray, t_s: float, dt_s: float, mu: float) -> np.ndarray:
    if t_s <= 0.0:
        return target0.copy()
    dt_use = max(dt_s, 1e-6)
    steps = max(1, int(np.ceil(t_s / dt_use)))
    result = propagate_cowell(
        x0_eci=target0,
        dt_s=t_s / steps,
        steps=steps,
        t0_s=0.0,
        mu=mu,
    )
    return result.x_eci[-1, :].copy()


def _orbital_elements_from_state(x_eci: np.ndarray, mu: float = MU_EARTH_KM3_S2) -> tuple[float, float, float]:
    r = x_eci[0:3]
    v = x_eci[3:6]
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)

    ecc_vec = ((v_norm * v_norm - mu / r_norm) * r - np.dot(r, v) * v) / mu
    ecc = float(np.linalg.norm(ecc_vec))

    energy = 0.5 * v_norm * v_norm - mu / r_norm
    if abs(energy) < 1e-12:
        a_km = np.inf
    else:
        a_km = -mu / (2.0 * energy)

    incl = np.arccos(np.clip(h[2] / max(h_norm, 1e-12), -1.0, 1.0))
    return float(a_km), ecc, float(np.rad2deg(incl))


def _plane_matching_azimuth(lat_deg: float, target_incl_deg: float) -> tuple[float, bool]:
    lat = np.deg2rad(lat_deg)
    inc = np.deg2rad(target_incl_deg)
    denom = max(abs(np.cos(lat)), 1e-12)
    ratio = np.cos(inc) / denom
    reachable = abs(ratio) <= 1.0
    sin_az = np.clip(ratio, -1.0, 1.0)
    return float(np.arcsin(sin_az)), bool(reachable)


def _local_basis(r_eci: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = _unit(r_eci)
    k = np.array([0.0, 0.0, 1.0], dtype=float)
    east = _unit(np.cross(k, up))
    if np.linalg.norm(east) < 1e-9:
        east = np.array([0.0, 1.0, 0.0], dtype=float)
    north = _unit(np.cross(up, east))
    return up, east, north


def _launch_delay_for_mode(
    mode: LaunchTimingMode,
    site: LaunchSite,
    target0_eci: np.ndarray,
    target_orbit: OrbitTarget,
    rocket: Rocket,
    mu: float,
) -> tuple[float, float, bool, float, float]:
    available_dv_km_s = rocket.max_ideal_delta_v_km_s()

    def metrics_at(t_s: float) -> tuple[float, float, bool]:
        target_t = _target_eci_at_time(target0_eci, t_s, dt_s=max(rocket.dt_s * 5.0, 20.0), mu=mu)
        target_h_hat = _unit(np.cross(target_t[0:3], target_t[3:6]))
        r_site = _site_eci_at_time(site, t_s)
        v_site = np.cross(np.array([0.0, 0.0, EARTH_ROT_RATE_RAD_S], dtype=float), r_site)
        plane_err_deg = float(np.rad2deg(np.arcsin(np.clip(abs(np.dot(_unit(r_site), target_h_hat)), 0.0, 1.0))))

        up, east, north = _local_basis(r_site)
        azimuth_rad, reachable = _plane_matching_azimuth(site.lat_deg, target_orbit.incl_deg)
        horiz = _unit(north * np.cos(azimuth_rad) + east * np.sin(azimuth_rad))
        target_v_vec = target_t[3:6]
        if np.linalg.norm(target_v_vec) <= 1e-9:
            target_speed_km_s = np.sqrt(mu / max(target_orbit.a_km, 1e-9))
            target_v_vec = target_speed_km_s * horiz
        dv_vector_km_s = np.linalg.norm(target_v_vec - v_site)

        mdot = rocket.mdot_kg_s()
        burn_time_s = rocket.fuel_mass_kg / max(mdot, 1e-12) if mdot > 0.0 else np.inf
        gravity_loss_km_s = float(np.clip(0.30 * G0_M_S2 * min(burn_time_s, 900.0) / 1000.0, 0.3, 2.7))
        ballistic_coeff = rocket.initial_mass_kg() / max(rocket.cd * rocket.area_m2, 1e-9)
        drag_loss_km_s = float(np.clip(1200.0 / max(ballistic_coeff, 1e-9), 0.03, 0.45))
        plane_penalty_km_s = float((0.6 * plane_err_deg / 90.0) + (0.8 if not reachable else 0.0))
        required_dv_km_s = dv_vector_km_s + gravity_loss_km_s + drag_loss_km_s + plane_penalty_km_s
        feasible = available_dv_km_s >= required_dv_km_s
        return plane_err_deg, required_dv_km_s, feasible

    if mode == LaunchTimingMode.GO_NOW:
        plane_err_deg, required_dv_km_s, feasible = metrics_at(0.0)
        return 0.0, plane_err_deg, feasible, required_dv_km_s, available_dv_km_s

    window_s = 86164.0905
    samples = 721
    t_grid = np.linspace(0.0, window_s, samples)
    plane_tol_for_feasible_deg = 2.5

    if mode == LaunchTimingMode.WHEN_FEASIBLE:
        fallback = (0.0, np.inf, False, np.inf)
        for t_s in t_grid:
            plane_err_deg, required_dv_km_s, feasible = metrics_at(float(t_s))
            if plane_err_deg < fallback[1]:
                fallback = (float(t_s), plane_err_deg, feasible, required_dv_km_s)
            if feasible and plane_err_deg <= plane_tol_for_feasible_deg:
                return float(t_s), plane_err_deg, feasible, required_dv_km_s, available_dv_km_s
        return fallback[0], fallback[1], fallback[2], fallback[3], available_dv_km_s

    best_t = 0.0
    best_err = 180.0
    best_required = np.inf
    best_feasible = False
    best_score = np.inf
    for t_s in t_grid:
        plane_err_deg, required_dv_km_s, feasible = metrics_at(float(t_s))
        dv_shortfall = max(required_dv_km_s - available_dv_km_s, 0.0)
        score = plane_err_deg + 25.0 * dv_shortfall + 1e-9 * float(t_s)
        if score < best_score:
            best_score = score
            best_t = float(t_s)
            best_err = plane_err_deg
            best_required = required_dv_km_s
            best_feasible = feasible
    return best_t, best_err, best_feasible, best_required, available_dv_km_s


def _insertion_met(x_eci: np.ndarray, criteria: InsertionCriteria) -> bool:
    a_km, ecc, incl_deg = _orbital_elements_from_state(x_eci)
    if not np.isfinite(a_km) or a_km <= 0.0:
        return False

    r = x_eci[0:3]
    v = x_eci[3:6]
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    if r_norm <= 0.0 or v_norm <= 0.0:
        return False
    rv_angle_deg = float(np.rad2deg(np.arccos(np.clip(np.dot(r, v) / (r_norm * v_norm), -1.0, 1.0))))

    re = EARTH_EQUATORIAL_RADIUS_KM
    targ = criteria.target_orbit
    incl_ok = abs(incl_deg - targ.incl_deg) <= criteria.inclination_tolerance_deg
    ecc_ok = abs(ecc - targ.ecc) <= criteria.eccentricity_tolerance
    fpa_ok = True
    if criteria.flight_path_angle_tolerance_deg is not None:
        fpa_ok = abs(rv_angle_deg - criteria.flight_path_angle_target_deg) <= criteria.flight_path_angle_tolerance_deg
    peri_floor_ok = True
    if criteria.min_periapsis_altitude_km is not None:
        rp_alt = a_km * (1.0 - ecc) - re
        peri_floor_ok = rp_alt >= criteria.min_periapsis_altitude_km

    if criteria.semimajor_axis_tolerance_km is not None:
        a_ok = abs(a_km - targ.a_km) <= criteria.semimajor_axis_tolerance_km
        return bool(a_ok and incl_ok and ecc_ok and fpa_ok and peri_floor_ok)

    rp_alt = a_km * (1.0 - ecc) - re
    ra_alt = a_km * (1.0 + ecc) - re
    targ_rp = targ.a_km * (1.0 - targ.ecc) - re
    targ_ra = targ.a_km * (1.0 + targ.ecc) - re
    peri_ok = abs(rp_alt - targ_rp) <= criteria.periapsis_tolerance_km
    apo_ok = abs(ra_alt - targ_ra) <= criteria.apoapsis_tolerance_km
    return bool(peri_ok and apo_ok and incl_ok and ecc_ok and fpa_ok and peri_floor_ok)


def _insertion_error_score(x_eci: np.ndarray, criteria: InsertionCriteria) -> float:
    """
    Normalized insertion error score.
    <= 1.0 means all active criteria are met or near-met.
    """
    a_km, ecc, incl_deg = _orbital_elements_from_state(x_eci)
    if not np.isfinite(a_km) or a_km <= 0.0:
        return 10.0

    r = x_eci[0:3]
    v = x_eci[3:6]
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    if r_norm <= 0.0 or v_norm <= 0.0:
        return 10.0

    rv_angle_deg = float(np.rad2deg(np.arccos(np.clip(np.dot(r, v) / (r_norm * v_norm), -1.0, 1.0))))
    re = EARTH_EQUATORIAL_RADIUS_KM
    rp_alt = a_km * (1.0 - ecc) - re
    ra_alt = a_km * (1.0 + ecc) - re
    targ = criteria.target_orbit

    terms: list[float] = []
    terms.append(abs(incl_deg - targ.incl_deg) / max(criteria.inclination_tolerance_deg, 1e-9))
    terms.append(abs(ecc - targ.ecc) / max(criteria.eccentricity_tolerance, 1e-9))
    if criteria.flight_path_angle_tolerance_deg is not None:
        terms.append(
            abs(rv_angle_deg - criteria.flight_path_angle_target_deg)
            / max(criteria.flight_path_angle_tolerance_deg, 1e-9)
        )
    if criteria.semimajor_axis_tolerance_km is not None:
        terms.append(abs(a_km - targ.a_km) / max(criteria.semimajor_axis_tolerance_km, 1e-9))
    else:
        targ_rp = targ.a_km * (1.0 - targ.ecc) - re
        targ_ra = targ.a_km * (1.0 + targ.ecc) - re
        terms.append(abs(rp_alt - targ_rp) / max(criteria.periapsis_tolerance_km, 1e-9))
        terms.append(abs(ra_alt - targ_ra) / max(criteria.apoapsis_tolerance_km, 1e-9))
    if criteria.min_periapsis_altitude_km is not None and rp_alt < criteria.min_periapsis_altitude_km:
        terms.append((criteria.min_periapsis_altitude_km - rp_alt) / 50.0)

    return float(max(terms) if terms else 10.0)


def _periapsis_apoapsis_altitudes_km(x_eci: np.ndarray, mu: float) -> tuple[float, float]:
    a_km, ecc, _incl = _orbital_elements_from_state(x_eci, mu=mu)
    if not np.isfinite(a_km) or a_km <= 0.0:
        return -np.inf, np.inf
    re = EARTH_EQUATORIAL_RADIUS_KM
    rp_alt = a_km * (1.0 - ecc) - re
    ra_alt = a_km * (1.0 + ecc) - re
    return float(rp_alt), float(ra_alt)


def _feedback_thrust_direction(
    x_eci: np.ndarray,
    target_orbit: OrbitTarget,
    target_h_hat: np.ndarray,
    nominal_horiz: np.ndarray,
    rocket: Rocket,
    mu: float,
) -> np.ndarray:
    r = x_eci[0:3]
    v = x_eci[3:6]
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    if r_norm <= 1e-9 or v_norm <= 1e-9:
        return nominal_horiz

    up = r / r_norm
    v_hat = v / v_norm

    h_hat = _unit(np.cross(r, v))
    plane_vec = np.cross(h_hat, target_h_hat)

    a_km, _ecc, _incl_deg = _orbital_elements_from_state(x_eci, mu=mu)
    if not np.isfinite(a_km):
        a_km = target_orbit.a_km
    err_a = np.clip((target_orbit.a_km - a_km) / max(target_orbit.a_km, 1e-9), -1.0, 1.0)
    energy_current = 0.5 * v_norm * v_norm - mu / r_norm
    energy_target = -mu / (2.0 * max(target_orbit.a_km, 1e-9))
    err_energy = np.clip((energy_target - energy_current) / max(abs(energy_target), 1e-9), -2.0, 2.0)

    vr = np.dot(v, up)
    radial_term = -rocket.guidance_gain_vr * (vr / max(v_norm, 1e-9)) * up
    plane_term = rocket.guidance_gain_plane * plane_vec
    prograde_scale = rocket.guidance_gain_a * err_a + rocket.guidance_gain_energy * err_energy
    prograde_term = prograde_scale * v_hat
    nominal_term = rocket.guidance_gain_nominal * nominal_horiz

    cmd = prograde_term + radial_term + plane_term + nominal_term
    cmd_u = _unit(cmd)
    if np.linalg.norm(cmd_u) < 1e-12:
        return nominal_horiz
    return cmd_u


def _predictive_shooting_thrust_direction(
    x_eci: np.ndarray,
    fuel_kg: float,
    phase: str,
    target_orbit: OrbitTarget,
    insertion_criteria: InsertionCriteria,
    nominal_horiz: np.ndarray,
    rocket: Rocket,
    mu: float,
) -> np.ndarray:
    r = x_eci[0:3]
    v = x_eci[3:6]
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    if r_norm <= 1e-9:
        return nominal_horiz

    up = _unit(r)
    h_hat = _unit(np.cross(r, v))
    if np.linalg.norm(h_hat) < 1e-12:
        h_hat = _unit(np.cross(up, nominal_horiz))
    prograde = _unit(v) if v_norm > 1e-9 else nominal_horiz
    if np.linalg.norm(prograde) < 1e-12:
        prograde = nominal_horiz
    cross_track = _unit(np.cross(prograde, up))
    if np.linalg.norm(cross_track) < 1e-12:
        cross_track = _unit(h_hat)

    target_ra_alt = target_orbit.a_km * (1.0 + target_orbit.ecc) - EARTH_EQUATORIAL_RADIUS_KM
    target_rp_alt = target_orbit.a_km * (1.0 - target_orbit.ecc) - EARTH_EQUATORIAL_RADIUS_KM
    ra_tol = max(insertion_criteria.apoapsis_tolerance_km, 20.0)
    rp_tol = max(insertion_criteria.periapsis_tolerance_km, 20.0)

    candidates = [
        prograde,
        _unit(prograde + 0.08 * up),
        _unit(prograde - 0.08 * up),
        _unit(prograde + 0.08 * cross_track),
        _unit(prograde - 0.08 * cross_track),
        _unit(0.8 * prograde + 0.2 * nominal_horiz),
    ]

    thrust_n = rocket.thrust_newton if (fuel_kg > 0.0 and rocket.mdot_kg_s() > 0.0) else 0.0
    mdot = rocket.mdot_kg_s()
    dt_sim = max(1.0, 2.0 * rocket.dt_s)
    horizon_s = max(20.0, rocket.predictive_horizon_s)
    steps = max(1, int(np.ceil(horizon_s / dt_sim)))

    def rollout(dir_cmd: np.ndarray) -> tuple[np.ndarray, float]:
        state = np.hstack((x_eci.copy(), max(fuel_kg, 0.0)))
        for _ in range(steps):
            def deriv(local_state: np.ndarray) -> np.ndarray:
                local_x = local_state[0:6]
                local_fuel = max(float(local_state[6]), 0.0)
                local_r = local_x[0:3]
                local_v = local_x[3:6]
                local_r_norm = np.linalg.norm(local_r)
                mass = rocket.dry_mass_kg + local_fuel
                a_grav = -mu * local_r / (local_r_norm**3)

                rho = atmos76_density_from_eci(local_r)
                v_atm = np.cross(np.array([0.0, 0.0, EARTH_ROT_RATE_RAD_S], dtype=float), local_r)
                v_rel = local_v - v_atm
                speed = np.linalg.norm(v_rel)
                area_over_mass_km2_per_kg = (rocket.area_m2 * 1e-6) / max(mass, 1e-9)
                a_drag = -0.5 * rocket.cd * area_over_mass_km2_per_kg * rho * speed * v_rel

                throttle = 1.0 if (local_fuel > 0.0 and thrust_n > 0.0) else 0.0
                a_thrust = (thrust_n / max(mass, 1e-9)) * dir_cmd * throttle / 1000.0
                fuel_dot = -mdot * throttle if throttle > 0.0 else 0.0
                return np.hstack((local_v, a_grav + a_drag + a_thrust, fuel_dot))

            k1 = deriv(state)
            k2 = deriv(state + 0.5 * dt_sim * k1)
            k3 = deriv(state + 0.5 * dt_sim * k2)
            k4 = deriv(state + dt_sim * k3)
            state = state + (dt_sim / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            state[6] = max(state[6], 0.0)

        pred_x = state[0:6]
        score = _insertion_error_score(pred_x, insertion_criteria)
        rp_alt, ra_alt = _periapsis_apoapsis_altitudes_km(pred_x, mu=mu)
        if phase == "burn1_raise_apogee":
            score += abs(ra_alt - target_ra_alt) / max(ra_tol, 1e-9)
        if phase == "burn2_circularize":
            score += abs(rp_alt - target_rp_alt) / max(rp_tol, 1e-9)
            if ra_alt > target_ra_alt + ra_tol:
                score += 4.0 * (ra_alt - (target_ra_alt + ra_tol)) / max(ra_tol, 1e-9)
        return pred_x, float(score)

    best_dir = prograde
    best_score = np.inf
    for cand in candidates:
        cand_u = _unit(cand)
        if np.linalg.norm(cand_u) < 1e-12:
            continue
        _, score = rollout(cand_u)
        if score < best_score:
            best_score = score
            best_dir = cand_u
    return best_dir


def simulate_launch_to_insertion(
    site: LaunchSite,
    rocket: Rocket,
    target_orbit: OrbitTarget,
    insertion_criteria: InsertionCriteria,
    timing_mode: LaunchTimingMode = LaunchTimingMode.GO_NOW,
    mu: float = MU_EARTH_KM3_S2,
    satellite_name: str = "inserted_satellite",
) -> LaunchResult:
    if rocket.dt_s <= 0.0:
        raise ValueError("rocket.dt_s must be positive.")
    if rocket.fuel_mass_kg < 0.0:
        raise ValueError("rocket.fuel_mass_kg must be non-negative.")
    if rocket.dry_mass_kg <= 0.0:
        raise ValueError("rocket.dry_mass_kg must be positive.")
    AttitudeConstraint(
        enabled=rocket.attitude_control_enabled,
        inertia_body_kg_m2=rocket.inertia_body_kg_m2,
        max_torque_nm=rocket.max_torque_nm,
    ).validate("Rocket")

    guidance_orbit = insertion_criteria.target_orbit
    target0 = target_orbit.to_eci(mu=mu)
    launch_delay_s, plane_err_deg, feasible, required_dv_km_s, available_dv_km_s = _launch_delay_for_mode(
        mode=timing_mode,
        site=site,
        target0_eci=target0,
        target_orbit=target_orbit,
        rocket=rocket,
        mu=mu,
    )
    target_at_launch = _target_eci_at_time(target0, launch_delay_s, dt_s=max(rocket.dt_s * 5.0, 20.0), mu=mu)
    target_h_hat = _unit(np.cross(target_at_launch[0:3], target_at_launch[3:6]))

    r0 = _site_eci_at_time(site, launch_delay_s)
    v0 = np.cross(np.array([0.0, 0.0, EARTH_ROT_RATE_RAD_S], dtype=float), r0)

    azimuth_rad, _ = _plane_matching_azimuth(site.lat_deg, guidance_orbit.incl_deg)
    dt_s = rocket.dt_s
    steps = int(np.ceil(rocket.max_flight_time_s / dt_s))

    t_log = np.zeros(steps + 1, dtype=float)
    x_log = np.zeros((steps + 1, 6), dtype=float)
    fuel_log = np.zeros(steps + 1, dtype=float)
    thrust_log = np.zeros(steps + 1, dtype=float)

    x = np.hstack((r0, v0))
    fuel_kg = rocket.fuel_mass_kg
    t_log[0] = launch_delay_s
    x_log[0, :] = x
    fuel_log[0] = fuel_kg
    thrust_log[0] = 0.0

    inserted = False
    insertion_idx = 0
    insertion_reason = "max_time_reached"
    mdot = rocket.mdot_kg_s()
    hold_steps = max(1, int(np.ceil(max(rocket.insertion_hold_time_s, 0.0) / dt_s)))
    insertion_met_count = 0
    phase = "burn1_raise_apogee"
    coast_prev_vr: Optional[float] = None
    shooting_cmd_dir = _unit(v0) if np.linalg.norm(v0) > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=float)
    shooting_last_update_t_rel_s = -np.inf
    rocket_attitude_state = AttitudeRateState(thrust_axis=_unit(v0), slew_rate_rad_s=0.0)
    rocket_attitude_constraint = AttitudeConstraint(
        enabled=rocket.attitude_control_enabled,
        inertia_body_kg_m2=rocket.inertia_body_kg_m2,
        max_torque_nm=rocket.max_torque_nm,
    )
    re = EARTH_EQUATORIAL_RADIUS_KM
    target_rp_alt = guidance_orbit.a_km * (1.0 - guidance_orbit.ecc) - re
    target_ra_alt = guidance_orbit.a_km * (1.0 + guidance_orbit.ecc) - re
    apogee_raise_tol_km = max(insertion_criteria.apoapsis_tolerance_km, 20.0)

    for k in range(steps):
        t_abs = launch_delay_s + k * dt_s
        t_rel = k * dt_s

        up, _, _ = _local_basis(x[0:3])
        vr_now = float(np.dot(x[3:6], up))
        r_norm_now = float(np.linalg.norm(x[0:3]))
        v_norm_now = float(np.linalg.norm(x[3:6]))
        _, ra_alt_now = _periapsis_apoapsis_altitudes_km(x, mu=mu)
        target_ra_radius_km = re + target_ra_alt
        if r_norm_now < target_ra_radius_km:
            transfer_a_km = 0.5 * (r_norm_now + target_ra_radius_km)
            burn1_speed_limit_km_s = np.sqrt(max(mu * (2.0 / r_norm_now - 1.0 / transfer_a_km), 0.0))
        else:
            burn1_speed_limit_km_s = np.sqrt(mu / max(r_norm_now, 1e-9))
        burn1_speed_guard_hit = v_norm_now >= 0.995 * burn1_speed_limit_km_s

        if phase == "burn1_raise_apogee":
            if ra_alt_now >= (target_ra_alt - apogee_raise_tol_km) or burn1_speed_guard_hit:
                if vr_now > 0.0:
                    phase = "coast_to_apogee"
                    coast_prev_vr = vr_now
                else:
                    phase = "burn2_circularize"
        elif phase == "coast_to_apogee":
            if coast_prev_vr is None:
                coast_prev_vr = vr_now
            if coast_prev_vr > 0.0 and vr_now <= 0.0:
                phase = "burn2_circularize"
            coast_prev_vr = vr_now

        guidance_mode = rocket.guidance_mode.strip().lower()
        if (
            guidance_mode == "predictive_shooting"
            and t_rel > rocket.vertical_ascent_time_s + rocket.pitch_over_time_s
            and phase != "coast_to_apogee"
            and (t_rel - shooting_last_update_t_rel_s) >= max(rocket.predictive_update_period_s, rocket.dt_s)
        ):
            up_now, east_now, north_now = _local_basis(x[0:3])
            nominal_horiz_now = _unit(north_now * np.cos(azimuth_rad) + east_now * np.sin(azimuth_rad))
            shooting_cmd_dir = _predictive_shooting_thrust_direction(
                x_eci=x,
                fuel_kg=fuel_kg,
                phase=phase,
                target_orbit=guidance_orbit,
                insertion_criteria=insertion_criteria,
                nominal_horiz=nominal_horiz_now,
                rocket=rocket,
                mu=mu,
            )
            shooting_last_update_t_rel_s = t_rel

        def thrust_direction_for_state(local_x: np.ndarray, local_t_rel_s: float) -> np.ndarray:
            local_up, local_east, local_north = _local_basis(local_x[0:3])
            local_horiz = _unit(local_north * np.cos(azimuth_rad) + local_east * np.sin(azimuth_rad))

            if local_t_rel_s <= rocket.vertical_ascent_time_s:
                return local_up
            if local_t_rel_s <= rocket.vertical_ascent_time_s + rocket.pitch_over_time_s:
                alpha = (local_t_rel_s - rocket.vertical_ascent_time_s) / max(rocket.pitch_over_time_s, 1e-9)
                return _unit((1.0 - alpha) * local_up + alpha * local_horiz)
            if guidance_mode == "predictive_shooting":
                return shooting_cmd_dir
            if rocket.use_feedback_guidance:
                return _feedback_thrust_direction(
                    x_eci=local_x,
                    target_orbit=guidance_orbit,
                    target_h_hat=target_h_hat,
                    nominal_horiz=local_horiz,
                    rocket=rocket,
                    mu=mu,
                )
            return local_horiz

        desired_thrust_dir_now = thrust_direction_for_state(x, t_rel)
        rocket_attitude_state = apply_attitude_rate_constraint(
            desired_dir=desired_thrust_dir_now,
            state=rocket_attitude_state,
            dt_s=dt_s,
            constraint=rocket_attitude_constraint,
        )
        thrust_dir_cmd = rocket_attitude_state.thrust_axis

        def deriv(local_t_rel_s: float, local_state: np.ndarray) -> np.ndarray:
            local_x = local_state[0:6]
            local_fuel_kg = max(float(local_state[6]), 0.0)
            r = local_x[0:3]
            v = local_x[3:6]
            r_norm = np.linalg.norm(r)
            mass = rocket.dry_mass_kg + local_fuel_kg
            a_grav = -mu * r / (r_norm**3)

            rho = atmos76_density_from_eci(r)
            v_atm = np.cross(np.array([0.0, 0.0, EARTH_ROT_RATE_RAD_S], dtype=float), r)
            v_rel = v - v_atm
            speed = np.linalg.norm(v_rel)
            area_over_mass_km2_per_kg = (rocket.area_m2 * 1e-6) / max(mass, 1e-9)
            a_drag = -0.5 * rocket.cd * area_over_mass_km2_per_kg * rho * speed * v_rel

            def throttle_for_state(state_x: np.ndarray, state_fuel_kg: float) -> float:
                throttle_cmd_local = 0.0
                if phase != "coast_to_apogee" and state_fuel_kg > 0.0 and mdot > 0.0:
                    throttle_cmd_local = 1.0
                    if phase == "burn2_circularize":
                        rp_alt, ra_alt = _periapsis_apoapsis_altitudes_km(state_x, mu=mu)
                        state_r = state_x[0:3]
                        state_v = state_x[3:6]
                        state_up = _unit(state_r)
                        state_vr = float(np.dot(state_v, state_up))
                        near_apogee = abs(state_vr) <= 0.12
                        apo_overshoot = ra_alt > (target_ra_alt + apogee_raise_tol_km)
                        perigee_still_low = rp_alt < (target_rp_alt - insertion_criteria.periapsis_tolerance_km)
                        if apo_overshoot or not (near_apogee and perigee_still_low):
                            throttle_cmd_local = 0.0
                    if throttle_cmd_local > 0.0 and rocket.throttle_down_near_insertion and phase == "burn2_circularize":
                        score = _insertion_error_score(state_x, insertion_criteria)
                        if np.isfinite(score):
                            throttle_cmd_local = float(np.clip(score, rocket.throttle_min_fraction, 1.0))
                return throttle_cmd_local

            throttle_cmd = throttle_for_state(local_x, local_fuel_kg)

            thrust_dir = thrust_dir_cmd
            thrust_n = rocket.thrust_newton * throttle_cmd
            a_thrust = (thrust_n / max(mass, 1e-9)) * thrust_dir / 1000.0
            fuel_dot = -mdot * throttle_cmd if throttle_cmd > 0.0 else 0.0
            a_total = a_grav + a_drag + a_thrust
            return np.hstack((v, a_total, fuel_dot))

        throttle_cmd_now = 0.0
        if phase != "coast_to_apogee" and fuel_kg > 0.0 and mdot > 0.0:
            throttle_cmd_now = 1.0
            if phase == "burn2_circularize":
                rp_alt_now, ra_alt_now_b2 = _periapsis_apoapsis_altitudes_km(x, mu=mu)
                state_up_now = _unit(x[0:3])
                state_vr_now = float(np.dot(x[3:6], state_up_now))
                near_apogee_now = abs(state_vr_now) <= 0.12
                apo_overshoot_now = ra_alt_now_b2 > (target_ra_alt + apogee_raise_tol_km)
                perigee_still_low_now = rp_alt_now < (target_rp_alt - insertion_criteria.periapsis_tolerance_km)
                if apo_overshoot_now or not (near_apogee_now and perigee_still_low_now):
                    throttle_cmd_now = 0.0
            if throttle_cmd_now > 0.0 and rocket.throttle_down_near_insertion and phase == "burn2_circularize":
                score_now = _insertion_error_score(x, insertion_criteria)
                if np.isfinite(score_now):
                    throttle_cmd_now = float(np.clip(score_now, rocket.throttle_min_fraction, 1.0))
        thrust_log[k] = rocket.thrust_newton * throttle_cmd_now

        state = np.hstack((x, fuel_kg))
        k1 = deriv(t_rel, state)
        k2 = deriv(t_rel + 0.5 * dt_s, state + 0.5 * dt_s * k1)
        k3 = deriv(t_rel + 0.5 * dt_s, state + 0.5 * dt_s * k2)
        k4 = deriv(t_rel + dt_s, state + dt_s * k3)
        state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x = state[0:6]
        fuel_kg = max(0.0, float(state[6]))

        if fuel_kg <= 0.0 and phase != "coast_to_apogee":
            phase = "coast_to_apogee"

        t_log[k + 1] = t_abs + dt_s
        x_log[k + 1, :] = x
        fuel_log[k + 1] = fuel_kg
        thrust_log[k + 1] = thrust_log[k]

        if _insertion_met(x, insertion_criteria):
            insertion_met_count += 1
            if insertion_met_count >= hold_steps:
                inserted = True
                insertion_idx = k + 1
                insertion_reason = "insertion_orbit_achieved"
                break
        else:
            insertion_met_count = 0

        if fuel_kg <= 0.0:
            insertion_idx = k + 1
            insertion_reason = "out_of_fuel"
            break

    if insertion_idx == 0:
        insertion_idx = steps

    x_insert = x_log[insertion_idx, :].copy()
    sat_params = SatParams(
        name=satellite_name,
        r0_eci_km=x_insert[0:3],
        v0_eci_km_s=x_insert[3:6],
        max_accel_km_s2=0.0,
        min_accel_km_s2=0.0,
        propellant_dv_km_s=np.inf,
    )
    inserted_sat = Satellite.from_params(sat_params, policy=None)
    inserted_sat.state.t = t_log[insertion_idx]

    target_at_insert = _target_eci_at_time(target0, t_log[insertion_idx], dt_s=10.0, mu=mu)
    a_km, ecc, incl_deg = _orbital_elements_from_state(x_insert, mu=mu)

    return LaunchResult(
        t_s=t_log[: insertion_idx + 1].copy(),
        x_eci=x_log[: insertion_idx + 1, :].copy(),
        fuel_mass_kg=fuel_log[: insertion_idx + 1].copy(),
        thrust_newton=thrust_log[: insertion_idx + 1].copy(),
        inserted=inserted,
        insertion_index=insertion_idx,
        insertion_reason=insertion_reason,
        satellite=inserted_sat,
        target_state_at_insertion_eci=target_at_insert,
        achieved_a_km=a_km,
        achieved_ecc=ecc,
        achieved_incl_deg=incl_deg,
        launch_delay_s=launch_delay_s,
        plane_error_deg_at_launch=plane_err_deg,
        feasible_by_delta_v_check=feasible,
        estimated_required_delta_v_km_s=required_dv_km_s,
        estimated_available_delta_v_km_s=available_dv_km_s,
    )
