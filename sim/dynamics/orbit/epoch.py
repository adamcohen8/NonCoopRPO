from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

AU_KM = 149597870.7


def datetime_to_julian_date(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    y = dt_utc.year
    m = dt_utc.month
    d = dt_utc.day
    frac_day = (
        dt_utc.hour / 24.0
        + dt_utc.minute / 1440.0
        + (dt_utc.second + dt_utc.microsecond * 1e-6) / 86400.0
    )
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + (a // 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + frac_day + b - 1524.5
    return float(jd)


def julian_date_to_datetime(jd_utc: float) -> datetime:
    jd = float(jd_utc) + 0.5
    z = int(np.floor(jd))
    f = jd - z
    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715

    day_int = int(np.floor(day))
    frac = day - day_int
    sec_total = frac * 86400.0
    hour = int(sec_total // 3600.0)
    sec_total -= hour * 3600.0
    minute = int(sec_total // 60.0)
    sec_total -= minute * 60.0
    second = int(np.floor(sec_total))
    usec = int(np.round((sec_total - second) * 1e6))
    if usec >= 1_000_000:
        usec -= 1_000_000
        second += 1
    if second >= 60:
        second -= 60
        minute += 1
    if minute >= 60:
        minute -= 60
        hour += 1
    if hour >= 24:
        hour -= 24
        day_int += 1
    return datetime(year, month, day_int, hour, minute, second, usec, tzinfo=timezone.utc)


def gmst_angle_rad_from_jd(jd_utc: float) -> float:
    jd = float(jd_utc)
    t = (jd - 2451545.0) / 36525.0
    theta_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * (t**2)
        - (t**3) / 38710000.0
    )
    return float(np.deg2rad(np.mod(theta_deg, 360.0)))


def sun_position_eci_km_simple(jd_utc: float) -> np.ndarray:
    n = float(jd_utc) - 2451545.0
    l_deg = np.mod(280.460 + 0.9856474 * n, 360.0)
    g_rad = np.deg2rad(np.mod(357.528 + 0.9856003 * n, 360.0))
    lam_deg = l_deg + 1.915 * np.sin(g_rad) + 0.020 * np.sin(2.0 * g_rad)
    lam_rad = np.deg2rad(lam_deg)
    eps_rad = np.deg2rad(23.439 - 0.0000004 * n)
    r_au = 1.00014 - 0.01671 * np.cos(g_rad) - 0.00014 * np.cos(2.0 * g_rad)
    r_km = r_au * AU_KM
    x = r_km * np.cos(lam_rad)
    y = r_km * np.cos(eps_rad) * np.sin(lam_rad)
    z = r_km * np.sin(eps_rad) * np.sin(lam_rad)
    return np.array([x, y, z], dtype=float)


def moon_position_eci_km_simple(jd_utc: float) -> np.ndarray:
    n = float(jd_utc) - 2451545.0
    l0_rad = np.deg2rad(np.mod(218.316 + 13.176396 * n, 360.0))
    m_moon_rad = np.deg2rad(np.mod(134.963 + 13.064993 * n, 360.0))
    f_rad = np.deg2rad(np.mod(93.272 + 13.229350 * n, 360.0))
    lon_rad = l0_rad + np.deg2rad(6.289) * np.sin(m_moon_rad)
    lat_rad = np.deg2rad(5.128) * np.sin(f_rad)
    r_km = 385001.0 - 20905.0 * np.cos(m_moon_rad)
    x_ecl = r_km * np.cos(lat_rad) * np.cos(lon_rad)
    y_ecl = r_km * np.cos(lat_rad) * np.sin(lon_rad)
    z_ecl = r_km * np.sin(lat_rad)
    eps_rad = np.deg2rad(23.439 - 0.0000004 * n)
    x = x_ecl
    y = y_ecl * np.cos(eps_rad) - z_ecl * np.sin(eps_rad)
    z = y_ecl * np.sin(eps_rad) + z_ecl * np.cos(eps_rad)
    return np.array([x, y, z], dtype=float)


def _wrap_deg(x: float) -> float:
    return float(np.mod(x, 360.0))


def sun_position_eci_km_enhanced(jd_utc: float) -> np.ndarray:
    """
    Enhanced low-cost Sun ephemeris (Meeus-style) in mean-equator-of-date ECI.
    """
    t = (float(jd_utc) - 2451545.0) / 36525.0
    l0 = _wrap_deg(280.46646 + 36000.76983 * t + 0.0003032 * (t**2))
    m = _wrap_deg(357.52911 + 35999.05029 * t - 0.0001537 * (t**2))
    m_rad = np.deg2rad(m)
    e = 0.016708634 - 0.000042037 * t - 0.0000001267 * (t**2)
    c = (
        (1.914602 - 0.004817 * t - 0.000014 * (t**2)) * np.sin(m_rad)
        + (0.019993 - 0.000101 * t) * np.sin(2.0 * m_rad)
        + 0.000289 * np.sin(3.0 * m_rad)
    )
    true_long = _wrap_deg(l0 + c)
    true_anom_deg = _wrap_deg(m + c)
    v_rad = np.deg2rad(true_anom_deg)
    r_au = (1.000001018 * (1.0 - e * e)) / (1.0 + e * np.cos(v_rad))

    omega = _wrap_deg(125.04 - 1934.136 * t)
    lam_app_deg = true_long - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega))
    lam = np.deg2rad(lam_app_deg)

    eps0 = 23.0 + 26.0 / 60.0 + 21.448 / 3600.0 - (
        46.8150 * t + 0.00059 * (t**2) - 0.001813 * (t**3)
    ) / 3600.0
    eps = np.deg2rad(eps0 + 0.00256 * np.cos(np.deg2rad(omega)))

    r_km = r_au * AU_KM
    x = r_km * np.cos(lam)
    y = r_km * np.cos(eps) * np.sin(lam)
    z = r_km * np.sin(eps) * np.sin(lam)
    return np.array([x, y, z], dtype=float)


def moon_position_eci_km_enhanced(jd_utc: float) -> np.ndarray:
    """
    Enhanced low-cost Moon ephemeris with dominant periodic terms.
    """
    t = (float(jd_utc) - 2451545.0) / 36525.0
    l_prime = _wrap_deg(218.3164477 + 481267.88123421 * t - 0.0015786 * (t**2) + (t**3) / 538841.0 - (t**4) / 65194000.0)
    d = _wrap_deg(297.8501921 + 445267.1114034 * t - 0.0018819 * (t**2) + (t**3) / 545868.0 - (t**4) / 113065000.0)
    m = _wrap_deg(357.5291092 + 35999.0502909 * t - 0.0001536 * (t**2) + (t**3) / 24490000.0)
    m_prime = _wrap_deg(134.9633964 + 477198.8675055 * t + 0.0087414 * (t**2) + (t**3) / 69699.0 - (t**4) / 14712000.0)
    f = _wrap_deg(93.2720950 + 483202.0175233 * t - 0.0036539 * (t**2) - (t**3) / 3526000.0 + (t**4) / 863310000.0)

    d_rad = np.deg2rad(d)
    m_rad = np.deg2rad(m)
    mp_rad = np.deg2rad(m_prime)
    f_rad = np.deg2rad(f)

    lon_deg = (
        l_prime
        + 6.289 * np.sin(mp_rad)
        + 1.274 * np.sin(2.0 * d_rad - mp_rad)
        + 0.658 * np.sin(2.0 * d_rad)
        + 0.214 * np.sin(2.0 * mp_rad)
        + 0.11 * np.sin(d_rad)
    )
    lat_deg = (
        5.128 * np.sin(f_rad)
        + 0.280 * np.sin(mp_rad + f_rad)
        + 0.277 * np.sin(mp_rad - f_rad)
        + 0.173 * np.sin(2.0 * d_rad - f_rad)
        + 0.055 * np.sin(2.0 * d_rad + f_rad - mp_rad)
        + 0.046 * np.sin(2.0 * d_rad - f_rad - mp_rad)
        + 0.033 * np.sin(2.0 * d_rad + f_rad)
        + 0.017 * np.sin(2.0 * mp_rad + f_rad)
    )
    r_km = (
        385000.56
        - 20905.0 * np.cos(mp_rad)
        - 3699.0 * np.cos(2.0 * d_rad - mp_rad)
        - 2956.0 * np.cos(2.0 * d_rad)
        - 570.0 * np.cos(2.0 * mp_rad)
        + 246.0 * np.cos(2.0 * mp_rad - 2.0 * d_rad)
        - 205.0 * np.cos(m_rad - 2.0 * d_rad)
        - 171.0 * np.cos(mp_rad + 2.0 * d_rad)
    )

    lon = np.deg2rad(_wrap_deg(lon_deg))
    lat = np.deg2rad(lat_deg)
    eps0 = 23.439291 - 0.0130042 * t
    eps = np.deg2rad(eps0)

    x_ecl = r_km * np.cos(lat) * np.cos(lon)
    y_ecl = r_km * np.cos(lat) * np.sin(lon)
    z_ecl = r_km * np.sin(lat)
    x = x_ecl
    y = y_ecl * np.cos(eps) - z_ecl * np.sin(eps)
    z = y_ecl * np.sin(eps) + z_ecl * np.cos(eps)
    return np.array([x, y, z], dtype=float)


def resolve_sun_moon_positions(env: dict, t_s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve Sun and Moon inertial position vectors (km) using explicit env values,
    optional callable hook, then configured analytic mode.
    """
    if "sun_pos_eci_km" in env and "moon_pos_eci_km" in env:
        return np.array(env["sun_pos_eci_km"], dtype=float), np.array(env["moon_pos_eci_km"], dtype=float)

    jd = resolved_jd_utc(env=env, t_s=t_s)
    if jd is None:
        sun = np.array(env.get("sun_pos_eci_km", np.array([AU_KM, 0.0, 0.0], dtype=float)), dtype=float)
        moon = np.array(env.get("moon_pos_eci_km", np.array([384400.0, 0.0, 0.0], dtype=float)), dtype=float)
        return sun, moon

    eph_callable = env.get("ephemeris_callable", None)
    if callable(eph_callable):
        out = eph_callable(float(jd), env)
        if isinstance(out, dict):
            if "sun_pos_eci_km" in out and "moon_pos_eci_km" in out:
                return np.array(out["sun_pos_eci_km"], dtype=float), np.array(out["moon_pos_eci_km"], dtype=float)

    mode = str(env.get("ephemeris_mode", "analytic_enhanced")).lower()
    if mode in ("spice", "spiceypy"):
        from sim.dynamics.orbit.spice import spice_sun_moon_positions_eci_km

        return spice_sun_moon_positions_eci_km(jd, env)
    if mode in ("analytic_simple", "simple"):
        return sun_position_eci_km_simple(jd), moon_position_eci_km_simple(jd)
    return sun_position_eci_km_enhanced(jd), moon_position_eci_km_enhanced(jd)


def resolve_body_position_eci_km(body_name: str, env: dict, t_s: float) -> np.ndarray:
    name = str(body_name).strip().lower()
    key = f"{name}_pos_eci_km"
    if key in env:
        return np.array(env[key], dtype=float)

    if name in ("sun", "moon"):
        sun, moon = resolve_sun_moon_positions(env, t_s)
        return sun if name == "sun" else moon

    jd = resolved_jd_utc(env=env, t_s=t_s)
    if jd is None:
        raise RuntimeError(
            f"Body '{body_name}' requested but no position provided in env['{key}'] and no epoch is available."
        )

    mode = str(env.get("ephemeris_mode", "analytic_enhanced")).lower()
    if mode in ("spice", "spiceypy"):
        from sim.dynamics.orbit.spice import spice_body_position_eci_km

        return spice_body_position_eci_km(name, jd, env)

    cb = env.get("ephemeris_body_callable", None)
    if callable(cb):
        out = cb(name, float(jd), env)
        return np.array(out, dtype=float).reshape(3)

    raise RuntimeError(
        f"Body '{body_name}' requested but ephemeris_mode='{mode}' cannot resolve it. "
        "Use SPICE mode or provide env position / callable."
    )


def resolved_jd_utc(env: dict, t_s: float) -> float | None:
    if "jd_utc" in env:
        return float(env["jd_utc"])
    if "jd_utc_start" in env:
        return float(env["jd_utc_start"]) + float(t_s) / 86400.0
    return None


def resolve_time_dependent_env(env: dict, t_s: float) -> dict:
    out = dict(env)
    out["sim_t_s"] = float(t_s)
    jd = resolved_jd_utc(env=out, t_s=t_s)
    if jd is None:
        return out
    out["jd_utc"] = float(jd)

    mode = str(out.get("ephemeris_mode", "analytic_enhanced")).lower()
    if mode in ("analytic", "analytic_simple", "analytic_enhanced", "enhanced", "simple", "spice", "spiceypy"):
        sun, moon = resolve_sun_moon_positions(out, t_s)
        if "sun_pos_eci_km" not in out:
            out["sun_pos_eci_km"] = sun
        if "moon_pos_eci_km" not in out:
            out["moon_pos_eci_km"] = moon
        s_norm = float(np.linalg.norm(sun))
        if s_norm > 0.0 and "sun_dir_eci" not in out:
            out["sun_dir_eci"] = sun / s_norm
    elif mode in ("external", "callable"):
        sun, moon = resolve_sun_moon_positions(out, t_s)
        if "sun_pos_eci_km" not in out:
            out["sun_pos_eci_km"] = sun
        if "moon_pos_eci_km" not in out:
            out["moon_pos_eci_km"] = moon
        s_norm = float(np.linalg.norm(sun))
        if s_norm > 0.0 and "sun_dir_eci" not in out:
            out["sun_dir_eci"] = sun / s_norm
    return out
