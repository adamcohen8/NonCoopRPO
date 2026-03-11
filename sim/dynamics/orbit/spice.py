from __future__ import annotations

from datetime import timezone
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.epoch import julian_date_to_datetime

_LOADED_KERNELS: set[str] = set()

_DEFAULT_TARGETS = {
    "sun": "SUN",
    "moon": "MOON",
    "mercury": "MERCURY BARYCENTER",
    "venus": "VENUS BARYCENTER",
    "mars": "MARS BARYCENTER",
    "jupiter": "JUPITER BARYCENTER",
    "saturn": "SATURN BARYCENTER",
    "uranus": "URANUS BARYCENTER",
    "neptune": "NEPTUNE BARYCENTER",
    "pluto": "PLUTO BARYCENTER",
}


def _normalize_vec3(v: object) -> np.ndarray:
    arr = np.array(v, dtype=float).reshape(3)
    return arr


def _load_kernels_if_needed(kernels: list[str], sp) -> None:
    global _LOADED_KERNELS
    for k in kernels:
        p = str(Path(k).expanduser().resolve())
        if p in _LOADED_KERNELS:
            continue
        if not Path(p).exists():
            raise RuntimeError(f"SPICE kernel file not found: {p}")
        sp.furnsh(p)
        _LOADED_KERNELS.add(p)


def spice_sun_moon_positions_eci_km(jd_utc: float, env: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve Sun/Moon ECI vectors using SPICE.

    Supported env settings:
    - spice_ephemeris_callable(jd_utc, env) -> {"sun_pos_eci_km": ..., "moon_pos_eci_km": ...}
    - spice_kernels: list[str] paths to kernels (.bsp/.tls/.tpc etc)
    - spice_frame: default "J2000"
    - spice_abcorr: default "NONE"
    - spice_observer: default "EARTH"
    - spice_sun_target: default "SUN"
    - spice_moon_target: default "MOON"
    """
    cb = env.get("spice_ephemeris_callable", None)
    if callable(cb):
        out = cb(float(jd_utc), env)
        if isinstance(out, dict) and "sun_pos_eci_km" in out and "moon_pos_eci_km" in out:
            return _normalize_vec3(out["sun_pos_eci_km"]), _normalize_vec3(out["moon_pos_eci_km"])
        raise RuntimeError(
            "spice_ephemeris_callable must return dict with 'sun_pos_eci_km' and 'moon_pos_eci_km'."
        )

    try:
        import spiceypy as sp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "SPICE mode requested but spiceypy is unavailable. Install 'spiceypy' "
            "or provide env['spice_ephemeris_callable']."
        ) from exc

    kernels = [str(k) for k in env.get("spice_kernels", [])]
    if not kernels:
        raise RuntimeError(
            "SPICE mode requested but no kernels provided. Set env['spice_kernels'] "
            "or provide env['spice_ephemeris_callable']."
        )
    _load_kernels_if_needed(kernels, sp)

    frame = str(env.get("spice_frame", "J2000"))
    abcorr = str(env.get("spice_abcorr", "NONE"))
    observer = str(env.get("spice_observer", "EARTH"))
    sun_target = str(env.get("spice_sun_target", "SUN"))
    moon_target = str(env.get("spice_moon_target", "MOON"))

    dt_utc = julian_date_to_datetime(float(jd_utc)).astimezone(timezone.utc)
    et = float(sp.str2et(dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")))

    sun_state, _ = sp.spkezr(sun_target, et, frame, abcorr, observer)
    moon_state, _ = sp.spkezr(moon_target, et, frame, abcorr, observer)
    return _normalize_vec3(sun_state[:3]), _normalize_vec3(moon_state[:3])


def spice_body_position_eci_km(body_name: str, jd_utc: float, env: dict) -> np.ndarray:
    name = str(body_name).strip().lower()
    cb = env.get("spice_body_ephemeris_callable", None)
    if callable(cb):
        out = cb(name, float(jd_utc), env)
        return _normalize_vec3(out)

    try:
        import spiceypy as sp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "SPICE body position requested but spiceypy is unavailable. Install 'spiceypy' "
            "or provide env['spice_body_ephemeris_callable']."
        ) from exc

    kernels = [str(k) for k in env.get("spice_kernels", [])]
    if not kernels:
        raise RuntimeError(
            "SPICE body position requested but no kernels provided. Set env['spice_kernels'] "
            "or provide env['spice_body_ephemeris_callable']."
        )
    _load_kernels_if_needed(kernels, sp)

    frame = str(env.get("spice_frame", "J2000"))
    abcorr = str(env.get("spice_abcorr", "NONE"))
    observer = str(env.get("spice_observer", "EARTH"))
    target_map = dict(_DEFAULT_TARGETS)
    target_map.update({str(k).strip().lower(): str(v) for k, v in env.get("spice_body_target_map", {}).items()})
    target = target_map.get(name)
    if target is None:
        raise RuntimeError(f"Unsupported SPICE body name '{body_name}'.")

    dt_utc = julian_date_to_datetime(float(jd_utc)).astimezone(timezone.utc)
    et = float(sp.str2et(dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")))
    state, _ = sp.spkezr(target, et, frame, abcorr, observer)
    return _normalize_vec3(state[:3])


def spice_supported_body_names() -> tuple[str, ...]:
    return tuple(sorted(_DEFAULT_TARGETS.keys()))
