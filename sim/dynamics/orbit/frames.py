from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.orbit.epoch import gmst_angle_rad_from_jd

_ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)


def eci_to_ecef_rotation(t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    if jd_utc_start is None:
        theta = EARTH_ROT_RATE_RAD_S * t_s
    else:
        theta = gmst_angle_rad_from_jd(float(jd_utc_start) + float(t_s) / 86400.0)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def eci_to_ecef(r_eci_km: np.ndarray, t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start) @ r_eci_km


def ecef_to_eci(r_ecef_km: np.ndarray, t_s: float, jd_utc_start: float | None = None) -> np.ndarray:
    return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start).T @ r_ecef_km


def _maybe_import_erfa():
    try:
        return importlib.import_module("erfa")
    except Exception:
        return None


@lru_cache(maxsize=4)
def _load_eop_table(eop_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mjd = []
    xp_arcsec = []
    yp_arcsec = []
    dut1_s = []
    dat_s = []
    with Path(eop_path).expanduser().resolve().open("r", encoding="utf-8", errors="ignore") as f:
        in_data = False
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("NUM_OBSERVED_POINTS") or line.startswith("NUM_PREDICTED_POINTS"):
                in_data = True
                continue
            if not in_data or line.startswith("#") or line.startswith("VERSION") or line.startswith("UPDATED"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                mjd.append(float(parts[3]))
                xp_arcsec.append(float(parts[4]))
                yp_arcsec.append(float(parts[5]))
                dut1_s.append(float(parts[6]))
                dat_s.append(float(parts[12]))
            except ValueError:
                continue
    if not mjd:
        raise ValueError(f"No EOP records parsed from {eop_path}")
    return (
        np.array(mjd, dtype=float),
        np.array(xp_arcsec, dtype=float),
        np.array(yp_arcsec, dtype=float),
        np.array(dut1_s, dtype=float),
        np.array(dat_s, dtype=float),
    )


def _interp_eop(mjd_utc: float, eop_path: str) -> tuple[float, float, float, float]:
    mjd, xp_arcsec, yp_arcsec, dut1_s, dat_s = _load_eop_table(eop_path)
    x = float(mjd_utc)
    xp = float(np.interp(x, mjd, xp_arcsec))
    yp = float(np.interp(x, mjd, yp_arcsec))
    dut1 = float(np.interp(x, mjd, dut1_s))
    dat = float(np.interp(x, mjd, dat_s))
    return xp, yp, dut1, dat


def eci_to_ecef_rotation_hpop_like(
    t_s: float,
    jd_utc_start: float | None = None,
    eop_path: str | None = None,
) -> np.ndarray:
    if jd_utc_start is None or not eop_path:
        return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start)
    erfa = _maybe_import_erfa()
    if erfa is None:
        return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start)

    jd_utc = float(jd_utc_start) + float(t_s) / 86400.0
    xp_arcsec, yp_arcsec, dut1_s, _dat_s = _interp_eop(jd_utc - 2400000.5, eop_path)
    utc1 = float(np.floor(jd_utc))
    utc2 = float(jd_utc - utc1)
    try:
        tai1, tai2 = erfa.utctai(utc1, utc2)
        tt1, tt2 = erfa.taitt(tai1, tai2)
        ut11, ut12 = erfa.utcut1(utc1, utc2, dut1_s)
        return np.array(
            erfa.c2t06a(
                tt1,
                tt2,
                ut11,
                ut12,
                xp_arcsec * _ARCSEC_TO_RAD,
                yp_arcsec * _ARCSEC_TO_RAD,
            ),
            dtype=float,
        )
    except Exception:
        return eci_to_ecef_rotation(t_s, jd_utc_start=jd_utc_start)


def eci_to_ecef_harmonic(
    r_eci_km: np.ndarray,
    t_s: float,
    jd_utc_start: float | None = None,
    frame_model: str = "simple",
    eop_path: str | None = None,
) -> np.ndarray:
    model = str(frame_model).strip().lower()
    if model == "hpop_like":
        rot = eci_to_ecef_rotation_hpop_like(t_s, jd_utc_start=jd_utc_start, eop_path=eop_path)
        return rot @ np.array(r_eci_km, dtype=float)
    return eci_to_ecef(np.array(r_eci_km, dtype=float), t_s, jd_utc_start=jd_utc_start)


def ecef_to_eci_harmonic(
    r_ecef_km: np.ndarray,
    t_s: float,
    jd_utc_start: float | None = None,
    frame_model: str = "simple",
    eop_path: str | None = None,
) -> np.ndarray:
    model = str(frame_model).strip().lower()
    if model == "hpop_like":
        rot = eci_to_ecef_rotation_hpop_like(t_s, jd_utc_start=jd_utc_start, eop_path=eop_path)
        return rot.T @ np.array(r_ecef_km, dtype=float)
    return ecef_to_eci(np.array(r_ecef_km, dtype=float), t_s, jd_utc_start=jd_utc_start)
