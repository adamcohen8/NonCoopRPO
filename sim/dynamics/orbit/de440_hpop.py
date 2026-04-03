from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.dynamics.orbit.frames import _interp_eop

_EMRAT = 81.3005682214972154


def default_de440_coeff_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "validation"
        / "High Precision Orbit Propagator_4-2"
        / "High Precision Orbit Propagator_4.2.2"
        / "DE440Coeff.mat"
    ).resolve()


def default_hpop_eop_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "validation"
        / "High Precision Orbit Propagator_4-2"
        / "High Precision Orbit Propagator_4.2.2"
        / "EOP-All.txt"
    ).resolve()


@lru_cache(maxsize=2)
def _load_de440_coeff_matrix(path_str: str) -> np.ndarray:
    try:
        from scipy.io import loadmat  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "DE440 HPOP ephemeris mode requires scipy.io.loadmat to read DE440Coeff.mat. "
            "Install a compatible SciPy in the active environment."
        ) from exc

    path = Path(path_str).expanduser().resolve()
    mat = loadmat(path)
    coeff = mat.get("DE440Coeff")
    if coeff is None:
        raise ValueError(f"DE440Coeff.mat did not contain 'DE440Coeff': {path}")
    arr = np.asarray(coeff, dtype=float)
    if arr.ndim != 2:
        arr = np.atleast_2d(arr)
    return arr


def _cheb3d(t: float, n: int, ta: float, tb: float, cx: np.ndarray, cy: np.ndarray, cz: np.ndarray) -> np.ndarray:
    if t < ta or t > tb:
        raise ValueError("Time out of range in _cheb3d")
    tau = (2.0 * t - ta - tb) / (tb - ta)
    f1 = np.zeros(3, dtype=float)
    f2 = np.zeros(3, dtype=float)
    for i in range(int(n) - 1, 0, -1):
        old_f1 = f1.copy()
        f1 = 2.0 * tau * f1 - f2 + np.array([cx[i], cy[i], cz[i]], dtype=float)
        f2 = old_f1
    return tau * f1 - f2 + np.array([cx[0], cy[0], cz[0]], dtype=float)


def _subinterval_state(jd_tdb: float, t1: float, span_days: float, segments: int) -> tuple[int, float, float]:
    dt = float(jd_tdb - t1)
    seg_len = float(span_days) / float(segments)
    idx = int(np.floor(dt / seg_len))
    idx = max(0, min(int(segments) - 1, idx))
    jd0 = float(t1 + seg_len * idx)
    return idx, jd0, jd0 + seg_len


def _extract_axis(row: np.ndarray, starts_1b: tuple[int, ...], coeff_count: int) -> np.ndarray:
    out = []
    for s1 in starts_1b:
        s0 = int(s1) - 1
        out.append(np.asarray(row[s0 : s0 + coeff_count], dtype=float))
    return np.concatenate(out)


_BODY_SPECS: dict[str, dict[str, object]] = {
    "mercury": {"starts": (3, 17, 31, 45), "coeffs": 14, "segments": 4, "span_days": 32.0},
    "venus": {"starts": (171, 181), "coeffs": 10, "segments": 2, "span_days": 32.0},
    "earthmoon": {"starts": (231, 244), "coeffs": 13, "segments": 2, "span_days": 32.0},
    "mars": {"starts": (309,), "coeffs": 11, "segments": 1, "span_days": 32.0},
    "jupiter": {"starts": (342,), "coeffs": 8, "segments": 1, "span_days": 32.0},
    "saturn": {"starts": (366,), "coeffs": 7, "segments": 1, "span_days": 32.0},
    "uranus": {"starts": (387,), "coeffs": 6, "segments": 1, "span_days": 32.0},
    "neptune": {"starts": (405,), "coeffs": 6, "segments": 1, "span_days": 32.0},
    "pluto": {"starts": (423,), "coeffs": 6, "segments": 1, "span_days": 32.0},
    "moon": {"starts": (441, 480, 519, 558, 597, 636, 675, 714), "coeffs": 13, "segments": 8, "span_days": 32.0},
    "sun": {"starts": (753, 786), "coeffs": 11, "segments": 2, "span_days": 32.0},
}


def _eval_body(row: np.ndarray, jd_tdb: float, body_name: str) -> np.ndarray:
    spec = _BODY_SPECS[body_name]
    starts = tuple(int(v) for v in spec["starts"])  # type: ignore[index]
    coeff_count = int(spec["coeffs"])  # type: ignore[index]
    segments = int(spec["segments"])  # type: ignore[index]
    span_days = float(spec["span_days"])  # type: ignore[index]
    t1 = float(row[0])
    idx, ta, tb = _subinterval_state(jd_tdb, t1, span_days, segments)

    x_all = _extract_axis(row, starts, coeff_count)
    y_all = _extract_axis(row, tuple(s + coeff_count for s in starts), coeff_count)
    z_all = _extract_axis(row, tuple(s + 2 * coeff_count for s in starts), coeff_count)

    i0 = idx * coeff_count
    i1 = i0 + coeff_count
    return 1e3 * _cheb3d(jd_tdb, coeff_count, ta, tb, x_all[i0:i1], y_all[i0:i1], z_all[i0:i1])


def _find_coeff_row(pc: np.ndarray, jd_tdb: float) -> np.ndarray:
    mask = (pc[:, 0] <= jd_tdb) & (jd_tdb <= pc[:, 1])
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"JD_TDB {jd_tdb} is outside DE440Coeff coverage.")
    return np.asarray(pc[int(idx[0]), :], dtype=float).reshape(-1)


def mjd_tt_to_mjd_tdb(mjd_tt: float) -> float:
    t_tt = (float(mjd_tt) - 51544.5) / 36525.0
    return float(mjd_tt) + (
        0.001657 * np.sin(628.3076 * t_tt + 6.2401)
        + 0.000022 * np.sin(575.3385 * t_tt + 4.2970)
        + 0.000014 * np.sin(1256.6152 * t_tt + 6.1969)
        + 0.000005 * np.sin(606.9777 * t_tt + 4.0212)
        + 0.000005 * np.sin(52.9691 * t_tt + 0.4444)
        + 0.000002 * np.sin(21.3299 * t_tt + 5.5431)
        + 0.000010 * np.sin(628.3076 * t_tt + 4.2490)
    ) / 86400.0


def jd_utc_to_jd_tdb(jd_utc: float, eop_path: str | None = None, tai_utc_s: float | None = None) -> float:
    mjd_utc = float(jd_utc) - 2400000.5
    if tai_utc_s is None:
        if eop_path:
            _xp, _yp, _dut1, dat_s = _interp_eop(mjd_utc, eop_path)
            tai_utc_s = float(dat_s)
        else:
            tai_utc_s = 37.0
    tt_utc_s = 32.184 + float(tai_utc_s)
    mjd_tt = mjd_utc + tt_utc_s / 86400.0
    mjd_tdb = mjd_tt_to_mjd_tdb(mjd_tt)
    return float(mjd_tdb + 2400000.5)


def hpop_de440_positions_m(jd_tdb: float, coeff_path: str | Path | None = None) -> dict[str, np.ndarray]:
    path = default_de440_coeff_path() if coeff_path is None else Path(coeff_path).expanduser().resolve()
    pc = _load_de440_coeff_matrix(str(path))
    row = _find_coeff_row(pc, float(jd_tdb))

    r_earthmoon = _eval_body(row, jd_tdb, "earthmoon")
    r_moon = _eval_body(row, jd_tdb, "moon")
    r_sun = _eval_body(row, jd_tdb, "sun")
    r_mercury = _eval_body(row, jd_tdb, "mercury")
    r_venus = _eval_body(row, jd_tdb, "venus")
    r_mars = _eval_body(row, jd_tdb, "mars")
    r_jupiter = _eval_body(row, jd_tdb, "jupiter")
    r_saturn = _eval_body(row, jd_tdb, "saturn")
    r_uranus = _eval_body(row, jd_tdb, "uranus")
    r_neptune = _eval_body(row, jd_tdb, "neptune")
    r_pluto = _eval_body(row, jd_tdb, "pluto")

    emrat1 = 1.0 / (1.0 + _EMRAT)
    r_earth = r_earthmoon - emrat1 * r_moon
    return {
        "earth": r_earth,
        "sun_ssb": r_sun.copy(),
        "sun": -r_earth + r_sun,
        "moon": r_moon,
        "mercury": -r_earth + r_mercury,
        "venus": -r_earth + r_venus,
        "mars": -r_earth + r_mars,
        "jupiter": -r_earth + r_jupiter,
        "saturn": -r_earth + r_saturn,
        "uranus": -r_earth + r_uranus,
        "neptune": -r_earth + r_neptune,
        "pluto": -r_earth + r_pluto,
    }


def hpop_de440_positions_km(jd_utc: float, env: dict) -> dict[str, np.ndarray]:
    coeff_path_raw = env.get("de440_coeff_path")
    coeff_path = default_de440_coeff_path() if coeff_path_raw is None else Path(str(coeff_path_raw)).expanduser().resolve()
    eop_path_raw = env.get("de440_eop_path") or env.get("spherical_harmonics_eop_path") or env.get("drag_eop_path")
    eop_path = None if eop_path_raw is None else str(Path(str(eop_path_raw)).expanduser().resolve())
    tai_utc_raw = env.get("de440_tai_utc_s")
    jd_tdb = jd_utc_to_jd_tdb(float(jd_utc), eop_path=eop_path, tai_utc_s=None if tai_utc_raw is None else float(tai_utc_raw))
    pos_m = hpop_de440_positions_m(jd_tdb, coeff_path=coeff_path)
    return {k: v / 1e3 for k, v in pos_m.items()}
