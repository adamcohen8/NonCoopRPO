from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import ecef_to_eci, eci_to_ecef


@dataclass(frozen=True)
class SphericalHarmonicTerm:
    """
    Single (n, m) spherical harmonic term.

    Notes:
    - Uses unnormalized associated Legendre polynomials P_n^m.
    - Coefficients C_nm and S_nm are interpreted in the same (unnormalized) convention.
    """

    n: int
    m: int
    c_nm: float
    s_nm: float = 0.0

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("n must be >= 2 for perturbation terms.")
        if self.m < 0 or self.m > self.n:
            raise ValueError("m must satisfy 0 <= m <= n.")


def _double_factorial(k: int) -> float:
    if k <= 0:
        return 1.0
    out = 1.0
    for i in range(k, 0, -2):
        out *= float(i)
    return out


def _associated_legendre_unnormalized(n: int, m: int, x: float) -> float:
    if m < 0 or m > n:
        return 0.0
    x = float(np.clip(x, -1.0, 1.0))
    # P_m^m
    p_mm = ((-1.0) ** m) * _double_factorial(2 * m - 1) * (max(0.0, 1.0 - x * x) ** (0.5 * m))
    if n == m:
        return p_mm
    # P_{m+1}^m
    p_m1m = x * (2 * m + 1) * p_mm
    if n == m + 1:
        return p_m1m
    p_nm2 = p_mm
    p_nm1 = p_m1m
    p_nm = p_nm1
    for ell in range(m + 2, n + 1):
        p_nm = ((2 * ell - 1) * x * p_nm1 - (ell + m - 1) * p_nm2) / (ell - m)
        p_nm2 = p_nm1
        p_nm1 = p_nm
    return p_nm


def _term_potential_ecef_km2_s2(
    r_ecef_km: np.ndarray,
    mu_km3_s2: float,
    re_km: float,
    term: SphericalHarmonicTerm,
) -> float:
    x, y, z = np.array(r_ecef_km, dtype=float)
    r = float(np.linalg.norm(r_ecef_km))
    if r <= 0.0:
        return 0.0
    sin_phi = z / r
    lon = float(np.arctan2(y, x))
    p_nm = _associated_legendre_unnormalized(term.n, term.m, sin_phi)
    amp = term.c_nm * np.cos(term.m * lon) + term.s_nm * np.sin(term.m * lon)
    return float(mu_km3_s2 / r * (re_km / r) ** term.n * p_nm * amp)


def accel_spherical_harmonics_terms(
    r_eci_km: np.ndarray,
    t_s: float,
    terms: list[SphericalHarmonicTerm],
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    re_km: float = EARTH_RADIUS_KM,
    fd_step_km: float = 1e-3,
) -> np.ndarray:
    """
    Acceleration in ECI from arbitrary spherical-harmonic terms (n,m).

    Uses finite-difference gradient of the perturbing potential in ECEF.
    """
    if not terms:
        return np.zeros(3)
    if fd_step_km <= 0.0:
        raise ValueError("fd_step_km must be positive.")

    r_ecef = eci_to_ecef(np.array(r_eci_km, dtype=float), float(t_s))

    def _u_at(pos_ecef_km: np.ndarray) -> float:
        u = 0.0
        for term in terms:
            u += _term_potential_ecef_km2_s2(pos_ecef_km, mu_km3_s2, re_km, term)
        return float(u)

    h = float(fd_step_km)
    grad_ecef = np.zeros(3)
    for i in range(3):
        d = np.zeros(3)
        d[i] = h
        up = _u_at(r_ecef + d)
        um = _u_at(r_ecef - d)
        grad_ecef[i] = (up - um) / (2.0 * h)

    # Gravity acceleration equals gradient of potential.
    return ecef_to_eci(grad_ecef, float(t_s))


def parse_spherical_harmonic_terms(raw_terms: list[dict] | None) -> list[SphericalHarmonicTerm]:
    if not raw_terms:
        return []
    out: list[SphericalHarmonicTerm] = []
    for i, item in enumerate(raw_terms):
        if not isinstance(item, dict):
            raise ValueError(f"spherical harmonic term index {i} must be a dict.")
        n = int(item.get("n", -1))
        m = int(item.get("m", -1))
        c_nm = float(item.get("c_nm", item.get("c", 0.0)))
        s_nm = float(item.get("s_nm", item.get("s", 0.0)))
        out.append(SphericalHarmonicTerm(n=n, m=m, c_nm=c_nm, s_nm=s_nm))
    return out
