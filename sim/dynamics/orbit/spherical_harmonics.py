from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import urllib.request

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
    normalized: bool = False

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
    if term.normalized:
        p_nm *= _fully_normalized_legendre_scale(term.n, term.m)
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


def _fully_normalized_legendre_scale(n: int, m: int) -> float:
    # sqrt((2-delta_0m)*(2n+1)*(n-m)!/(n+m)!)
    delta = 1.0 if m == 0 else 0.0
    log_scale = (
        math.log((2.0 - delta) * (2.0 * n + 1.0))
        + math.lgamma(n - m + 1.0)
        - math.lgamma(n + m + 1.0)
    )
    return float(math.exp(0.5 * log_scale))


def parse_spherical_harmonic_terms(raw_terms: list[dict | SphericalHarmonicTerm] | None) -> list[SphericalHarmonicTerm]:
    if not raw_terms:
        return []
    out: list[SphericalHarmonicTerm] = []
    for i, item in enumerate(raw_terms):
        if isinstance(item, SphericalHarmonicTerm):
            out.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(f"spherical harmonic term index {i} must be a dict or SphericalHarmonicTerm.")
        n = int(item.get("n", -1))
        m = int(item.get("m", -1))
        c_nm = float(item.get("c_nm", item.get("c", 0.0)))
        s_nm = float(item.get("s_nm", item.get("s", 0.0)))
        normalized = bool(item.get("normalized", False))
        out.append(SphericalHarmonicTerm(n=n, m=m, c_nm=c_nm, s_nm=s_nm, normalized=normalized))
    return out


def load_icgem_gfc_terms(
    gfc_path: str | Path,
    max_degree: int,
    max_order: int | None = None,
) -> list[SphericalHarmonicTerm]:
    """
    Load real gravity coefficients from an ICGEM-style .gfc file.

    The parser supports coefficients on `gfc` lines:
      gfc n m Cnm Snm ...
    and respects the `norm` header when present.
    """
    path = Path(gfc_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Gravity coefficient file not found: {path}")

    n_max = int(max_degree)
    if n_max < 2:
        raise ValueError("max_degree must be >= 2.")
    m_max = n_max if max_order is None else int(max_order)
    if m_max < 0:
        raise ValueError("max_order must be >= 0.")

    norm_kind = "unknown"
    out: list[SphericalHarmonicTerm] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            parts = s.split()
            key = parts[0].lower()
            if key == "norm" and len(parts) >= 2:
                norm_kind = parts[1].strip().lower()
                continue
            if key != "gfc" or len(parts) < 5:
                continue
            n = int(parts[1])
            m = int(parts[2])
            if n < 2 or n > n_max or m < 0 or m > min(n, m_max):
                continue
            c_nm = float(parts[3])
            s_nm = float(parts[4])
            out.append(
                SphericalHarmonicTerm(
                    n=n,
                    m=m,
                    c_nm=c_nm,
                    s_nm=s_nm,
                    normalized=("unnormalized" not in norm_kind),
                )
            )

    if not out:
        raise ValueError(
            f"No usable spherical harmonic terms found in {path} for n<= {n_max}, m<= {m_max}."
        )
    return out


def load_hpop_ggm03_terms(
    coeff_path: str | Path,
    max_degree: int,
    max_order: int | None = None,
    normalized: bool = True,
) -> list[SphericalHarmonicTerm]:
    """
    Load HPOP-style GGM03 coefficient table (e.g., GGM03C.txt).

    Expected row format:
      n  m  Cnm  Snm  sigmaC  sigmaS
    """
    path = Path(coeff_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"HPOP gravity coefficient file not found: {path}")

    n_max = int(max_degree)
    if n_max < 2:
        raise ValueError("max_degree must be >= 2.")
    m_max = n_max if max_order is None else int(max_order)
    if m_max < 0:
        raise ValueError("max_order must be >= 0.")

    out: list[SphericalHarmonicTerm] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                n = int(parts[0])
                m = int(parts[1])
                c_nm = float(parts[2])
                s_nm = float(parts[3])
            except ValueError:
                continue
            if n < 2 or n > n_max or m < 0 or m > min(n, m_max):
                continue
            out.append(
                SphericalHarmonicTerm(
                    n=n,
                    m=m,
                    c_nm=c_nm,
                    s_nm=s_nm,
                    normalized=bool(normalized),
                )
            )

    if not out:
        raise ValueError(f"No usable GGM03 terms found in {path} for n<= {n_max}, m<= {m_max}.")
    return out


_REAL_MODEL_URLS = {
    "EGM96": [
        # SatelliteToolboxGravityModels.jl documented direct ICGEM link.
        "https://icgem.gfz-potsdam.de/getmodel/gfc/971b0a3b49a497910aad23cd85e066d4cd9af0aeafe7ce6301a696bed8570be3/EGM96.gfc",
    ]
}


def _download_model_file(model: str, outpath: Path) -> None:
    urls = _REAL_MODEL_URLS.get(model.upper(), [])
    last_err: Exception | None = None
    for url in urls:
        try:
            outpath.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(outpath))
            if outpath.exists() and outpath.stat().st_size > 0:
                return
        except Exception as exc:
            last_err = exc
    if last_err is not None:
        raise RuntimeError(f"Failed downloading gravity model '{model}' to {outpath}: {last_err}") from last_err
    raise RuntimeError(f"No download URL configured for gravity model '{model}'.")


@lru_cache(maxsize=16)
def _cached_real_terms(
    model: str,
    coeff_path: str | None,
    max_degree: int,
    max_order: int | None,
    allow_download: bool,
) -> tuple[SphericalHarmonicTerm, ...]:
    if coeff_path:
        path = Path(coeff_path).expanduser().resolve()
    else:
        cache_dir = Path.home() / ".noncooprpo" / "gravity_models"
        path = cache_dir / f"{model.upper()}.gfc"
        if not path.exists():
            if not allow_download:
                raise FileNotFoundError(
                    f"Real gravity coefficients requested but file not found: {path}. "
                    "Provide spherical_harmonics_coeff_path or enable download."
                )
            _download_model_file(model=model, outpath=path)

    terms = load_icgem_gfc_terms(path, max_degree=max_degree, max_order=max_order)
    return tuple(terms)


def load_real_earth_gravity_terms(
    max_degree: int,
    max_order: int | None = None,
    model: str = "EGM96",
    coeff_path: str | None = None,
    allow_download: bool = True,
) -> list[SphericalHarmonicTerm]:
    """
    Load real Earth gravity terms from an external coefficient file/model.

    Returns real (not synthetic) coefficients suitable for spherical harmonics propagation.
    """
    return list(
        _cached_real_terms(
            model=str(model).upper(),
            coeff_path=None if coeff_path is None else str(Path(coeff_path).expanduser().resolve()),
            max_degree=int(max_degree),
            max_order=None if max_order is None else int(max_order),
            allow_download=bool(allow_download),
        )
    )
