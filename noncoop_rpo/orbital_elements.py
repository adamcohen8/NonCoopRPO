import numpy as np

from .constants import MU_EARTH_KM3_S2


def rot1(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    r = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ]
    )
    return r @ vec


def rot3(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    r = np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return r @ vec


def coe2rv(
    p: float,
    ecc: float,
    incl: float,
    raan: float,
    argp: float,
    nu: float,
    arglat: float = 0.0,
    truelon: float = 0.0,
    lonper: float = 0.0,
    mu: float = MU_EARTH_KM3_S2,
    small: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert classical elements to ECI r,v (km, km/s), Vallado-style.
    """
    if ecc < small:
        if (incl < small) or (abs(incl - np.pi) < small):
            argp = 0.0
            raan = 0.0
            nu = truelon
        else:
            argp = 0.0
            nu = arglat
    elif (incl < small) or (abs(incl - np.pi) < small):
        argp = lonper
        raan = 0.0

    cosnu = np.cos(nu)
    sinnu = np.sin(nu)
    if abs(p) < 1e-4:
        p = 1e-4

    temp = p / (1.0 + ecc * cosnu)
    rpqw = np.array([temp * cosnu, temp * sinnu, 0.0], dtype=float)
    vpqw = np.array(
        [
            -sinnu * np.sqrt(mu) / np.sqrt(p),
            (ecc + cosnu) * np.sqrt(mu) / np.sqrt(p),
            0.0,
        ],
        dtype=float,
    )

    r_eci = rot3(rot1(rot3(rpqw, -argp), -incl), -raan)
    v_eci = rot3(rot1(rot3(vpqw, -argp), -incl), -raan)
    return r_eci, v_eci
