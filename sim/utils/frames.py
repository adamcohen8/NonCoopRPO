from __future__ import annotations

import numpy as np


def ric_dcm_ir_from_rv(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> np.ndarray:
    r_hat = r_eci_km / max(np.linalg.norm(r_eci_km), 1e-12)
    h = np.cross(r_eci_km, v_eci_km_s)
    c_hat = h / max(np.linalg.norm(h), 1e-12)
    i_hat = np.cross(c_hat, r_hat)
    i_hat = i_hat / max(np.linalg.norm(i_hat), 1e-12)
    return np.column_stack((r_hat, i_hat, c_hat))


def dcm_to_euler_321(dcm: np.ndarray) -> np.ndarray:
    psi = np.arctan2(dcm[1, 0], dcm[0, 0])
    theta = -np.arcsin(np.clip(dcm[2, 0], -1.0, 1.0))
    phi = np.arctan2(dcm[2, 1], dcm[2, 2])
    return np.array([phi, theta, psi])
