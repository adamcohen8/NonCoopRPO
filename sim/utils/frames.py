from __future__ import annotations

import numpy as np


def ric_dcm_ir_from_rv(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> np.ndarray:
    r_hat = r_eci_km / max(np.linalg.norm(r_eci_km), 1e-12)
    h = np.cross(r_eci_km, v_eci_km_s)
    c_hat = h / max(np.linalg.norm(h), 1e-12)
    i_hat = np.cross(c_hat, r_hat)
    i_hat = i_hat / max(np.linalg.norm(i_hat), 1e-12)
    return np.column_stack((r_hat, i_hat, c_hat))


def ric_angular_rate_eci_from_rv(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> np.ndarray:
    r = np.array(r_eci_km, dtype=float).reshape(3)
    v = np.array(v_eci_km_s, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    if r2 <= 1e-12:
        return np.zeros(3, dtype=float)
    return np.cross(r, v) / r2


def ric_rect_state_to_eci(
    x_rel_ric_rect: np.ndarray,
    r_chief_eci_km: np.ndarray,
    v_chief_eci_km_s: np.ndarray,
) -> np.ndarray:
    x_rel = np.array(x_rel_ric_rect, dtype=float).reshape(6)
    c_ir = ric_dcm_ir_from_rv(r_chief_eci_km, v_chief_eci_km_s)
    omega_ric_eci = ric_angular_rate_eci_from_rv(r_chief_eci_km, v_chief_eci_km_s)
    dr_eci = c_ir @ x_rel[:3]
    dv_eci = c_ir @ x_rel[3:] + np.cross(omega_ric_eci, dr_eci)
    return np.hstack(
        (
            np.array(r_chief_eci_km, dtype=float).reshape(3) + dr_eci,
            np.array(v_chief_eci_km_s, dtype=float).reshape(3) + dv_eci,
        )
    )


def eci_relative_to_ric_rect(
    x_dep_eci: np.ndarray,
    x_chief_eci: np.ndarray,
) -> np.ndarray:
    x_dep = np.array(x_dep_eci, dtype=float).reshape(6)
    x_chief = np.array(x_chief_eci, dtype=float).reshape(6)
    r_chief = x_chief[:3]
    v_chief = x_chief[3:]
    c_ir = ric_dcm_ir_from_rv(r_chief, v_chief)
    omega_ric_eci = ric_angular_rate_eci_from_rv(r_chief, v_chief)
    dr_eci = x_dep[:3] - r_chief
    dv_eci = x_dep[3:] - v_chief
    dr_ric = c_ir.T @ dr_eci
    dv_ric = c_ir.T @ (dv_eci - np.cross(omega_ric_eci, dr_eci))
    return np.hstack((dr_ric, dv_ric))


def dcm_to_euler_321(dcm: np.ndarray) -> np.ndarray:
    psi = np.arctan2(dcm[1, 0], dcm[0, 0])
    theta = -np.arcsin(np.clip(dcm[2, 0], -1.0, 1.0))
    phi = np.arctan2(dcm[2, 1], dcm[2, 2])
    return np.array([phi, theta, psi])


def ric_curv_to_rect(x_ric_curv: np.ndarray, r0_km: float, eps: float = 1e-12) -> np.ndarray:
    x_r_curv, x_i_curv, x_c_curv, x_r_curv_dot, x_i_curv_dot, x_c_curv_dot = np.array(
        x_ric_curv, dtype=float
    ).reshape(6)
    r0 = max(float(r0_km), eps)

    r = max(r0 + x_r_curv, eps)
    theta_i = x_i_curv / r0
    theta_c = x_c_curv / r0

    c_i = np.cos(theta_i)
    s_i = np.sin(theta_i)
    c_c = np.cos(theta_c)
    s_c = np.sin(theta_c)

    x = r * c_c * c_i
    y = r * c_c * s_i
    z = r * s_c

    x_r = x - r0
    x_i = y
    x_c = z

    r_dot = x_r_curv_dot
    theta_i_dot = x_i_curv_dot / r0
    theta_c_dot = x_c_curv_dot / r0

    xdot = r_dot * c_c * c_i - r * s_c * theta_c_dot * c_i - r * c_c * s_i * theta_i_dot
    ydot = r_dot * c_c * s_i - r * s_c * theta_c_dot * s_i + r * c_c * c_i * theta_i_dot
    zdot = r_dot * s_c + r * c_c * theta_c_dot

    return np.array([x_r, x_i, x_c, xdot, ydot, zdot], dtype=float)


def ric_rect_to_curv(x_ric_rect: np.ndarray, r0_km: float, eps: float = 1e-12) -> np.ndarray:
    x_r, x_i, x_c, x_rdot, x_idot, x_cdot = np.array(x_ric_rect, dtype=float).reshape(6)
    r0 = max(float(r0_km), eps)

    x = r0 + x_r
    y = x_i
    z = x_c
    r = np.sqrt(x * x + y * y + z * z)
    r = max(r, eps)
    p2 = x * x + y * y
    p = np.sqrt(max(p2, eps))

    theta_i = np.arctan2(y, x)
    theta_c = np.arctan2(z, p)

    x_r_curv = r - r0
    x_i_curv = r0 * theta_i
    x_c_curv = r0 * theta_c

    r_dot = (x * x_rdot + y * x_idot + z * x_cdot) / r
    theta_i_dot = (x * x_idot - y * x_rdot) / max(p2, eps)
    p_dot = (x * x_rdot + y * x_idot) / p
    theta_c_dot = (p * x_cdot - z * p_dot) / (r * r)

    return np.array(
        [
            x_r_curv,
            x_i_curv,
            x_c_curv,
            r_dot,
            r0 * theta_i_dot,
            r0 * theta_c_dot,
        ],
        dtype=float,
    )
