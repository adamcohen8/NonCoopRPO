import numpy as np

from .orbital_elements import coe2rv


def mag(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def unit(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return x / n


def eci2hcw(x_host: np.ndarray, x_eci: np.ndarray) -> np.ndarray:
    """
    Convert absolute ECI host/chaser states into relative RIC state.
    """
    r_host = x_host[0:3]
    v_host = x_host[3:6]
    r_ch = x_eci[0:3]
    v_ch = x_eci[3:6]

    dr = r_ch - r_host
    dv = v_ch - v_host

    h_host = np.cross(r_host, v_host)
    in_vec = np.cross(h_host, r_host)
    rsw = np.column_stack((unit(r_host), unit(in_vec), unit(h_host)))

    rtemp = np.cross(h_host, v_host)
    vtemp = np.cross(h_host, r_host)
    drsw = np.column_stack((v_host / mag(r_host), rtemp / mag(vtemp), np.zeros(3)))

    x_hcw_r = rsw.T @ dr
    frame_mvmnt = np.array(
        [
            x_hcw_r[0] * (r_host @ v_host) / (mag(r_host) ** 2),
            x_hcw_r[1] * (vtemp @ rtemp) / (mag(vtemp) ** 2),
            0.0,
        ]
    )
    x_hcw_v = (rsw.T @ dv) + (drsw.T @ dr) - frame_mvmnt
    return np.hstack((x_hcw_r, x_hcw_v))


def hcw2eci(x_host: np.ndarray, x_hcw: np.ndarray) -> np.ndarray:
    """
    Convert relative RIC state to absolute ECI state.
    """
    r_host = x_host[0:3]
    v_host = x_host[3:6]
    x_hcw_r = x_hcw[0:3]
    x_hcw_v = x_hcw[3:6]

    h_host = np.cross(r_host, v_host)
    in_vec = np.cross(h_host, r_host)
    rsw = np.column_stack((unit(r_host), unit(in_vec), unit(h_host)))
    dr = np.linalg.inv(rsw.T) @ x_hcw_r

    rtemp = np.cross(h_host, v_host)
    vtemp = np.cross(h_host, r_host)
    drsw = np.column_stack((v_host / mag(r_host), rtemp / mag(vtemp), np.zeros(3)))

    frame_mvmnt = np.array(
        [
            x_hcw_r[0] * (r_host @ v_host) / (mag(r_host) ** 2),
            x_hcw_r[1] * (vtemp @ rtemp) / (mag(vtemp) ** 2),
            0.0,
        ]
    )
    dv = np.linalg.inv(rsw.T) @ (x_hcw_v + frame_mvmnt - (drsw.T @ dr))

    x_eci = np.zeros(6)
    x_eci[0:3] = dr + r_host
    x_eci[3:6] = dv + v_host
    return x_eci


def ric_rect_to_curv(x_ric_rect: np.ndarray, r0: float, eps: float = 1e-12) -> np.ndarray:
    """
    Rectangular RIC -> Curvilinear RIC.
    x_ric_rect = [xR, xI, xC, xRdot, xIdot, xCdot] (km, km/s)
    r0 = |r_chief| (km)
    """
    x_r, x_i, x_c, x_rdot, x_idot, x_cdot = x_ric_rect

    u = r0 + x_r
    v = x_i
    rho = np.hypot(u, v)
    rho = max(rho, eps)
    theta = np.arctan2(v, u)

    x_r_curv = rho - r0
    x_i_curv = r0 * theta
    x_c_curv = x_c

    udot = x_rdot
    vdot = x_idot
    rho_dot = (u * udot + v * vdot) / rho
    theta_dot = (u * vdot - v * udot) / (rho * rho)

    x_r_curv_dot = rho_dot
    x_i_curv_dot = r0 * theta_dot
    x_c_curv_dot = x_cdot

    return np.array([x_r_curv, x_i_curv, x_c_curv, x_r_curv_dot, x_i_curv_dot, x_c_curv_dot], dtype=float)


def ric_curv_to_rect(x_ric_curv: np.ndarray, r0: float, eps: float = 1e-12) -> np.ndarray:
    """
    Curvilinear RIC -> Rectangular RIC.
    x_ric_curv = [xRcurv, xIcurv, xC, xRcurv_dot, xIcurv_dot, xCdot] (km, km/s)
    r0 = |r_chief| (km)
    """
    x_r_curv, x_i_curv, x_c, x_r_curv_dot, x_i_curv_dot, x_cdot = x_ric_curv

    rho = r0 + x_r_curv
    rho = max(rho, eps)
    theta = x_i_curv / r0

    u = rho * np.cos(theta)
    v = rho * np.sin(theta)

    x_r = u - r0
    x_i = v

    rho_dot = x_r_curv_dot
    theta_dot = x_i_curv_dot / r0

    udot = rho_dot * np.cos(theta) - rho * np.sin(theta) * theta_dot
    vdot = rho_dot * np.sin(theta) + rho * np.cos(theta) * theta_dot

    x_rdot = udot
    x_idot = vdot
    return np.array([x_r, x_i, x_c, x_rdot, x_idot, x_cdot], dtype=float)


def eci2hcw_curv(x_host: np.ndarray, x_eci: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert absolute ECI host/chaser states directly to curvilinear RIC.
    """
    x_rect = eci2hcw(x_host, x_eci)
    r0 = np.linalg.norm(x_host[0:3])
    return ric_rect_to_curv(x_rect, r0=r0, eps=eps)


def eci_to_rsw_dcm(r_host: np.ndarray, v_host: np.ndarray) -> np.ndarray:
    """
    Returns DCM whose columns are [R, S, W] unit vectors.
    """
    h = np.cross(r_host, v_host)
    in_vec = np.cross(h, r_host)
    r_hat = r_host / np.linalg.norm(r_host)
    s_hat = in_vec / np.linalg.norm(in_vec)
    w_hat = h / np.linalg.norm(h)
    return np.column_stack((r_hat, s_hat, w_hat))
