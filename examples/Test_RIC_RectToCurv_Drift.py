import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly: `python examples/Test_RIC_RectToCurv_Drift.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noncoop_rpo.constants import MU_EARTH_KM3_S2
from noncoop_rpo.dynamics import two_body_deriv
from noncoop_rpo.frames import eci2hcw, eci2hcw_curv, hcw2eci, ric_rect_to_curv
from noncoop_rpo.integrators import rk4_step
from noncoop_rpo.orbital_elements import coe2rv


def main() -> None:
    mu = MU_EARTH_KM3_S2

    re_km = 6378.137
    alt_km = 500.0
    a_km = re_km + alt_km
    ecc = 0.0
    p_km = a_km
    incl = np.deg2rad(51.6)
    raan = 0.0
    argp = 0.0
    nu = 0.0

    r_t, v_t = coe2rv(p_km, ecc, incl, raan, argp, nu, arglat=nu, truelon=nu, lonper=0.0, mu=mu)
    x_target = np.hstack((r_t, v_t))

    # Start 10 km behind target and +10 m/s along-track in rectangular RIC.
    x0_rel_rect = np.array([0.0, -100.0, 0.0, 0.0, 0.01, 0.0], dtype=float)
    x_chaser = hcw2eci(x_target, x0_rel_rect)

    n = np.sqrt(mu / a_km**3)
    t_orbit = 2.0 * np.pi / n
    total_t = 10.0 * t_orbit
    dt = 10.0
    steps = int(np.ceil(total_t / dt))

    t_log = np.zeros(steps + 1)
    rel_rect_log = np.zeros((steps + 1, 6))
    rel_curv_log = np.zeros((steps + 1, 6))
    rel_curv_direct_log = np.zeros((steps + 1, 6))
    conv_err_log = np.zeros(steps + 1)

    for k in range(steps + 1):
        rel_rect = eci2hcw(x_target, x_chaser)
        r0 = np.linalg.norm(x_target[:3])
        rel_curv = ric_rect_to_curv(rel_rect, r0=r0)
        rel_curv_direct = eci2hcw_curv(x_target, x_chaser)

        t_log[k] = k * dt
        rel_rect_log[k, :] = rel_rect
        rel_curv_log[k, :] = rel_curv
        rel_curv_direct_log[k, :] = rel_curv_direct
        conv_err_log[k] = np.linalg.norm(rel_curv - rel_curv_direct)

        if k < steps:
            x_target = rk4_step(two_body_deriv, x_target, dt, mu, None)
            x_chaser = rk4_step(two_body_deriv, x_chaser, dt, mu, None)

    print("Rect->Curv conversion max ||difference||:", np.max(conv_err_log))
    print("Initial curvilinear along-track I (km):", rel_curv_log[0, 1])
    print("Final curvilinear along-track I (km):", rel_curv_log[-1, 1])
    print("Net along-track change over 10 orbits (km):", rel_curv_log[-1, 1] - rel_curv_log[0, 1])

    fig1 = plt.figure()
    plt.plot(t_log / t_orbit, rel_rect_log[:, 1], label="Rect I (km)")
    plt.plot(t_log / t_orbit, rel_curv_log[:, 1], label="Curv I (km)")
    plt.xlabel("Time (orbits)")
    plt.ylabel("Along-track relative position (km)")
    plt.title("Along-track Drift: 10 km Behind, +10 m/s Along-track")
    plt.grid(True)
    plt.legend()

    fig2 = plt.figure()
    plt.plot(t_log / t_orbit, conv_err_log)
    plt.xlabel("Time (orbits)")
    plt.ylabel("||ric_rect_to_curv - eci2hcw_curv||")
    plt.title("Rect->Curv Conversion Consistency")
    plt.grid(True)

    fig3 = plt.figure()
    plt.plot(rel_curv_log[:, 1], rel_curv_log[:, 0], label="Curv I (km)")
    plt.plot(rel_rect_log[:, 1], rel_rect_log[:, 0], label="Rect I (km)")
    plt.xlabel("Curvilinear I (km)")
    plt.ylabel("Curvilinear R (km)")
    plt.title("Curvilinear R vs I")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
