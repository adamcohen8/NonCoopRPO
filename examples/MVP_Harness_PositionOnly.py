import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly: `python examples/MVP_Harness_PositionOnly.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noncoop_rpo.constants import MU_EARTH_KM3_S2
from noncoop_rpo.frames import eci2hcw_curv, hcw2eci, ric_curv_to_rect
from noncoop_rpo.hcw_control import build_hcw_lqr_position_only
from noncoop_rpo.sat_params import SatParams
from noncoop_rpo.satellite import Satellite
from noncoop_rpo.sim import Scenario, Simulator


def specific_energy(x: np.ndarray, mu: float = MU_EARTH_KM3_S2) -> float:
    r = x[0:3]
    v = x[3:6]
    return 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)


def main() -> None:
    mu = MU_EARTH_KM3_S2
    re_km = 6378.137
    alt_km = 500.0
    a_km = re_km + alt_km

    target_coe = np.array(
        [
            a_km,  # p (circular)
            0.0,  # ecc
            np.deg2rad(51.6),  # incl
            np.deg2rad(0.0),  # raan
            np.deg2rad(0.0),  # argp
            np.deg2rad(0.0),  # nu
        ]
    )

    target_params = SatParams(name="target", mu=mu, coe=target_coe)
    x_target0 = target_params.initial_eci_state()

    x0_rel_ric_curv = np.array([0.0, 1.0, 0.1, 0.0, 0.0, 0.0], dtype=float)
    x0_rel_ric_rect = ric_curv_to_rect(x0_rel_ric_curv, r0=np.linalg.norm(x_target0[:3]))
    x_chaser0 = hcw2eci(x_target0, x0_rel_ric_rect)

    x_rel_check = eci2hcw_curv(x_target0, x_chaser0)
    print("Initial rel state (curv RIC) requested: ", x0_rel_ric_curv)
    print("Initial rel state (RIC) computed : ", x_rel_check)
    print("Initial RIC round-trip error     : ", x_rel_check - x0_rel_ric_curv)

    n = np.sqrt(mu / a_km**3)
    t_orbit = 2 * np.pi / n
    dt = 1.0
    steps = int(np.ceil(t_orbit / dt))
    a_max = 1e-6
    lqr = build_hcw_lqr_position_only(n, dt, a_max)

    def chaser_policy(_t: float, x_rel_ric: np.ndarray, _x_eci: np.ndarray) -> np.ndarray:
        x_model = x_rel_ric.copy()
        x_model[1] *= -1.0
        x_model[4] *= -1.0
        return lqr(x_model)

    chaser_params = SatParams(
        name="chaser",
        mu=mu,
        max_accel_km_s2=a_max,
        min_accel_km_s2=0.0,
        propellant_dv_km_s=0.05,
        r0_eci_km=x_chaser0[:3],
        v0_eci_km_s=x_chaser0[3:],
    )

    target = Satellite.from_params(target_params, policy=None)
    chaser = Satellite.from_params(chaser_params, policy=chaser_policy)
    log = Simulator(Scenario(dt_s=dt, steps=steps)).run(target, chaser)

    e_t = np.array([specific_energy(x, mu) for x in log.target_x_eci])
    e_c = np.array([specific_energy(x, mu) for x in log.chaser_x_eci])
    print("\nEnergy drift target (max-min):", np.max(e_t) - np.min(e_t), " (km^2/s^2)")
    print("Energy drift chaser (max-min):", np.max(e_c) - np.min(e_c), " (km^2/s^2)")

    fig1 = plt.figure()
    plt.plot(log.t_s / 60.0, log.rel_ric[:, 0], label="R (km)")
    plt.plot(log.t_s / 60.0, log.rel_ric[:, 1], label="I (km)")
    plt.plot(log.t_s / 60.0, log.rel_ric[:, 2], label="C (km)")
    plt.xlabel("Time (min)")
    plt.ylabel("Relative Position (km)")
    plt.title("Relative Position in Curvilinear RIC vs Time (2-body truth)")
    plt.grid(True)
    plt.legend()

    fig2 = plt.figure()
    plt.plot(log.t_s / 60.0, log.rel_ric[:, 3], label="dR (km/s)")
    plt.plot(log.t_s / 60.0, log.rel_ric[:, 4], label="dI (km/s)")
    plt.plot(log.t_s / 60.0, log.rel_ric[:, 5], label="dC (km/s)")
    plt.xlabel("Time (min)")
    plt.ylabel("Relative Velocity (km/s)")
    plt.title("Relative Velocity in Curvilinear RIC vs Time (2-body truth)")
    plt.grid(True)
    plt.legend()

    fig3 = plt.figure()
    plt.plot(log.rel_ric[:, 1] * 1000.0, log.rel_ric[:, 0] * 1000.0)
    plt.scatter(log.rel_ric[1, 1] * 1000.0, log.rel_ric[1, 0] * 1000.0, c="green")
    plt.scatter(log.rel_ric[-1, 1] * 1000.0, log.rel_ric[-1, 0] * 1000.0, c="red")
    plt.xlabel("In-track I (m)")
    plt.ylabel("Radial R (m)")
    plt.title("Relative Trajectory (Curvilinear R vs I)")
    plt.grid(True)

    fig4 = plt.figure()
    plt.plot(log.t_s / 60, log.u_ric[:, 0], label="u_R")
    plt.plot(log.t_s / 60, log.u_ric[:, 1], label="u_I")
    plt.plot(log.t_s / 60, log.u_ric[:, 2], label="u_C")
    plt.xlabel("Time (min)")
    plt.ylabel("Acceleration (km/s^2)")
    plt.title("Control Input (RIC)")
    plt.legend()
    plt.grid(True)

    fig5 = plt.figure()
    plt.plot(log.t_s / 60, log.u_mag)
    plt.xlabel("Time (min)")
    plt.ylabel("|u| (km/s^2)")
    plt.title("Control Magnitude")
    plt.grid(True)

    fig6 = plt.figure()
    plt.plot(log.t_s / 60, log.dv_cum_km_s * 1000.0)
    plt.xlabel("Time (min)")
    plt.ylabel("Cumulative Δv (m/s)")
    plt.title("Total Δv Expended")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
