import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly: `python examples/TwoBurn_FirstLeg_Harness.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noncoop_rpo import (
    EARTH_EQUATORIAL_RADIUS_KM,
    MU_EARTH_KM3_S2,
    Scenario,
    SatParams,
    Satellite,
    Simulator,
    TwoBurnFirstLegStrategy,
    eci2hcw_curv,
    hcw2eci,
    ric_curv_to_rect,
)


def specific_energy(x: np.ndarray, mu: float = MU_EARTH_KM3_S2) -> float:
    r = x[0:3]
    v = x[3:6]
    return 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test harness for TwoBurnFirstLegStrategy.")
    parser.add_argument("--no-show", action="store_true", help="Run simulation without opening plot windows.")
    parser.add_argument("--dt", type=float, default=1.0, help="Simulation time step (s).")
    parser.add_argument("--sim-min", type=float, default=120.0, help="Simulation duration (minutes).")
    parser.add_argument("--tof-min", type=float, default=35.0, help="Two-burn first-leg targeting TOF (minutes).")
    parser.add_argument("--init-r-km", type=float, default=0.0, help="Initial radial offset R in curvilinear RIC (km).")
    parser.add_argument("--init-i-km", type=float, default=-100.0, help="Initial in-track arc offset I in curvilinear RIC (km).")
    parser.add_argument("--init-c-km", type=float, default=0.0, help="Initial cross-track arc offset C in curvilinear RIC (km).")
    args = parser.parse_args()

    mu = MU_EARTH_KM3_S2
    re_km = EARTH_EQUATORIAL_RADIUS_KM

    # Target orbit setup.
    target_alt_km = 500.0
    target_a_km = re_km + target_alt_km
    target_coe = np.array(
        [
            target_a_km,
            0.0,
            np.deg2rad(25.0),
            np.deg2rad(20.0),
            0.0,
            0.0,
        ],
        dtype=float,
    )

    target_params = SatParams(name="target", mu=mu, coe=target_coe)
    x_target0 = target_params.initial_eci_state()

    x0_rel_curv = np.array([args.init_r_km, args.init_i_km, args.init_c_km, 0.0, 0.0, 0.0], dtype=float)
    x0_rel_rect = ric_curv_to_rect(x0_rel_curv, r0=np.linalg.norm(x_target0[:3]))
    x_chaser0 = hcw2eci(x_target0, x0_rel_rect)

    n_target = np.sqrt(mu / (target_a_km**3))
    strategy = TwoBurnFirstLegStrategy(
        n_rad_s=n_target,
        time_of_flight_s=args.tof_min * 60.0,
        burn_time_constant_s=180.0,
    )

    chaser_params = SatParams(
        name="chaser",
        mu=mu,
        max_accel_km_s2=5e-6,
        min_accel_km_s2=0.0,
        propellant_dv_km_s=0.3,
        r0_eci_km=x_chaser0[:3],
        v0_eci_km_s=x_chaser0[3:],
    )

    target = Satellite.from_params(target_params, strategy=None)
    chaser = Satellite.from_params(chaser_params, strategy=strategy)

    steps = 2*int(np.ceil((args.sim_min * 60.0) / args.dt))
    log = Simulator(Scenario(dt_s=args.dt, steps=steps)).run(target, chaser)

    rel_norm0 = np.linalg.norm(log.rel_ric[0, :3])
    rel_normf = np.linalg.norm(log.rel_ric[-1, :3])
    e_t = np.array([specific_energy(x, mu) for x in log.target_x_eci])
    e_c = np.array([specific_energy(x, mu) for x in log.chaser_x_eci])

    print("=== TwoBurn First-Leg Harness ===")
    print("Initial rel state (curv RIC):", x0_rel_curv)
    print("Initial |rel| (km):", rel_norm0)
    print("Final   |rel| (km):", rel_normf)
    print("Min     |rel| (km):", float(np.min(np.linalg.norm(log.rel_ric[:, :3], axis=1))))
    idx_tof = min(len(log.t_s) - 1, int(round((args.tof_min * 60.0) / max(args.dt, 1e-9))))
    print("State at first-leg TOF (km): R,I,C =", log.rel_ric[idx_tof, 0], log.rel_ric[idx_tof, 1], log.rel_ric[idx_tof, 2])
    print("Chaser dv used (m/s):", 1000.0 * chaser.state.dv_used_km_s)
    print("Chaser dv remaining (m/s):", 1000.0 * chaser.state.dv_remaining_km_s)
    print("Target energy drift (km^2/s^2):", float(np.max(e_t) - np.min(e_t)))
    print("Chaser energy drift (km^2/s^2):", float(np.max(e_c) - np.min(e_c)))

    fig1 = plt.figure(figsize=(9, 6))
    plt.plot(log.t_s / 60.0, log.rel_ric[:, 0], label="R")
    plt.plot(log.t_s / 60.0, log.rel_ric[:, 1], label="I")
    plt.plot(log.t_s / 60.0, log.rel_ric[:, 2], label="C")
    plt.xlabel("Time (min)")
    plt.ylabel("Relative Position (km)")
    plt.title("Relative Position (Curvilinear RIC)")
    plt.grid(True)
    plt.legend()

    fig2 = plt.figure(figsize=(9, 6))
    plt.plot(log.t_s / 60.0, log.rel_ric[:, 3] * 1000.0, label="dR")
    plt.plot(log.t_s / 60.0, log.rel_ric[:, 4] * 1000.0, label="dI")
    plt.plot(log.t_s / 60.0, log.rel_ric[:, 5] * 1000.0, label="dC")
    plt.xlabel("Time (min)")
    plt.ylabel("Relative Velocity (m/s)")
    plt.title("Relative Velocity (Curvilinear RIC)")
    plt.grid(True)
    plt.legend()

    fig3 = plt.figure(figsize=(8, 6))
    plt.plot(log.rel_ric[:, 1], log.rel_ric[:, 0], color="tab:orange")
    plt.xlabel("In-track I (km)")
    plt.ylabel("Radial R (km)")
    plt.title("Relative Trajectory (I vs R)")
    plt.grid(True)

    fig4 = plt.figure(figsize=(9, 6))
    plt.plot(log.t_s / 60.0, log.u_ric[:, 0], label="u_R")
    plt.plot(log.t_s / 60.0, log.u_ric[:, 1], label="u_I")
    plt.plot(log.t_s / 60.0, log.u_ric[:, 2], label="u_C")
    plt.plot(log.t_s / 60.0, log.u_mag, label="|u|", linewidth=2.0)
    plt.xlabel("Time (min)")
    plt.ylabel("Acceleration (km/s^2)")
    plt.title("TwoBurn First-Leg Control Input")
    plt.grid(True)
    plt.legend()

    fig5 = plt.figure(figsize=(9, 6))
    plt.plot(log.t_s / 60.0, log.dv_cum_km_s * 1000.0)
    plt.xlabel("Time (min)")
    plt.ylabel("Cumulative Control Integral (m/s)")
    plt.title("Approximate DV Usage")
    plt.grid(True)

    if args.no_show:
        plt.close("all")
        return

    plt.show()


if __name__ == "__main__":
    main()
