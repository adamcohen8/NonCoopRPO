import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly: `python examples/Cowell_Drag_Burn_Example.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noncoop_rpo.constants import MU_EARTH_KM3_S2
from noncoop_rpo.cowell import DragConfig, make_constant_burn, make_drag_perturbation, propagate_cowell
from noncoop_rpo.orbital_elements import coe2rv


def specific_energy(x: np.ndarray, mu: float = MU_EARTH_KM3_S2) -> float:
    r = x[:3]
    v = x[3:]
    return 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)


def main() -> None:
    mu = MU_EARTH_KM3_S2

    re_km = 6378.137
    alt_km = 300.0
    a_km = re_km + alt_km

    r0, v0 = coe2rv(
        p=a_km,
        ecc=0.0,
        incl=np.deg2rad(51.6),
        raan=0.0,
        argp=0.0,
        nu=0.0,
        arglat=0.0,
        truelon=0.0,
        lonper=0.0,
        mu=mu,
    )
    x0 = np.hstack((r0, v0))

    dt = 5.0
    t_orbit = 2.0 * np.pi * np.sqrt(a_km**3 / mu)
    steps = int(np.ceil(2.0 * t_orbit / dt))

    drag = make_drag_perturbation(
        DragConfig(
            cd=2.2,
            area_m2=2.0,
            mass_kg=300.0,
        )
    )

    # Example finite burn in +V direction for first 120 seconds.
    vhat0 = v0 / np.linalg.norm(v0)
    burn = make_constant_burn(accel_eci_km_s2=2.0e-7 * vhat0, start_s=0.0, stop_s=120.0)

    result = propagate_cowell(
        x0_eci=x0,
        dt_s=dt,
        steps=steps,
        mu=mu,
        perturbations=[drag],
        burn_fn=burn,
    )

    e_log = np.array([specific_energy(x, mu) for x in result.x_eci])
    r_norm = np.linalg.norm(result.x_eci[:, :3], axis=1)
    drag_mag = np.linalg.norm(result.a_pert_total_eci, axis=1)
    burn_mag = np.linalg.norm(result.a_burn_eci, axis=1)

    print("Initial radius (km):", r_norm[0])
    print("Final radius (km):", r_norm[-1])
    print("Initial specific energy (km^2/s^2):", e_log[0])
    print("Final specific energy (km^2/s^2):", e_log[-1])

    fig1 = plt.figure()
    plt.plot(result.t_s / 60.0, r_norm)
    plt.xlabel("Time (min)")
    plt.ylabel("|r| (km)")
    plt.title("Cowell Propagation with Drag + Burn")
    plt.grid(True)

    fig2 = plt.figure()
    plt.plot(result.t_s / 60.0, e_log - e_log[0])
    plt.xlabel("Time (min)")
    plt.ylabel("Specific energy change (km^2/s^2)")
    plt.title("Energy Change")
    plt.grid(True)

    fig3 = plt.figure()
    plt.plot(result.t_s / 60.0, drag_mag, label="|a_drag|")
    plt.plot(result.t_s / 60.0, burn_mag, label="|a_burn|")
    plt.xlabel("Time (min)")
    plt.ylabel("Acceleration (km/s^2)")
    plt.title("Perturbation Acceleration Magnitudes")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
