import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly: `python examples/Launch_To_Insertion_RPO_Example.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noncoop_rpo import (
    EARTH_EQUATORIAL_RADIUS_KM,
    InsertionCriteria,
    LaunchSite,
    LaunchTimingMode,
    OrbitTarget,
    Rocket,
    eci2hcw_curv,
    propagate_cowell,
    simulate_launch_to_insertion,
)


def main() -> None:
    re_km = EARTH_EQUATORIAL_RADIUS_KM

    # Target satellite orbit input (RPO reference target).
    target_orbit = OrbitTarget(
        a_km=re_km + 350.0,
        ecc=0.001,
        incl_deg=51.6,
        raan_deg=40.0,
        argp_deg=0.0,
        nu_deg=40.0,
    )

    # Desired insertion orbit: lower altitude and behind target in anomaly.
    insertion_target = OrbitTarget(
        a_km=re_km + 340.0,
        ecc=0.0,
        incl_deg=51.6,
        raan_deg=40.0,
        argp_deg=0.0,
        nu_deg=20.0,
    )
    criteria = InsertionCriteria(
        target_orbit=insertion_target,
        semimajor_axis_tolerance_km=25.0,
        inclination_tolerance_deg=0.2,
        eccentricity_tolerance=0.002,
        flight_path_angle_target_deg=90.0,
        flight_path_angle_tolerance_deg=2.0,
        min_periapsis_altitude_km=330.0,
    )

    launch_site = LaunchSite(
        lat_deg=28.5618571,
        lon_deg=-80.577366,
        altitude_km=0.0,
    )

    rocket = Rocket(
        # Approximate hydrolox SSTO-concept values (still simplified for this model).
        isp_s=440.0,
        dry_mass_kg=45000.0,
        fuel_mass_kg=420000.0,
        thrust_newton=5.2e6,
        cd=0.3,
        area_m2=18.0,
        vertical_ascent_time_s=40.0,
        pitch_over_time_s=220.0,
        guidance_gain_energy=10.0,
        guidance_gain_vr=6.0,
        insertion_hold_time_s=6.0,
        max_flight_time_s=60.0*90.0,
        dt_s=1.0,
    )

    launch_result = simulate_launch_to_insertion(
        site=launch_site,
        rocket=rocket,
        target_orbit=target_orbit,
        insertion_criteria=criteria,
        timing_mode=LaunchTimingMode.OPTIMAL,
        satellite_name="rpo_inserted_sat",
    )

    x_target_insert = launch_result.target_state_at_insertion_eci
    x_insert = launch_result.satellite.state.x_eci

    target_period_s = 2.0 * np.pi * np.sqrt((target_orbit.a_km**3) / 398600.4418)
    drift_dt_s = 1.0
    drift_steps = int(np.ceil(target_period_s / drift_dt_s))

    target_prop = propagate_cowell(
        x0_eci=x_target_insert,
        dt_s=drift_dt_s,
        steps=drift_steps*2,
        terminate_below_radius_km=re_km,
    )
    inserted_prop = propagate_cowell(
        x0_eci=x_insert,
        dt_s=drift_dt_s,
        steps=drift_steps*2,
        terminate_below_radius_km=re_km,
    )

    rel_len = min(len(target_prop.t_s), len(inserted_prop.t_s))
    rel_curv = np.zeros((rel_len, 6), dtype=float)
    for k in range(rel_len):
        rel_curv[k, :] = eci2hcw_curv(target_prop.x_eci[k, :], inserted_prop.x_eci[k, :])

    print("Launch delay (s):", launch_result.launch_delay_s)
    print("Plane error at launch (deg):", launch_result.plane_error_deg_at_launch)
    print("Fuel-feasible heuristic:", launch_result.feasible_by_delta_v_check)
    print("Estimated required delta-v (km/s):", launch_result.estimated_required_delta_v_km_s)
    print("Estimated available delta-v (km/s):", launch_result.estimated_available_delta_v_km_s)
    print(
        "Estimated delta-v margin (km/s):",
        launch_result.estimated_available_delta_v_km_s - launch_result.estimated_required_delta_v_km_s,
    )
    print("Insertion achieved:", launch_result.inserted)
    print("Insertion reason:", launch_result.insertion_reason)
    print("Achieved a (km):", launch_result.achieved_a_km)
    print("Achieved e:", launch_result.achieved_ecc)
    print("Achieved i (deg):", launch_result.achieved_incl_deg)
    print("Remaining fuel (kg):", launch_result.fuel_mass_kg[-1])
    burn_duration_s = rocket.fuel_mass_kg / rocket.mdot_kg_s()
    pitch_complete_s = rocket.vertical_ascent_time_s + rocket.pitch_over_time_s
    print("Burn duration (s):", burn_duration_s)
    print("Pitch complete by (s):", pitch_complete_s)
    r_insert = x_insert[0:3]
    v_insert = x_insert[3:6]
    print("Insertion R ECI (km):", r_insert)
    print("Insertion |R| (km):", np.linalg.norm(r_insert))
    print("Insertion V ECI (km/s):", v_insert)
    print("Insertion |V| (km/s):", np.linalg.norm(v_insert))
    rv_angle_deg = np.rad2deg(
        np.arccos(np.clip(np.dot(r_insert, v_insert) / (np.linalg.norm(r_insert) * np.linalg.norm(v_insert)), -1.0, 1.0))
    )
    print("Insertion angle(R,V) (deg):", rv_angle_deg)
    print("Target coast terminated early:", target_prop.terminated_early, target_prop.termination_reason)
    print("Inserted coast terminated early:", inserted_prop.terminated_early, inserted_prop.termination_reason)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    traj = launch_result.x_eci[:, 0:3]
    sat_drift_eci = inserted_prop.x_eci[:, 0:3]
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="Rocket Trajectory", color="tab:blue")
    ax.plot(
        sat_drift_eci[:, 0],
        sat_drift_eci[:, 1],
        sat_drift_eci[:, 2],
        label="Post-Switch Satellite Drift",
        color="tab:orange",
        linestyle="--",
    )
    ax.scatter(
        traj[0, 0], traj[0, 1], traj[0, 2], color="green", s=50, label="Start", depthshade=False
    )
    ax.scatter(
        traj[-1, 0], traj[-1, 1], traj[-1, 2], color="red", s=50, label="End", depthshade=False
    )

    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    x_sphere = re_km * np.outer(np.cos(u), np.sin(v))
    y_sphere = re_km * np.outer(np.sin(u), np.sin(v))
    z_sphere = re_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.25, linewidth=0.0, color="tab:gray")

    ax.set_xlabel("X ECI (km)")
    ax.set_ylabel("Y ECI (km)")
    ax.set_zlabel("Z ECI (km)")
    ax.set_title("Rocket Ascent Trajectory in ECI")
    ax.legend()
    ax.set_box_aspect((1.0, 1.0, 1.0))

    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(rel_curv[:, 1], rel_curv[:, 0], color="tab:orange")
    plt.scatter(rel_curv[0, 1], rel_curv[0, 0], color="green", s=40, label="Start")
    plt.scatter(rel_curv[-1, 1], rel_curv[-1, 0], color="red", s=40, label="End")
    plt.xlabel("In-track (km)")
    plt.ylabel("Radial (km)")
    plt.title("Post-Insertion Natural Drift (Curvilinear RIC)\nOne Target Orbital Period")
    plt.grid(True)
    plt.legend()

    fig2b = plt.figure(figsize=(8, 6))
    plt.plot(rel_curv[:, 2], rel_curv[:, 0], color="tab:purple")
    plt.scatter(rel_curv[0, 2], rel_curv[0, 0], color="green", s=40, label="Start")
    plt.scatter(rel_curv[-1, 2], rel_curv[-1, 0], color="red", s=40, label="End")
    plt.xlabel("Cross-track (km)")
    plt.ylabel("Radial (km)")
    plt.title("Post-Insertion Natural Drift (Radial vs Cross-track)")
    plt.grid(True)
    plt.legend()

    rocket_t_rel_s = launch_result.t_s - launch_result.launch_delay_s
    rocket_alt_km = np.linalg.norm(launch_result.x_eci[:, 0:3], axis=1) - re_km
    sat_t_rel_s = rocket_t_rel_s[-1] + inserted_prop.t_s
    sat_alt_km = np.linalg.norm(inserted_prop.x_eci[:, 0:3], axis=1) - re_km

    fig3 = plt.figure(figsize=(9, 6))
    handoff_t_min = rocket_t_rel_s[-1] / 60.0
    plt.plot(
        rocket_t_rel_s / 60.0,
        rocket_alt_km,
        color="tab:blue",
        linewidth=2.8,
        linestyle="--",
        label="Rocket Altitude (Ascent)",
        zorder=3,
    )
    plt.plot(
        sat_t_rel_s / 60.0,
        sat_alt_km,
        color="tab:orange",
        linewidth=2.2,
        linestyle="-",
        label="Satellite Altitude (Post-Switch)",
        zorder=2,
    )
    plt.axvline(
        handoff_t_min,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Rocket-to-Satellite Handoff",
        zorder=1,
    )
    plt.scatter(rocket_t_rel_s[0] / 60.0, rocket_alt_km[0], color="green", s=35, label="Start", zorder=4)
    plt.scatter(sat_t_rel_s[-1] / 60.0, sat_alt_km[-1], color="red", s=35, label="End", zorder=4)
    plt.xlabel("Time Since Liftoff (min)")
    plt.ylabel("Altitude (km)")
    plt.title("Rocket and Satellite Altitude vs Time")
    plt.grid(True)
    plt.legend()

    fig4 = plt.figure(figsize=(9, 5))
    plt.plot(rocket_t_rel_s / 60.0, launch_result.thrust_newton / 1e6, color="tab:red", linewidth=2.0)
    plt.xlabel("Time Since Liftoff (min)")
    plt.ylabel("Thrust (MN)")
    plt.title("Rocket Thrust vs Time")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
