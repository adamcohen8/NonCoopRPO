import sys
from pathlib import Path

import numpy as np

# Allow running this file directly: `python examples/Master_Simulator_Example.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noncoop_rpo import (
    EARTH_EQUATORIAL_RADIUS_KM,
    InsertionCriteria,
    LaunchSite,
    LaunchTimingMode,
    MasterScenario,
    MasterSimulator,
    OrbitTarget,
    Rocket,
    RocketAgentConfig,
    SatParams,
    Satellite,
    SatelliteAgentConfig,
    TwoBurnFirstLegStrategy,
    simulate_launch_to_insertion,
)


def main() -> None:
    re_km = EARTH_EQUATORIAL_RADIUS_KM
    log_path = str(REPO_ROOT / "outputs" / "master_sim_log.npz")

    mu = 398600.4418
    target_alt_km = 350.0
    insertion_alt_km = target_alt_km - 10.0
    target_incl_deg = 0.0
    launch_site = LaunchSite(lat_deg=0.0, lon_deg=-52.768, altitude_km=0.0)

    target_params = SatParams(
        name="target",
        mu=mu,
        coe=np.array([re_km + target_alt_km, 0.0, np.deg2rad(target_incl_deg), np.deg2rad(20.0), 0.0, 0.0]),
        max_accel_km_s2=0.0,
        min_accel_km_s2=0.0,
        propellant_dv_km_s=np.inf,
    )
    target = Satellite.from_params(target_params, strategy=None)

    rocket = Rocket(
        isp_s=440.0,
        dry_mass_kg=45000.0,
        fuel_mass_kg=420000.0,
        thrust_newton=5.2e6,
        cd=0.3,
        area_m2=18.0,
        vertical_ascent_time_s=140.0,
        pitch_over_time_s=320.0,
        guidance_mode="simple",
        use_feedback_guidance=False,
        guidance_gain_energy=10.0,
        guidance_gain_vr=6.0,
        insertion_hold_time_s=6.0,
        max_flight_time_s=60.0 * 90.0,
        dt_s=1.0,
        simple_insertion_check_period_s=10.0,
    )

    launch_target_orbit = OrbitTarget(
        a_km=re_km + target_alt_km,
        ecc=0.0,
        incl_deg=target_incl_deg,
        raan_deg=20.0,
        argp_deg=0.0,
        nu_deg=0.0,
    )
    insertion_criteria = InsertionCriteria(
        target_orbit=OrbitTarget(
            a_km=re_km + insertion_alt_km,
            ecc=0.0,
            incl_deg=target_incl_deg,
            raan_deg=20.0,
            argp_deg=0.0,
            nu_deg=0.0,
        ),
        semimajor_axis_tolerance_km=20.0,
        inclination_tolerance_deg=0.2,
        eccentricity_tolerance=0.005,
    )

    # We end the sim 2 hours after insertion. Since sim time starts at launch,
    # include launch-to-insertion duration as an offset.
    preview = simulate_launch_to_insertion(
        site=launch_site,
        rocket=rocket,
        target_orbit=launch_target_orbit,
        insertion_criteria=insertion_criteria,
        timing_mode=LaunchTimingMode.OPTIMAL,
        satellite_name="preview_inserted",
    )
    insertion_duration_s = float(preview.t_s[-1] - preview.launch_delay_s)
    rendezvous_budget_after_insertion_s = 2.0 * 3600.0

    scenario = MasterScenario(
        dt_s=1.0,
        pre_sim_duration_s=6.0 * 3600.0,
        sim_duration_s=insertion_duration_s + rendezvous_budget_after_insertion_s,
        terminate_within_distance_km=0.05,  # 50 meters
        terminate_primary_agent="inserted_sat",
        terminate_secondary_agent="target",
    )
    sim = MasterSimulator(scenario)

    a_target_km = re_km + target_alt_km
    n_target_rad_s = np.sqrt(mu / (a_target_km**3))
    inserted_first_burn = TwoBurnFirstLegStrategy(
        n_rad_s=n_target_rad_s,
        time_of_flight_s=35.0 * 60.0,
        burn_time_constant_s=180.0,
    )

    log = sim.run(
        satellites=[
            SatelliteAgentConfig(name="target", satellite=target),
        ],
        rockets=[
            RocketAgentConfig(
                name="booster_1",
                site=launch_site,
                rocket=rocket,
                target_orbit=launch_target_orbit,
                insertion_criteria=insertion_criteria,
                launch_time_s=0.0,
                timing_mode=LaunchTimingMode.OPTIMAL,
                inserted_satellite_name="inserted_sat",
                inserted_strategy=inserted_first_burn,
                observe_target_name="target",
            )
        ],
        log_path=log_path,
    )

    print("Sim complete.")
    print("Saved log:", log_path)
    print("Total samples:", len(log.t_s))
    print("Target orbit altitude (km):", target_alt_km)
    print("Target inclination (deg):", target_incl_deg)
    print("Insertion orbit altitude (km):", insertion_alt_km)
    print("Rendezvous budget after insertion (s):", rendezvous_budget_after_insertion_s)
    print("Launch events:", len(log.launches))
    for evt in log.launches:
        print(
            f"  {evt.rocket_name}: launch@{evt.scheduled_launch_time_s:.1f}s -> "
            f"inserted@{evt.insertion_time_s:.1f}s ({evt.insertion_reason})"
        )

    for name, x_hist in log.x_eci_by_agent.items():
        valid = np.isfinite(x_hist[:, 0])
        if not np.any(valid):
            print(f"{name}: never active")
            continue
        x_final = x_hist[np.where(valid)[0][-1], :]
        alt_km = np.linalg.norm(x_final[:3]) - re_km
        print(f"{name}: final altitude {alt_km:.2f} km")

    x_target = log.x_eci_by_agent["target"]
    x_chaser = log.x_eci_by_agent["inserted_sat"]
    valid_pair = np.isfinite(x_target[:, 0]) & np.isfinite(x_chaser[:, 0])
    if np.any(valid_pair):
        sep_km = np.linalg.norm(x_chaser[valid_pair, :3] - x_target[valid_pair, :3], axis=1)
        print("Inserted-vs-target min separation (km):", float(np.min(sep_km)))
        print("Inserted-vs-target final separation (km):", float(sep_km[-1]))


if __name__ == "__main__":
    main()
