from __future__ import annotations

import numpy as np

from ..atmosphere import EARTH_EQUATORIAL_RADIUS_KM
from ..launch import InsertionCriteria, LaunchSite, LaunchTimingMode, OrbitTarget, simulate_launch_to_insertion
from ..master_simulator import MasterScenario, MasterSimulator, RocketAgentConfig, SatelliteAgentConfig
from ..presets import create_rocket
from ..sat_params import SatParams
from ..satellite import Satellite
from ..strategies import HCWLQRStrategy
from .schema import DashboardConfig


def run_master_sim_from_config(config: DashboardConfig):
    """
    Scaffold controller: translates dashboard config into a master sim run.
    Returns the produced MasterSimLog.
    """
    config.validate()

    re_km = EARTH_EQUATORIAL_RADIUS_KM
    mu = 398600.4418

    target_alt_km = config.target_alt_km
    insertion_alt_km = target_alt_km - config.insertion_delta_alt_km

    target_orbit = OrbitTarget(
        a_km=re_km + target_alt_km,
        ecc=0.0,
        incl_deg=config.target_incl_deg,
        raan_deg=config.target_raan_deg,
        argp_deg=0.0,
        nu_deg=0.0,
    )
    insertion_orbit = OrbitTarget(
        a_km=re_km + insertion_alt_km,
        ecc=0.0,
        incl_deg=config.target_incl_deg,
        raan_deg=config.target_raan_deg,
        argp_deg=0.0,
        nu_deg=0.0,
    )

    criteria = InsertionCriteria(
        target_orbit=insertion_orbit,
        semimajor_axis_tolerance_km=20.0,
        inclination_tolerance_deg=0.2,
        eccentricity_tolerance=0.005,
    )

    launch_site = LaunchSite(
        lat_deg=config.launch_site_lat_deg,
        lon_deg=config.launch_site_lon_deg,
        altitude_km=config.launch_site_alt_km,
    )

    timing_mode = LaunchTimingMode(config.launch_timing_mode.strip().lower())

    target_params = SatParams(
        name="target",
        mu=mu,
        coe=np.array(
            [
                re_km + target_alt_km,
                0.0,
                np.deg2rad(config.target_incl_deg),
                np.deg2rad(config.target_raan_deg),
                0.0,
                0.0,
            ]
        ),
        max_accel_km_s2=0.0,
        min_accel_km_s2=0.0,
        propellant_dv_km_s=np.inf,
    )
    target = Satellite.from_params(target_params, strategy=None)

    rocket = create_rocket(config.rocket_preset, dt_s=config.dt_s)

    # Preview launch duration so main sim gets exactly rendezvous_duration_s after insertion.
    preview = simulate_launch_to_insertion(
        site=launch_site,
        rocket=rocket,
        target_orbit=target_orbit,
        insertion_criteria=criteria,
        timing_mode=timing_mode,
        satellite_name="preview_inserted",
    )
    insertion_duration_s = float(preview.t_s[-1] - preview.launch_delay_s)

    scenario = MasterScenario(
        dt_s=config.dt_s,
        pre_sim_duration_s=config.pre_sim_duration_s,
        sim_duration_s=insertion_duration_s + config.rendezvous_duration_s,
    )

    n_target_rad_s = np.sqrt(mu / ((re_km + target_alt_km) ** 3))
    inserted_strategy = HCWLQRStrategy(
        n_rad_s=n_target_rad_s,
        dt_s=scenario.dt_s,
        a_max_km_s2=config.inserted_lqr_a_max_km_s2,
        position_only=False,
    )

    sim = MasterSimulator(scenario)
    log = sim.run(
        satellites=[
            SatelliteAgentConfig(name="target", satellite=target),
        ],
        rockets=[
            RocketAgentConfig(
                name="booster_1",
                site=launch_site,
                rocket=rocket,
                target_orbit=target_orbit,
                insertion_criteria=criteria,
                launch_time_s=0.0,
                timing_mode=timing_mode,
                inserted_satellite_name="inserted_sat",
                inserted_strategy=inserted_strategy,
                observe_target_name="target",
            )
        ],
        log_path=str(config.output_log_path_abs),
    )

    return log
