from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from presets import BASIC_CHEMICAL_BOTTOM_Z, build_sim_object_from_presets
from sim.control.orbit.impulsive import AttitudeAgnosticImpulsiveManeuverer, ThrustLimitedDeltaVManeuver


if __name__ == "__main__":
    sat = build_sim_object_from_presets(object_id="sat_impulse_thrust_demo", dt_s=2.0)
    truth = sat.truth

    available_dv_km_s = 0.06
    dv_cmd = np.array([0.12, -0.02, 0.03])
    dt_s = 20.0
    use_attitude_requirement = False
    alignment_tolerance_deg = 5.0

    thruster_position = None
    thruster_direction = None
    if use_attitude_requirement:
        thruster_position = BASIC_CHEMICAL_BOTTOM_Z.mount.position_body_m
        thruster_direction = BASIC_CHEMICAL_BOTTOM_Z.mount.thrust_direction_body

    maneuverer = AttitudeAgnosticImpulsiveManeuverer()
    new_truth, result = maneuverer.execute_delta_v_with_thrust_limit(
        truth=truth,
        maneuver=ThrustLimitedDeltaVManeuver(
            delta_v_eci_km_s=dv_cmd,
            max_thrust_n=BASIC_CHEMICAL_BOTTOM_Z.max_thrust_n,
            dt_s=dt_s,
            min_thrust_n=BASIC_CHEMICAL_BOTTOM_Z.min_impulse_bit_n_s / dt_s,
            require_attitude_alignment=use_attitude_requirement,
            thruster_position_body_m=thruster_position,
            thruster_direction_body=thruster_direction,
            alignment_tolerance_rad=np.deg2rad(alignment_tolerance_deg),
        ),
        available_delta_v_km_s=available_dv_km_s,
    )

    print("executed:", result.executed)
    print("commanded_dv_km_s:", result.commanded_delta_v_km_s)
    print("commanded_thrust_n:", result.commanded_thrust_n)
    print("min_thrust_n:", result.min_thrust_n)
    print("thrust_limited_dv_km_s:", result.thrust_limited_delta_v_km_s)
    print("applied_dv_km_s:", result.applied_delta_v_km_s)
    print("remaining_dv_km_s:", result.remaining_delta_v_km_s)
    print("alignment_ok:", result.alignment_ok)
    print("alignment_angle_deg:", None if result.alignment_angle_rad is None else np.rad2deg(result.alignment_angle_rad))
    print("old_velocity_km_s:", truth.velocity_eci_km_s)
    print("new_velocity_km_s:", new_truth.velocity_eci_km_s)
