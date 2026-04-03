from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.presets import build_sim_object_from_presets
from sim.control.orbit.impulsive import AttitudeAgnosticImpulsiveManeuverer, ImpulsiveManeuver


if __name__ == "__main__":
    sat = build_sim_object_from_presets(object_id="sat_impulse_demo", dt_s=2.0)
    truth = sat.truth

    available_dv_km_s = 0.12
    target_velocity = truth.velocity_eci_km_s + np.array([0.08, -0.03, 0.01])

    maneuverer = AttitudeAgnosticImpulsiveManeuverer()
    new_truth, result = maneuverer.execute(
        truth=truth,
        maneuver=ImpulsiveManeuver(desired_velocity_eci_km_s=target_velocity),
        available_delta_v_km_s=available_dv_km_s,
    )

    print("executed:", result.executed)
    print("required_dv_km_s:", result.required_delta_v_km_s)
    print("remaining_dv_km_s:", result.remaining_delta_v_km_s)
    print("old_velocity_km_s:", truth.velocity_eci_km_s)
    print("new_velocity_km_s:", new_truth.velocity_eci_km_s)
