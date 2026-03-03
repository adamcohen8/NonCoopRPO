from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from presets import build_sim_object_from_presets
from sim.core.kernel import SimulationKernel
from sim.core.models import SimConfig
from sim.metrics.scoring import compute_scores


if __name__ == "__main__":
    dt_s = 2.0
    sat = build_sim_object_from_presets(
        object_id="sat_preset",
        dt_s=dt_s,
        orbit_radius_km=6778.0,
        phase_rad=0.0,
        enable_disturbances=False,
    )

    kernel = SimulationKernel(
        config=SimConfig(dt_s=dt_s, steps=600, controller_budget_ms=1.0),
        objects=[sat],
    )
    log = kernel.run()
    score = compute_scores(log)
    print("Preset quickstart run complete")
    print(f"steps: {len(log.t_s)-1}")
    print(f"controller_overruns: {score.controller_overruns['sat_preset']}")
