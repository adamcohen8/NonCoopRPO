from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.optimization import PSOConfig, tune_controller_gains


def _print_result(result) -> None:
    print("Gain optimization complete.")
    print(f"  algorithm: {result.algorithm}")
    print(f"  aggregate_cost: {result.aggregate_cost:.6f}")
    print("  best_parameters:")
    for k in sorted(result.best_parameters.keys()):
        print(f"    {k}: {result.best_parameters[k]:.6f}")
    print("  per_case:")
    for c in result.per_case_results:
        print(
            f"    {c.name}: final_err_deg={c.final_error_deg:.3f}, "
            f"mean_err_deg={c.mean_error_deg:.3f}, mean_rate={c.mean_rate_norm_rad_s:.5f}, "
            f"mean_tau={c.mean_torque_nm:.6f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize attitude controller gains using PSO.")
    parser.add_argument("--algorithm", choices=["pd", "pid", "ric_pd", "ric_pid"], default="ric_pid")
    parser.add_argument(
        "--preset-cases",
        choices=["attitude_ric_nominal", "attitude_eci_nominal"],
        default="attitude_ric_nominal",
    )
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--inertia", type=float, default=0.72)
    parser.add_argument("--c1", type=float, default=1.49)
    parser.add_argument("--c2", type=float, default=1.49)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = tune_controller_gains(
        algorithm=args.algorithm,
        preset_case_set=args.preset_cases,
        optimizer="pso",
        pso_config=PSOConfig(
            particles=int(args.particles),
            iterations=int(args.iterations),
            inertia=float(args.inertia),
            cognitive=float(args.c1),
            social=float(args.c2),
        ),
        seed=int(args.seed),
    )
    _print_result(result)
