import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.scenarios.full_stack_demo import run_full_stack_demo
from sim.scenarios.monte_carlo import MonteCarloConfig, run_monte_carlo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo framework batch.")
    parser.add_argument(
        "--plot-mode",
        choices=["interactive", "save", "both"],
        default="interactive",
        help="Plot behavior; interactive is default.",
    )
    args = parser.parse_args()

    cfg = MonteCarloConfig(
        runs=10,
        base_seed=11,
        pos_sigma_km=0.02,
        vel_sigma_km_s=2e-4,
        threshold_km=0.1,
        object_ids=("sat_a", "sat_b"),
    )
    outputs = run_monte_carlo(
        config=cfg,
        scenario_fn=lambda seed, pos_sigma_km, vel_sigma_km_s, mc_sample=None: run_full_stack_demo(
            output_dir=f"outputs/full_stack_demo_mc/run_{seed}",
            seed=seed,
            pos_sigma_km=pos_sigma_km,
            vel_sigma_km_s=vel_sigma_km_s,
            mc_sample=mc_sample,
            plot_mode=args.plot_mode,
        ),
        output_dir="outputs/full_stack_demo_mc",
        plot_mode=args.plot_mode,
    )
    print("Monte Carlo outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
