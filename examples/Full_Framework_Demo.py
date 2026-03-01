import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.scenarios.full_stack_demo import run_full_stack_demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full framework demo.")
    parser.add_argument(
        "--plot-mode",
        choices=["interactive", "save", "both"],
        default="interactive",
        help="Plot behavior; interactive is default.",
    )
    args = parser.parse_args()

    result = run_full_stack_demo(plot_mode=args.plot_mode)
    print("Completed full framework demo")
    print(f"output_dir: {result['output_dir']}")
    print(f"min_separation_km: {result['metrics'].min_separation_km:.3f}")
