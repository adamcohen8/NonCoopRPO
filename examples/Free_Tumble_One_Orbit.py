import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.scenarios.free_tumble_one_orbit import run_free_tumble_one_orbit
from sim.config import profile_choices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one-orbit free tumble scenario.")
    parser.add_argument(
        "--plot-mode",
        choices=["interactive", "save", "both"],
        default="interactive",
        help="Plot behavior: interactive is default for IDE display.",
    )
    parser.add_argument(
        "--profile",
        choices=list(profile_choices()),
        default="ops",
        help="Fidelity profile: fast, ops, or high_fidelity.",
    )
    args = parser.parse_args()

    outputs = run_free_tumble_one_orbit(plot_mode=args.plot_mode, profile=args.profile)
    print("Generated outputs:")
    for key, path in outputs.items():
        if path:
            print(f"  {key}: {path}")
