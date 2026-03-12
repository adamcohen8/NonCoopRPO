from __future__ import annotations

import argparse
from pathlib import Path

from sim.master_simulator import run_master_simulation


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Master simulation runner: one YAML config, no other inputs required.")
    parser.add_argument(
        "--config",
        default=str(repo_root / "configs" / "simulation_template.yaml"),
        help="Path to simulation YAML config.",
    )
    args = parser.parse_args()
    out = run_master_simulation(config_path=args.config)
    print("Master simulation completed:")
    for k, v in out.items():
        if k == "runs":
            print(f"  runs: {len(v)}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
