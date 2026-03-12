from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.config import load_simulation_yaml


def run_demo(config_path: str) -> dict[str, str]:
    cfg = load_simulation_yaml(config_path)
    return {
        "scenario_name": cfg.scenario_name,
        "rocket_enabled": str(cfg.rocket.enabled),
        "chaser_enabled": str(cfg.chaser.enabled),
        "target_enabled": str(cfg.target.enabled),
        "duration_s": f"{cfg.simulator.duration_s:.3f}",
        "dt_s": f"{cfg.simulator.dt_s:.3f}",
        "mc_enabled": str(cfg.monte_carlo.enabled),
        "mc_iterations": str(cfg.monte_carlo.iterations),
        "config_path": str(Path(config_path).resolve()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and validate a simulation YAML config.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "simulation_template.yaml"),
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    out = run_demo(config_path=args.config)
    print("Simulation config load demo:")
    for k, v in out.items():
        print(f"  {k}: {v}")
