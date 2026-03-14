from __future__ import annotations

import argparse
import math
from pathlib import Path

from sim.config import load_simulation_yaml
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
    cfg = load_simulation_yaml(args.config)

    # Show progress only for the integration loop; config parsing and plotting are excluded.
    if cfg.monte_carlo.enabled:
        out = run_master_simulation(config_path=args.config)
    else:
        total_steps = int(max(math.floor(float(cfg.simulator.duration_s) / float(cfg.simulator.dt_s)), 0))
        pbar = None
        last_step = 0
        if total_steps > 0:
            try:
                from tqdm.auto import tqdm  # type: ignore

                pbar = tqdm(total=total_steps, desc="Simulation", unit="step")
            except Exception:
                pbar = None

        def _on_step(step: int, total: int) -> None:
            nonlocal last_step, pbar
            if pbar is None:
                return
            if int(total) > 0 and int(total) != int(pbar.total):
                pbar.total = int(total)
            s = max(int(step), 0)
            if s > last_step:
                pbar.update(s - last_step)
            last_step = s

        try:
            out = run_master_simulation(config_path=args.config, step_callback=_on_step)
        finally:
            if pbar is not None:
                pbar.close()
    print("Master simulation completed:")
    for k, v in out.items():
        if k == "runs":
            print(f"  runs: {len(v)}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
