from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noncoop_rpo.dashboard import DashboardConfig, launch_dashboard, run_master_sim_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Master Simulator dashboard scaffold.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one simulation with default dashboard config without opening the GUI.",
    )
    args = parser.parse_args()

    if args.dry_run:
        cfg = DashboardConfig()
        log = run_master_sim_from_config(cfg)
        print("Dry run complete.")
        print(f"Saved log: {cfg.output_log_path_abs}")
        print(f"Samples: {len(log.t_s)}")
        print(f"Termination: {log.termination_reason}")
        return

    launch_dashboard()


if __name__ == "__main__":
    main()
