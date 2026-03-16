from __future__ import annotations

import argparse
import math
from pathlib import Path

from sim.config import load_simulation_yaml
from sim.master_simulator import run_master_simulation


def _fmt_float(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def _print_single_run_summary(out: dict) -> None:
    run = dict(out.get("run", {}) or {})
    thrust = dict(run.get("thrust_stats", {}) or {})
    print("")
    print("=" * 72)
    print("MASTER SIMULATION COMPLETED")
    print("=" * 72)
    print(f"Config     : {out.get('config_path', '')}")
    print(f"Scenario   : {out.get('scenario_name', run.get('scenario_name', 'unknown'))}")
    print(f"Objects    : {', '.join(run.get('objects', []))}")
    print(
        f"Timing     : samples={run.get('samples', 0)}, "
        f"dt={_fmt_float(float(run.get('dt_s', 0.0)), 3)} s, "
        f"duration={_fmt_float(float(run.get('duration_s', 0.0)), 1)} s"
    )
    if bool(run.get("terminated_early", False)):
        print(
            "Termination: EARLY "
            f"(reason={run.get('termination_reason')}, "
            f"t={run.get('termination_time_s')}, object={run.get('termination_object_id')})"
        )
    else:
        print("Termination: nominal")
    if "rocket_insertion_achieved" in run:
        if bool(run.get("rocket_insertion_achieved", False)):
            print(f"Insertion  : achieved at t={run.get('rocket_insertion_time_s')}")
        else:
            print("Insertion  : not achieved")

    if thrust:
        print("-" * 72)
        print("Thrust Stats")
        print(f"{'Object':<14}{'Burn Samples':>14}{'Max Accel (km/s^2)':>24}{'Total dV (m/s)':>18}")
        for oid in sorted(thrust.keys()):
            s = dict(thrust.get(oid, {}) or {})
            print(
                f"{oid:<14}"
                f"{int(s.get('burn_samples', 0)):>14d}"
                f"{float(s.get('max_accel_km_s2', 0.0)):>24.3e}"
                f"{float(s.get('total_dv_m_s', 0.0)):>18.3f}"
            )
    print("=" * 72)


def _print_monte_carlo_summary(out: dict) -> None:
    runs = list(out.get("runs", []) or [])
    print("")
    print("=" * 72)
    print("MASTER MONTE CARLO COMPLETED")
    print("=" * 72)
    print(f"Config     : {out.get('config_path', '')}")
    print(f"Scenario   : {out.get('scenario_name', 'unknown')}")
    print(f"Iterations : {len(runs)}")
    if runs:
        durations = [float(dict(r.get("summary", {}) or {}).get("duration_s", 0.0)) for r in runs]
        print(f"Duration   : min={min(durations):.1f}s  max={max(durations):.1f}s")
    print("=" * 72)


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
    if bool(out.get("monte_carlo", {}).get("enabled", False)):
        _print_monte_carlo_summary(out)
    else:
        _print_single_run_summary(out)


if __name__ == "__main__":
    main()
