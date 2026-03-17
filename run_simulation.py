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
    guardrails = dict(run.get("attitude_guardrail_stats", {}) or {})
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
    if guardrails:
        hits = int(sum(int(v) for v in guardrails.values()))
        print("-" * 72)
        print(f"Guardrails : attitude_events={hits}")
    print("=" * 72)


def _print_monte_carlo_summary(out: dict) -> None:
    runs = list(out.get("runs", []) or [])
    agg_stats = dict(out.get("aggregate_stats", {}) or {})
    guardrail_event_totals = [
        int(sum(int(v) for v in dict(dict(r.get("summary", {}) or {}).get("attitude_guardrail_stats", {})).values()))
        for r in runs
    ]
    print("")
    print("=" * 72)
    print("MASTER MONTE CARLO COMPLETED")
    print("=" * 72)
    print(f"Config     : {out.get('config_path', '')}")
    print(f"Scenario   : {out.get('scenario_name', 'unknown')}")
    print(f"Iterations : {len(runs)}")
    if agg_stats:
        d_min = float(agg_stats.get("duration_s_min", 0.0))
        d_mean = float(agg_stats.get("duration_s_mean", 0.0))
        d_max = float(agg_stats.get("duration_s_max", 0.0))
        t_rate = float(agg_stats.get("terminated_early_rate", 0.0))
        print(f"Duration   : min={d_min:.1f}s  mean={d_mean:.1f}s  max={d_max:.1f}s")
        print(f"Early Term : {100.0 * t_rate:.1f}%")
        ca_min = agg_stats.get("closest_approach_km_min")
        ca_mean = agg_stats.get("closest_approach_km_mean")
        ca_max = agg_stats.get("closest_approach_km_max")
        if all(v is not None for v in (ca_min, ca_mean, ca_max)):
            try:
                print(
                    "Closest App: "
                    f"min={float(ca_min):.3f} km  mean={float(ca_mean):.3f} km  max={float(ca_max):.3f} km"
                )
            except (TypeError, ValueError):
                pass
        by_obj = dict(agg_stats.get("by_object", {}) or {})
        if by_obj:
            print("-" * 72)
            print("Object Stats")
            print(f"{'Object':<14}{'Mean dV (m/s)':>16}{'Min dV':>12}{'Max dV':>12}{'Mean Burns':>14}")
            for oid in sorted(by_obj.keys()):
                s = dict(by_obj.get(oid, {}) or {})
                print(
                    f"{oid:<14}"
                    f"{float(s.get('total_dv_m_s_mean', 0.0)):>16.3f}"
                    f"{float(s.get('total_dv_m_s_min', 0.0)):>12.3f}"
                    f"{float(s.get('total_dv_m_s_max', 0.0)):>12.3f}"
                    f"{float(s.get('burn_samples_mean', 0.0)):>14.1f}"
                )
    elif runs:
        durations = [float(dict(r.get("summary", {}) or {}).get("duration_s", 0.0)) for r in runs]
        print(f"Duration   : min={min(durations):.1f}s  max={max(durations):.1f}s")
    if guardrail_event_totals:
        print(f"Guardrails : mean={sum(guardrail_event_totals)/len(guardrail_event_totals):.1f}  max={max(guardrail_event_totals)}")
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
        mc_total = int(max(cfg.monte_carlo.iterations, 0))
        mc_bar = None
        sim_bar = None
        started_runs = 0
        last_step = 0
        run_done = True
        try:
            from tqdm.auto import tqdm  # type: ignore

            if mc_total > 0:
                mc_bar = tqdm(total=mc_total, desc="Monte Carlo", unit="run")

            def _on_mc_step(step: int, total: int) -> None:
                nonlocal sim_bar, started_runs, last_step, run_done, mc_bar
                s = max(int(step), 0)
                t = max(int(total), 0)
                if s == 0:
                    started_runs += 1
                    run_done = False
                    last_step = 0
                    if sim_bar is not None:
                        sim_bar.close()
                    sim_bar = tqdm(
                        total=t,
                        desc=f"Simulation {started_runs}/{max(mc_total, 1)}",
                        unit="step",
                        leave=False,
                    )
                    if t == 0:
                        if mc_bar is not None:
                            mc_bar.update(1)
                        run_done = True
                        if sim_bar is not None:
                            sim_bar.close()
                            sim_bar = None
                    return

                if sim_bar is None:
                    sim_bar = tqdm(total=t, desc=f"Simulation {started_runs}/{max(mc_total, 1)}", unit="step", leave=False)
                if t > 0 and int(sim_bar.total) != t:
                    sim_bar.total = t
                if s > last_step:
                    sim_bar.update(s - last_step)
                last_step = s

                if t > 0 and s >= t and not run_done:
                    if mc_bar is not None:
                        mc_bar.update(1)
                    run_done = True
                    if sim_bar is not None:
                        sim_bar.close()
                        sim_bar = None

            out = run_master_simulation(config_path=args.config, step_callback=_on_mc_step)
        finally:
            if sim_bar is not None:
                sim_bar.close()
            if mc_bar is not None:
                if mc_bar.n < mc_total:
                    mc_bar.update(mc_total - mc_bar.n)
                mc_bar.close()
    else:
        total_steps = int(max(math.floor(float(cfg.simulator.duration_s) / float(cfg.simulator.dt_s)), 0))
        pbar = None
        last_step = 0
        if total_steps > 0:
            try:
                from tqdm.auto import tqdm  # type: ignore

                pbar = tqdm(total=total_steps, desc="Simulation", unit="step")
            except ImportError:
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
