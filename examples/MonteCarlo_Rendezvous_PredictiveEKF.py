import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.scenarios.rendezvous_predictive_monte_carlo import PredictiveRendezvousMCConfig, run_predictive_rendezvous_monte_carlo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo validation harness for predictive rendezvous coupling stack.")
    parser.add_argument(
        "--plot-mode",
        choices=["interactive", "save", "both"],
        default="interactive",
        help="Plot behavior; interactive is default.",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=5400.0)
    parser.add_argument("--lead-steps", type=int, default=100)
    parser.add_argument("--align-deg", type=float, default=10.0)
    parser.add_argument("--wheel-scale", type=float, default=5.0)
    parser.add_argument("--thrust-mode", choices=["attitude", "perfect"], default="attitude")
    parser.add_argument("--capture-radius-km", type=float, default=0.2)
    parser.add_argument("--capture-hold-s", type=float, default=60.0)
    parser.add_argument("--pass-final-miss-km", type=float, default=0.5)
    parser.add_argument("--pass-max-dv-km-s", type=float, default=0.40)
    parser.add_argument("--pass-max-alignment-violations", type=int, default=0)
    args = parser.parse_args()

    cfg = PredictiveRendezvousMCConfig(
        runs=int(args.runs),
        base_seed=int(args.base_seed),
        dt_s=float(args.dt),
        duration_s=float(args.duration),
        lead_steps=int(args.lead_steps),
        align_deg=float(args.align_deg),
        wheel_scale=float(args.wheel_scale),
        thrust_mode=args.thrust_mode,
        capture_radius_km=float(args.capture_radius_km),
        capture_hold_s=float(args.capture_hold_s),
        pass_final_miss_km=float(args.pass_final_miss_km),
        pass_max_dv_km_s=float(args.pass_max_dv_km_s),
        pass_max_alignment_violations=int(args.pass_max_alignment_violations),
    )

    outputs = run_predictive_rendezvous_monte_carlo(
        config=cfg,
        output_dir=str(REPO_ROOT / "outputs" / "rendezvous_predictive_mc"),
        plot_mode=args.plot_mode,
    )
    print("Monte Carlo outputs:")
    for k, v in outputs.items():
        if v:
            print(f"  {k}: {v}")
