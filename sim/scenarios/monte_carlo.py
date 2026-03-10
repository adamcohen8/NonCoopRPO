from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from sim.metrics.engagement import compute_engagement_metrics
from sim.utils.io import write_json

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass(frozen=True)
class MonteCarloConfig:
    runs: int
    base_seed: int = 0
    pos_sigma_km: float = 0.01
    vel_sigma_km_s: float = 1e-4


def run_monte_carlo(
    config: MonteCarloConfig,
    scenario_fn,
    output_dir: str,
    plot_mode: Literal["interactive", "save", "both"] = "interactive",
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    min_sep = []
    overruns = []
    summaries = []

    for i in tqdm(range(config.runs), desc="Monte Carlo Runs", unit="run", dynamic_ncols=True):
        seed = config.base_seed + i
        result = scenario_fn(seed=seed, pos_sigma_km=config.pos_sigma_km, vel_sigma_km_s=config.vel_sigma_km_s)
        metrics = compute_engagement_metrics(result["log"], keepout_radius_km=result.get("keepout_radius_km"))
        min_sep.append(metrics.min_separation_km)
        overruns.append(sum(metrics.compute_overruns_by_object.values()))
        summaries.append(
            {
                "seed": seed,
                "metrics": asdict(metrics),
            }
        )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(min_sep, bins=min(20, max(5, int(np.sqrt(len(min_sep))))), alpha=0.8)
    ax.set_title("Monte Carlo Minimum Separation")
    ax.set_xlabel("Minimum Separation (km)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    hist_path = out / "min_separation_hist.png"
    if plot_mode in ("save", "both"):
        fig.savefig(hist_path, dpi=150)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    summary = {
        "config": asdict(config),
        "runs": summaries,
        "aggregate": {
            "min_separation_km_mean": float(np.mean(min_sep)) if min_sep else 0.0,
            "min_separation_km_min": float(np.min(min_sep)) if min_sep else 0.0,
            "total_overruns": int(np.sum(overruns)),
        },
    }
    summary_path = out / "monte_carlo_summary.json"
    write_json(str(summary_path), summary)

    return {
        "summary_json": str(summary_path),
        "histogram_png": str(hist_path) if plot_mode in ("save", "both") else "",
    }
