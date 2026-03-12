from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.scenarios import (
    ASATPhasedScenarioConfig,
    AgentStrategyConfig,
    KnowledgeGateConfig,
    run_asat_phased_engagement,
)


def run_demo(plot_mode: str = "interactive") -> dict[str, str]:
    cfg = ASATPhasedScenarioConfig(
        dt_s=2.0,
        duration_s=3600.0,
        deploy_time_s=900.0,
        chaser_deploy_dv_body_m_s=np.array([10.0, 0.0, 0.0]),
    )
    out = run_asat_phased_engagement(
        cfg,
        rocket_strategy=AgentStrategyConfig(mode="coast", max_accel_km_s2=0.0, target_id=cfg.target_id),
        target_strategy=AgentStrategyConfig(mode="knowledge_evade", max_accel_km_s2=2.0e-6, target_id=cfg.rocket_id),
        chaser_strategy=AgentStrategyConfig(mode="knowledge_pursuit", max_accel_km_s2=4.0e-6, target_id=cfg.target_id),
        gates=KnowledgeGateConfig(
            rocket_starts_tracking_target_at_s=5.0,
            target_starts_tracking_rocket_at_s=20.0,
            chaser_starts_tracking_target_at_s=cfg.deploy_time_s + 20.0,
        ),
    )

    t = out["time_s"]
    truth = out["truth_by_object"]
    rocket_id = cfg.rocket_id
    target_id = cfg.target_id
    chaser_id = cfg.chaser_id

    rr = np.linalg.norm(truth[rocket_id][:, :3] - truth[target_id][:, :3], axis=1)
    rc = np.linalg.norm(truth[chaser_id][:, :3] - truth[target_id][:, :3], axis=1)

    outdir = REPO_ROOT / "outputs" / "asat_phased_engagement_demo"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(t, rr, label="Rocket-Target Range")
    ax[0].plot(t, rc, label="Chaser-Target Range")
    ax[0].axvline(cfg.deploy_time_s, color="k", linestyle="--", linewidth=1.0, label="Deploy")
    ax[0].set_ylabel("km")
    ax[0].set_title("Engagement Ranges")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="best")

    ax[1].plot(t, out["rocket_throttle_cmd"], label="Rocket Throttle")
    ax[1].set_ylabel("-")
    ax[1].set_title("Rocket Throttle Command")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="best")

    kb = out["knowledge_by_observer"]
    rk_tgt = np.any(np.isfinite(kb[rocket_id][target_id]), axis=1).astype(float)
    tg_rkt = np.any(np.isfinite(kb[target_id][rocket_id]), axis=1).astype(float)
    ch_tgt = np.any(np.isfinite(kb[chaser_id][target_id]), axis=1).astype(float)
    ax[2].plot(t, rk_tgt, label="Rocket knows Target")
    ax[2].plot(t, tg_rkt, label="Target knows Rocket")
    ax[2].plot(t, ch_tgt, label="Chaser knows Target")
    ax[2].set_ylabel("Known (0/1)")
    ax[2].set_ylim(-0.1, 1.1)
    ax[2].set_title("Knowledge Activation/Tracking")
    ax[2].set_xlabel("Time (s)")
    ax[2].grid(True, alpha=0.3)
    ax[2].legend(loc="best")

    fig.tight_layout()
    p = outdir / "asat_phased_engagement_summary.png"
    if plot_mode in ("save", "both"):
        fig.savefig(p, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    return {
        "chaser_deployed": str(out["chaser_deployed"]),
        "chaser_deploy_time_s": str(out["chaser_deploy_time_s"]),
        "min_chaser_target_range_km": f"{out['min_chaser_target_range_km']:.3f}",
        "plot": str(p) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phased ASAT scenario demo: rocket launch, knowledge gating, and chaser deployment.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    args = parser.parse_args()
    res = run_demo(plot_mode=args.plot_mode)
    print("ASAT phased engagement demo outputs:")
    for k, v in res.items():
        if v:
            print(f"  {k}: {v}")
