from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machine_learning import (
    AttitudeRICPPOConfig,
    AttitudeRICPPOLightningModule,
    AttitudeRICRLEnv,
    AttitudeRICRLConfig,
)


def _extract_version(path: Path) -> int:
    stem = path.stem
    if "-v" not in stem:
        return 0
    try:
        return int(stem.rsplit("-v", 1)[1])
    except Exception:
        return 0


def _default_best_ckpt_path() -> Path:
    ckpt_dir = REPO_ROOT / "outputs" / "ml" / "attitude_ric_ppo" / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    last_candidates = sorted(ckpt_dir.glob("last*.ckpt"))
    if last_candidates:
        latest_last = max(last_candidates, key=lambda p: (_extract_version(p), p.stat().st_mtime))
        v = _extract_version(latest_last)
        same_version_best = sorted(ckpt_dir.glob(f"epoch*-err*-v{v}.ckpt"))
        if same_version_best:
            return max(same_version_best, key=lambda p: p.stat().st_mtime)
    best_named = sorted(ckpt_dir.glob("epoch*-err*.ckpt"))
    if best_named:
        return max(best_named, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")


def run_demo(checkpoint_path: Path, env_cfg: AttitudeRICRLConfig, plot_mode: str = "interactive") -> dict[str, str]:
    env = AttitudeRICRLEnv(env_cfg)
    env.sample_new_target_for_epoch()
    model = AttitudeRICPPOLightningModule(env=env, cfg=AttitudeRICPPOConfig())
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    obs = env.reset()
    done = False
    t = [0.0]
    err = []
    rate = []
    wheel = []
    act = []
    while not done:
        u = model.act_deterministic(obs)
        obs, _, done, info = env.step(u)
        err.append(float(info["attitude_error_deg"]))
        rate.append(float(info["rate_norm_rad_s"]))
        wheel.append(float(info["wheel_speed_norm_rad_s"]))
        act.append(np.array(u, dtype=float))
        t.append(env.sat.truth.t_s)

    t = np.array(t[1:], dtype=float)
    err = np.array(err, dtype=float)
    rate = np.array(rate, dtype=float)
    wheel = np.array(wheel, dtype=float)
    act = np.array(act, dtype=float)

    outdir = REPO_ROOT / "outputs" / "ml" / "attitude_ric_ppo"
    outdir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax1[0].plot(t, err)
    ax1[0].set_ylabel("deg")
    ax1[0].set_title("RIC Attitude Error")
    ax1[0].grid(True, alpha=0.3)
    ax1[1].plot(t, rate)
    ax1[1].set_ylabel("rad/s")
    ax1[1].set_title("RIC Rate Norm")
    ax1[1].grid(True, alpha=0.3)
    ax1[2].plot(t, wheel)
    ax1[2].set_ylabel("rad/s")
    ax1[2].set_xlabel("Time (s)")
    ax1[2].set_title("Wheel Speed Norm")
    ax1[2].grid(True, alpha=0.3)
    fig1.tight_layout()
    p1 = outdir / "best_epoch_ric_attitude_metrics.png"
    if plot_mode in ("save", "both"):
        fig1.savefig(p1, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig1)

    fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["u1", "u2", "u3"]
    for i in range(3):
        ax2[i].plot(t, act[:, i], label=labels[i])
        ax2[i].set_ylabel("norm")
        ax2[i].grid(True, alpha=0.3)
    ax2[0].set_title("RL Wheel Commands")
    ax2[2].set_xlabel("Time (s)")
    fig2.tight_layout()
    p2 = outdir / "best_epoch_ric_attitude_actions.png"
    if plot_mode in ("save", "both"):
        fig2.savefig(p2, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig2)

    return {
        "checkpoint": str(checkpoint_path),
        "final_error_deg": str(float(err[-1]) if err.size else np.nan),
        "min_error_deg": str(float(np.min(err)) if err.size else np.nan),
        "metrics_plot": str(p1) if plot_mode in ("save", "both") else "",
        "actions_plot": str(p2) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run best-epoch RIC attitude RL policy rollout.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--episode-duration-s", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else _default_best_ckpt_path()
    env_cfg = AttitudeRICRLConfig(
        dt_s=float(args.dt),
        episode_duration_s=float(args.episode_duration_s),
        seed=int(args.seed),
    )
    outputs = run_demo(ckpt_path, env_cfg, plot_mode=args.plot_mode)
    print("RIC-attitude best-epoch demo outputs:")
    for k, v in outputs.items():
        if v:
            print(f"  {k}: {v}")
