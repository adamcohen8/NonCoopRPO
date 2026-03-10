import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machine_learning import PPOConfig, PPOLightningModule, RLRendezvousConfig, RLRendezvousEnv


def _extract_version(path: Path) -> int:
    stem = path.stem
    if "-v" not in stem:
        return 0
    try:
        return int(stem.rsplit("-v", 1)[1])
    except Exception:
        return 0


def _default_best_ckpt_path() -> Path:
    ckpt_dir = REPO_ROOT / "outputs" / "ml" / "rendezvous_ppo" / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    last_candidates = sorted(ckpt_dir.glob("last*.ckpt"))
    if last_candidates:
        latest_last = max(last_candidates, key=lambda p: (_extract_version(p), p.stat().st_mtime))
        v = _extract_version(latest_last)
        same_version_best = sorted(ckpt_dir.glob(f"epoch*-closest*-v{v}.ckpt"))
        if same_version_best:
            return max(same_version_best, key=lambda p: p.stat().st_mtime)

    best_named = sorted(ckpt_dir.glob("epoch*-closest*.ckpt"))
    if best_named:
        return max(best_named, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")


def run_demo(
    checkpoint_path: Path,
    env_cfg: RLRendezvousConfig,
    plot_mode: str = "interactive",
) -> dict[str, str]:
    env = RLRendezvousEnv(env_cfg)
    model = PPOLightningModule(env=env, cfg=PPOConfig())

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    obs = env.reset()
    done = False
    t = [0.0]
    range_m = []
    closest_m = []
    fuel_kg = []
    thrust_on = []
    torque_cmd_nm = []
    q_hist = []
    w_hist = []

    while not done:
        action = model.act_deterministic(obs)
        obs, _, done, info = env.step(action)
        range_m.append(float(info["range_km"]) * 1e3)
        closest_m.append(float(info["closest_range_km"]) * 1e3)
        fuel_kg.append(float(info["fuel_remaining_kg"]))
        thrust_on.append(float(action[3] > 0.5))
        rw_lim = np.array(env.chaser.actuator.attitude.reaction_wheels.max_torque_nm, dtype=float)
        torque_cmd_nm.append(np.clip(np.array(action[:3], dtype=float), -1.0, 1.0) * rw_lim)
        q_hist.append(np.array(env.chaser.truth.attitude_quat_bn, dtype=float))
        w_hist.append(np.array(env.chaser.truth.angular_rate_body_rad_s, dtype=float))
        t.append(env.chaser.truth.t_s)

    t = np.array(t[1:], dtype=float)
    range_m = np.array(range_m, dtype=float)
    closest_m = np.array(closest_m, dtype=float)
    fuel_kg = np.array(fuel_kg, dtype=float)
    thrust_on = np.array(thrust_on, dtype=float)
    torque_cmd_nm = np.array(torque_cmd_nm, dtype=float) if torque_cmd_nm else np.zeros((0, 3))
    q_hist = np.array(q_hist, dtype=float) if q_hist else np.zeros((0, 4))
    w_hist = np.array(w_hist, dtype=float) if w_hist else np.zeros((0, 3))

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, range_m, label="range")
    axes[0].plot(t, closest_m, "--", label="running closest")
    axes[0].axhline(float(env_cfg.capture_radius_m), linestyle=":", color="k", label="capture threshold")
    axes[0].set_ylabel("Range (m)")
    axes[0].set_title("Best-Epoch Policy Rollout on Training Scenario")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, fuel_kg)
    axes[1].set_ylabel("Fuel Remaining (kg)")
    axes[1].grid(True, alpha=0.3)

    axes[2].step(t, thrust_on, where="post")
    axes[2].set_ylabel("Thrust On")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()

    outdir = REPO_ROOT / "outputs" / "ml" / "rendezvous_ppo"
    outdir.mkdir(parents=True, exist_ok=True)
    plot_path = outdir / "best_epoch_rollout.png"
    if plot_mode in ("save", "both"):
        fig.savefig(plot_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    fig_u, axes_u = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    torque_labels = ["tau_x (N m)", "tau_y (N m)", "tau_z (N m)"]
    for i in range(3):
        axes_u[i].plot(t, torque_cmd_nm[:, i])
        axes_u[i].set_ylabel(torque_labels[i])
        axes_u[i].grid(True, alpha=0.3)
    axes_u[0].set_title("Control Inputs Over Time")
    axes_u[3].step(t, thrust_on, where="post")
    axes_u[3].set_ylabel("thrust")
    axes_u[3].set_ylim(-0.1, 1.1)
    axes_u[3].set_xlabel("Time (s)")
    axes_u[3].grid(True, alpha=0.3)
    fig_u.tight_layout()

    control_path = outdir / "best_epoch_controls.png"
    if plot_mode in ("save", "both"):
        fig_u.savefig(control_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_u)

    fig_a, axes_a = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i in range(4):
        axes_a[0].plot(t, q_hist[:, i], label=f"q{i}")
    axes_a[0].set_ylabel("Quaternion")
    axes_a[0].set_title("Chaser Attitude Over Time")
    axes_a[0].legend(loc="best")
    axes_a[0].grid(True, alpha=0.3)
    for i, lbl in enumerate(["wx", "wy", "wz"]):
        axes_a[1].plot(t, w_hist[:, i], label=lbl)
    axes_a[1].set_ylabel("rad/s")
    axes_a[1].set_xlabel("Time (s)")
    axes_a[1].legend(loc="best")
    axes_a[1].grid(True, alpha=0.3)
    fig_a.tight_layout()

    att_path = outdir / "best_epoch_attitude.png"
    if plot_mode in ("save", "both"):
        fig_a.savefig(att_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_a)

    return {
        "checkpoint": str(checkpoint_path),
        "closest_approach_m": f"{float(np.min(range_m)) if range_m.size else np.inf}",
        "final_range_m": f"{float(range_m[-1]) if range_m.size else np.inf}",
        "final_fuel_kg": f"{float(fuel_kg[-1]) if fuel_kg.size else np.nan}",
        "rollout_plot": str(plot_path) if plot_mode in ("save", "both") else "",
        "controls_plot": str(control_path) if plot_mode in ("save", "both") else "",
        "attitude_plot": str(att_path) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the best-epoch PPO checkpoint on the training rendezvous scenario.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint. Defaults to best in outputs/ml/rendezvous_ppo/checkpoints.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--orbits", type=int, default=1, help="Scenario episode length in orbits (1..10).")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else _default_best_ckpt_path()
    env_cfg = RLRendezvousConfig(
        dt_s=float(args.dt),
        orbits_per_episode=max(1, min(10, int(args.orbits))),
        capture_radius_m=5.0,
        seed=int(args.seed),
        knowledge_refresh_s=float(args.dt),
        knowledge_max_range_km=5000.0,
        knowledge_require_los=True,
        knowledge_dropout_prob=0.0,
    )
    outputs = run_demo(checkpoint_path=ckpt_path, env_cfg=env_cfg, plot_mode=args.plot_mode)
    print("Best-epoch demo outputs:")
    for k, v in outputs.items():
        if v:
            print(f"  {k}: {v}")
