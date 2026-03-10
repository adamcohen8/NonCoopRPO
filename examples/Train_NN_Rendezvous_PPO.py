import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machine_learning import PPOConfig, PPOLightningModule, RLRendezvousConfig, RLRendezvousEnv


def _import_pl():
    try:
        import lightning.pytorch as pl
        return pl
    except Exception:
        import pytorch_lightning as pl
    return pl


def _to_float(x):
    if x is None:
        return None
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _build_duration_callback(pl):
    class AdaptiveDurationCallback(pl.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_closest_m = float("inf")
            self.duration_s = 300.0
            self.min_duration_s = 300.0
            self.max_duration_s = 5400.0
            self.increment_s = 60.0

        def on_fit_start(self, trainer, pl_module):
            pl_module.env.set_episode_duration_s(self.duration_s)
            print(f"[adaptive-duration] epoch 0 start duration: {self.duration_s/60.0:.1f} min")

        def on_train_epoch_end(self, trainer, pl_module):
            epoch = int(trainer.current_epoch)
            closest = _to_float(trainer.callback_metrics.get("train_closest_m"))
            if closest is None:
                return
            improved = closest < (self.best_closest_m - 1e-9)
            if improved:
                self.best_closest_m = closest

            if epoch < 9:
                self.duration_s = self.min_duration_s
            elif improved:
                self.duration_s = min(self.max_duration_s, self.duration_s + self.increment_s)

            pl_module.env.set_episode_duration_s(self.duration_s)
            print(
                f"[adaptive-duration] epoch {epoch} closest={closest:.3f} m, best={self.best_closest_m:.3f} m, "
                f"next duration={self.duration_s/60.0:.1f} min"
            )

    return AdaptiveDurationCallback()


def _build_checkpoint_callback(pl, ckpt_dir: Path):
    return pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch{epoch:03d}-closest{train_closest_m:.2f}",
        monitor="train_closest_m",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        every_n_epochs=1,
    )


def evaluate_policy(model: PPOLightningModule, env: RLRendezvousEnv) -> dict:
    obs = env.reset()
    done = False
    ranges = []
    fuels = []
    while not done:
        action = model.act_deterministic(obs)
        obs, _, done, info = env.step(action)
        ranges.append(float(info["range_km"]))
        fuels.append(float(info["fuel_remaining_kg"]))
    return {
        "closest_approach_m": float(min(ranges) * 1e3) if ranges else np.inf,
        "final_range_m": float(ranges[-1] * 1e3) if ranges else np.inf,
        "final_fuel_kg": float(fuels[-1]) if fuels else np.nan,
        "steps": int(len(ranges)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO neural controller (attitude wheels + binary thrust) for passive-target rendezvous."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--orbits", type=int, default=1, help="Episode length in orbits (1..10).")
    parser.add_argument("--episodes-per-epoch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    env_cfg = RLRendezvousConfig(
        dt_s=float(args.dt),
        orbits_per_episode=max(1, min(10, int(args.orbits))),
        capture_radius_m=5.0,
        seed=int(args.seed),
        knowledge_refresh_s=float(args.dt),
        knowledge_max_range_km=5000.0,
        knowledge_require_los=True,
        knowledge_dropout_prob=0.0,
        reward_mode="lookahead_terminal_closest",
        lookahead_horizon_s=600.0,
        lookahead_dt_s=10.0,
    )
    env = RLRendezvousEnv(env_cfg)

    ppo_cfg = PPOConfig(
        episodes_per_epoch=int(args.episodes_per_epoch),
        rollout_progress=False,
        value_coef=0.0,
        entropy_coef=0.0,
    )
    module = PPOLightningModule(env=env, cfg=ppo_cfg)

    pl = _import_pl()
    outdir = REPO_ROOT / "outputs" / "ml" / "rendezvous_ppo"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = _build_checkpoint_callback(pl, ckpt_dir)
    duration_cb = _build_duration_callback(pl)
    trainer = pl.Trainer(
        max_epochs=int(args.epochs),
        logger=True,
        callbacks=[ckpt_cb, duration_cb],
        enable_checkpointing=True,
        log_every_n_steps=int(args.log_every_n_steps),
        accelerator="auto",
        devices=1,
    )
    trainer.fit(module)

    model_path = outdir / "policy_last.pt"
    torch.save(module.model.state_dict(), model_path)

    eval_stats = evaluate_policy(module, env)
    print("Training complete.")
    print(f"Saved policy: {model_path}")
    print(f"Best checkpoint: {ckpt_cb.best_model_path}")
    print(f"Last checkpoint: {ckpt_cb.last_model_path}")
    for k, v in eval_stats.items():
        print(f"{k}: {v}")
