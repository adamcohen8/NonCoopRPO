from __future__ import annotations

import argparse
from pathlib import Path
import sys

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


def _import_pl():
    try:
        import lightning.pytorch as pl
        return pl
    except Exception:
        import pytorch_lightning as pl
        return pl


def _build_checkpoint_callback(pl, ckpt_dir: Path):
    return pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch{epoch:03d}-err{train_err_deg_final_mean:.2f}",
        monitor="train_err_deg_final_mean",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        every_n_epochs=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO RIC-frame 3-axis reaction-wheel attitude controller.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--episode-duration-s", type=float, default=100.0)
    parser.add_argument("--episodes-per-epoch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    env_cfg = AttitudeRICRLConfig(
        dt_s=float(args.dt),
        episode_duration_s=float(args.episode_duration_s),
        seed=int(args.seed),
    )
    env = AttitudeRICRLEnv(env_cfg)
    ppo_cfg = AttitudeRICPPOConfig(episodes_per_epoch=int(args.episodes_per_epoch))
    module = AttitudeRICPPOLightningModule(env=env, cfg=ppo_cfg)

    pl = _import_pl()
    outdir = REPO_ROOT / "outputs" / "ml" / "attitude_ric_ppo"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = _build_checkpoint_callback(pl, ckpt_dir)

    trainer = pl.Trainer(
        max_epochs=int(args.epochs),
        logger=True,
        callbacks=[ckpt_cb],
        enable_checkpointing=True,
        log_every_n_steps=int(args.log_every_n_steps),
        accelerator="auto",
        devices=1,
    )
    trainer.fit(module)

    model_path = outdir / "policy_last.pt"
    torch.save(module.model.state_dict(), model_path)
    print("Training complete.")
    print(f"Saved policy: {model_path}")
    print(f"Best checkpoint: {ckpt_cb.best_model_path}")
    print(f"Last checkpoint: {ckpt_cb.last_model_path}")
