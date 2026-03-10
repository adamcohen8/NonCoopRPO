from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset

from machine_learning.attitude_ric_env import AttitudeRICRLEnv


@dataclass(frozen=True)
class AttitudeRICPPOConfig:
    gamma: float = 0.995
    lam: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    ppo_epochs: int = 6
    minibatch_size: int = 512
    episodes_per_epoch: int = 16
    max_grad_norm: float = 0.5


class _OneItemDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.zeros(1)


class _ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mu = self.mu_head(h)
        value = self.value_head(h).squeeze(-1)
        return mu, value


def _gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    t = rewards.size
    adv = np.zeros_like(rewards)
    last = 0.0
    for i in reversed(range(t)):
        next_non_terminal = 1.0 - float(dones[i])
        next_val = values[i + 1]
        delta = rewards[i] + gamma * next_val * next_non_terminal - values[i]
        last = delta + gamma * lam * next_non_terminal * last
        adv[i] = last
    returns = adv + values[:-1]
    return adv, returns


def _pl_base():
    try:
        import lightning.pytorch as pl
    except Exception:
        import pytorch_lightning as pl
    return pl.LightningModule


LightningModule = _pl_base()


class AttitudeRICPPOLightningModule(LightningModule):
    def __init__(self, env: AttitudeRICRLEnv, cfg: AttitudeRICPPOConfig):
        super().__init__()
        self.env = env
        self.cfg = cfg
        self.model = _ActorCritic(obs_dim=env.obs_dim, action_dim=env.action_dim)
        self.automatic_optimization = False
        self._target_epoch = -1
        self.save_hyperparameters(ignore=["env"])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(_OneItemDataset(), batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def training_step(self, batch, batch_idx):
        if self._target_epoch != int(self.current_epoch):
            target = self.env.sample_new_target_for_epoch()
            self._target_epoch = int(self.current_epoch)
            self.log("train/target_yaw_deg", float(np.rad2deg(target[0])), on_step=False, on_epoch=True)
            self.log("train/target_roll_deg", float(np.rad2deg(target[1])), on_step=False, on_epoch=True)
            self.log("train/target_pitch_deg", float(np.rad2deg(target[2])), on_step=False, on_epoch=True)

        opt = self.optimizers()
        roll = self._collect_rollouts()
        obs = torch.from_numpy(roll["obs"]).float().to(self.device)
        act = torch.from_numpy(roll["act_raw"]).float().to(self.device)
        old_logp = torch.from_numpy(roll["logp"]).float().to(self.device)
        adv = torch.from_numpy(roll["adv"]).float().to(self.device)
        ret = torch.from_numpy(roll["ret"]).float().to(self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs.shape[0]
        mb = min(self.cfg.minibatch_size, n)
        last_loss = torch.tensor(0.0, device=self.device)
        for _ in range(self.cfg.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, mb):
                idx = perm[i : i + mb]
                mu, v = self.model(obs[idx])
                std = torch.exp(self.model.log_std).unsqueeze(0).expand_as(mu)
                dist = Normal(mu, std)
                logp = dist.log_prob(act[idx]).sum(dim=1)
                ratio = torch.exp(logp - old_logp[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (ret[idx] - v).pow(2).mean()
                entropy = dist.entropy().sum(dim=1).mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                last_loss = loss

                opt.zero_grad()
                self.manual_backward(loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                opt.step()

        self.log("train/loss", last_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/err_deg_mean", float(roll["err_deg_mean"]), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/err_deg_final_mean", float(roll["err_deg_final_mean"]), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/success_rate", float(roll["success_rate"]), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_err_deg_final_mean", float(roll["err_deg_final_mean"]), on_step=False, on_epoch=True)
        return last_loss

    @torch.no_grad()
    def _collect_rollouts(self) -> dict[str, np.ndarray]:
        obs_all = []
        act_all = []
        logp_all = []
        adv_all = []
        ret_all = []
        err_mean = []
        err_final = []
        succ = 0
        for _ in range(self.cfg.episodes_per_epoch):
            obs = self.env.reset()
            done = False
            ep_rewards = []
            ep_dones = []
            ep_vals = []
            ep_err = []
            while not done:
                obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
                mu, v = self.model(obs_t)
                std = torch.exp(self.model.log_std).unsqueeze(0).expand_as(mu)
                dist = Normal(mu, std)
                a_raw = dist.sample()
                logp = dist.log_prob(a_raw).sum(dim=1)
                action = torch.clamp(a_raw, -1.0, 1.0).squeeze(0).cpu().numpy()
                obs_next, reward, done, info = self.env.step(action)

                obs_all.append(obs.copy())
                act_all.append(a_raw.squeeze(0).cpu().numpy())
                logp_all.append(float(logp.item()))
                ep_rewards.append(float(reward))
                ep_dones.append(bool(done))
                ep_vals.append(float(v.item()))
                ep_err.append(float(info["attitude_error_deg"]))
                if info.get("success", False):
                    succ += 1
                obs = obs_next

            ep_rewards_np = np.array(ep_rewards, dtype=np.float32)
            ep_dones_np = np.array(ep_dones, dtype=bool)
            ep_vals_np = np.array(ep_vals + [0.0], dtype=np.float32)
            ep_adv, ep_ret = _gae(ep_rewards_np, ep_vals_np, ep_dones_np, gamma=self.cfg.gamma, lam=self.cfg.lam)
            adv_all.extend(ep_adv.tolist())
            ret_all.extend(ep_ret.tolist())
            err_mean.append(float(np.mean(ep_err)) if ep_err else np.nan)
            err_final.append(float(ep_err[-1]) if ep_err else np.nan)

        return {
            "obs": np.array(obs_all, dtype=np.float32),
            "act_raw": np.array(act_all, dtype=np.float32),
            "logp": np.array(logp_all, dtype=np.float32),
            "adv": np.array(adv_all, dtype=np.float32),
            "ret": np.array(ret_all, dtype=np.float32),
            "err_deg_mean": float(np.nanmean(err_mean)) if err_mean else np.nan,
            "err_deg_final_mean": float(np.nanmean(err_final)) if err_final else np.nan,
            "success_rate": float(succ / max(1, self.cfg.episodes_per_epoch)),
        }

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        mu, _ = self.model(obs_t)
        return torch.clamp(mu, -1.0, 1.0).squeeze(0).cpu().numpy()
