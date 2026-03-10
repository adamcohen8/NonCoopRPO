from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal
from torch.utils.data import DataLoader, Dataset

from machine_learning.rendezvous_env import RLRendezvousEnv

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        if iterable is None:
            class _NoopBar:
                def update(self, n=1):
                    return None

                def close(self):
                    return None

            return _NoopBar()
        return iterable


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.995
    lam: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    ppo_epochs: int = 6
    minibatch_size: int = 256
    episodes_per_epoch: int = 4
    max_grad_norm: float = 0.5
    rollout_progress: bool = True


class _OneItemDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.zeros(1)


class _ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, 3)
        self.log_std = nn.Parameter(torch.zeros(3))
        self.thrust_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mu = self.mu_head(h)
        thrust_logits = self.thrust_head(h).squeeze(-1)
        value = self.value_head(h).squeeze(-1)
        return mu, thrust_logits, value


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


class PPOLightningModuleBase:
    pass


def create_pl_module():
    try:
        import lightning.pytorch as pl
    except Exception:
        import pytorch_lightning as pl
    return pl.LightningModule


LightningModule = create_pl_module()


class PPOLightningModule(LightningModule):
    def __init__(self, env: RLRendezvousEnv, cfg: PPOConfig):
        super().__init__()
        self.env = env
        self.cfg = cfg
        self.model = _ActorCritic(obs_dim=env.obs_dim)
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["env"])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(_OneItemDataset(), batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        roll = self._collect_rollouts()
        obs = torch.from_numpy(roll["obs"]).float().to(self.device)
        act_torque_raw = torch.from_numpy(roll["act_torque_raw"]).float().to(self.device)
        act_thrust = torch.from_numpy(roll["act_thrust"]).float().to(self.device)
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
                mu, thrust_logits, v = self.model(obs[idx])
                std = torch.exp(self.model.log_std).unsqueeze(0).expand_as(mu)
                dist_torque = Normal(mu, std)
                dist_thrust = Bernoulli(logits=thrust_logits)
                logp_t = dist_torque.log_prob(act_torque_raw[idx]).sum(dim=1)
                logp_b = dist_thrust.log_prob(act_thrust[idx]).squeeze(-1)
                logp = logp_t + logp_b

                ratio = torch.exp(logp - old_logp[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (ret[idx] - v).pow(2).mean()
                entropy = dist_torque.entropy().sum(dim=1).mean() + dist_thrust.entropy().mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                last_loss = loss

                opt.zero_grad()
                self.manual_backward(loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                opt.step()

        closest_m = float(roll["closest_range_km"] * 1e3)
        capture_rate = float(roll["capture_rate"])
        self.log("train/loss", last_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/closest_m", closest_m, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/capture_rate", capture_rate, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_closest_m", closest_m, prog_bar=False, on_step=False, on_epoch=True)
        return last_loss

    @torch.no_grad()
    def _collect_rollouts(self) -> dict[str, np.ndarray]:
        obs_all: list[np.ndarray] = []
        act_torque_raw_all: list[np.ndarray] = []
        act_thrust_all: list[np.ndarray] = []
        logp_all: list[float] = []
        adv_all: list[float] = []
        ret_all: list[float] = []
        closest_km: list[float] = []
        captures = 0

        show_progress = bool(self.cfg.rollout_progress and (self.trainer is None or getattr(self.trainer, "is_global_zero", True)))
        ep_iter = range(self.cfg.episodes_per_epoch)
        if show_progress:
            ep_iter = tqdm(
                ep_iter,
                desc=f"Epoch {int(self.current_epoch)} rollout",
                unit="ep",
                leave=False,
                dynamic_ncols=True,
            )

        for ep_idx in ep_iter:
            obs = self.env.reset()
            done = False
            ep_closest = np.inf
            ep_rewards: list[float] = []
            ep_dones: list[bool] = []
            ep_values: list[float] = []
            step_bar = tqdm(
                total=self.env.max_steps,
                desc=f"  Ep {ep_idx + 1} steps",
                unit="step",
                leave=False,
                dynamic_ncols=True,
            ) if show_progress else None
            while not done:
                obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
                mu, thrust_logits, v = self.model(obs_t)
                std = torch.exp(self.model.log_std).unsqueeze(0).expand_as(mu)
                dist_torque = Normal(mu, std)
                dist_thrust = Bernoulli(logits=thrust_logits)
                raw_torque = dist_torque.sample()
                thrust_sample = dist_thrust.sample()

                logp = dist_torque.log_prob(raw_torque).sum(dim=1) + dist_thrust.log_prob(thrust_sample).squeeze(-1)
                torque_norm = torch.clamp(raw_torque, -1.0, 1.0)
                action = torch.cat([torque_norm, thrust_sample.unsqueeze(-1)], dim=1).squeeze(0).cpu().numpy()
                obs_next, reward, done, info = self.env.step(action)

                obs_all.append(obs.copy())
                act_torque_raw_all.append(raw_torque.squeeze(0).cpu().numpy())
                act_thrust_all.append(np.array([float(thrust_sample.item())], dtype=np.float32))
                logp_all.append(float(logp.item()))
                ep_rewards.append(float(reward))
                ep_dones.append(bool(done))
                ep_values.append(float(v.item()))
                ep_closest = min(ep_closest, float(info["closest_range_km"]))
                if bool(info.get("capture", False)):
                    captures = 1 + captures
                obs = obs_next
                if step_bar is not None:
                    step_bar.update(1)

            if step_bar is not None:
                step_bar.close()

            ep_rewards_np = np.array(ep_rewards, dtype=np.float32)
            ep_dones_np = np.array(ep_dones, dtype=bool)
            ep_values_np = np.array(ep_values + [0.0], dtype=np.float32)
            ep_adv, ep_ret = _gae(ep_rewards_np, ep_values_np, ep_dones_np, gamma=self.cfg.gamma, lam=self.cfg.lam)
            adv_all.extend(ep_adv.tolist())
            ret_all.extend(ep_ret.tolist())
            closest_km.append(ep_closest)

        return {
            "obs": np.array(obs_all, dtype=np.float32),
            "act_torque_raw": np.array(act_torque_raw_all, dtype=np.float32),
            "act_thrust": np.array(act_thrust_all, dtype=np.float32),
            "logp": np.array(logp_all, dtype=np.float32),
            "adv": np.array(adv_all, dtype=np.float32),
            "ret": np.array(ret_all, dtype=np.float32),
            "closest_range_km": float(np.mean(closest_km)) if closest_km else np.inf,
            "capture_rate": float(captures / max(1, self.cfg.episodes_per_epoch)),
        }

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        mu, thrust_logits, _ = self.model(obs_t)
        torque_norm = torch.clamp(mu, -1.0, 1.0).squeeze(0).cpu().numpy()
        thrust_on = (torch.sigmoid(thrust_logits).item() >= 0.5)
        return np.hstack((torque_norm, np.array([1.0 if thrust_on else 0.0], dtype=np.float32)))
