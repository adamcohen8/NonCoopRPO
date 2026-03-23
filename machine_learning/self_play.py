from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from machine_learning.training_adapter import MultiAgentRolloutBatch, collect_multi_agent_rollout


class SupportsPolicy(Protocol):
    def act(self, obs: np.ndarray) -> np.ndarray:
        ...

    def clone(self) -> "SupportsPolicy":
        ...


class SupportsTrainablePolicy(SupportsPolicy, Protocol):
    def mutate(self, rng: np.random.Generator, sigma: float) -> np.ndarray:
        ...

    def update(self, direction: np.ndarray, scale: float) -> None:
        ...


@dataclass(frozen=True)
class SelfPlayTrainerConfig:
    update_mode: str = "alternating"
    iterations: int = 10
    rollout_horizon: int = 32
    learning_rate: float = 0.05
    mutation_sigma: float = 0.01
    snapshot_interval: int = 1
    max_opponents: int = 8
    seed: int = 0


@dataclass
class OpponentPool:
    max_size: int = 8
    snapshots_by_agent: dict[str, list[SupportsPolicy]] = field(default_factory=dict)

    def add(self, agent_id: str, policy: SupportsPolicy) -> None:
        snaps = self.snapshots_by_agent.setdefault(str(agent_id), [])
        snaps.append(policy.clone())
        overflow = len(snaps) - int(self.max_size)
        if overflow > 0:
            del snaps[0:overflow]

    def sample(self, agent_id: str, rng: np.random.Generator) -> SupportsPolicy | None:
        snaps = self.snapshots_by_agent.get(str(agent_id), [])
        if not snaps:
            return None
        idx = int(rng.integers(0, len(snaps)))
        return snaps[idx].clone()


@dataclass
class LinearPolicy:
    weights: np.ndarray
    bias: np.ndarray

    @classmethod
    def random(cls, obs_dim: int, action_dim: int, rng: np.random.Generator) -> "LinearPolicy":
        return cls(
            weights=rng.normal(0.0, 0.05, size=(int(obs_dim), int(action_dim))),
            bias=np.zeros(int(action_dim), dtype=np.float64),
        )

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.array(obs, dtype=np.float64).reshape(-1)
        y = np.tanh(x @ self.weights + self.bias)
        return y.astype(np.float32)

    def update(self, direction: np.ndarray, scale: float) -> None:
        self.weights += np.array(direction, dtype=np.float64) * float(scale)

    def mutate(self, rng: np.random.Generator, sigma: float) -> np.ndarray:
        return rng.normal(0.0, float(sigma), size=self.weights.shape)

    def clone(self) -> "LinearPolicy":
        return LinearPolicy(weights=self.weights.copy(), bias=self.bias.copy())


def summarize_multi_agent_batch(batch: MultiAgentRolloutBatch) -> dict[str, float]:
    out: dict[str, float] = {}
    for agent_id, rewards in batch.rewards_by_agent.items():
        out[f"{agent_id}_reward_mean"] = float(np.mean(rewards))
        out[f"{agent_id}_reward_sum"] = float(np.sum(rewards))
    if "chaser" in batch.infos_by_agent and "metrics" in batch.infos_by_agent["chaser"]:
        metrics = list(batch.infos_by_agent["chaser"]["metrics"])
        if metrics:
            out["range_final"] = float(metrics[-1]["range_km"])
            out["range_closest"] = float(min(m["closest_range_km"] for m in metrics))
    return out


def evaluate_self_play_policies(
    env: Any,
    policies_by_agent: dict[str, SupportsPolicy],
    *,
    horizon: int,
    seed: int,
) -> tuple[MultiAgentRolloutBatch, dict[str, float]]:
    batch = collect_multi_agent_rollout(
        env,
        policy_fns_by_agent={agent_id: policy.act for agent_id, policy in policies_by_agent.items()},
        horizon=int(horizon),
        reset_kwargs={"seed": int(seed)},
    )
    return batch, summarize_multi_agent_batch(batch)


def run_self_play_training(
    env: Any,
    *,
    policies_by_agent: dict[str, SupportsTrainablePolicy],
    trainer_cfg: SelfPlayTrainerConfig,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if str(trainer_cfg.update_mode) not in {"alternating", "simultaneous"}:
        raise ValueError("update_mode must be 'alternating' or 'simultaneous'.")
    rng = np.random.default_rng(int(trainer_cfg.seed))
    opponent_pool = OpponentPool(max_size=int(trainer_cfg.max_opponents))
    agent_ids = tuple(sorted(policies_by_agent.keys()))
    history: list[dict[str, float]] = []
    for agent_id, policy in policies_by_agent.items():
        opponent_pool.add(agent_id, policy)

    for iteration in range(int(trainer_cfg.iterations)):
        opponents = {
            agent_id: opponent_pool.sample(agent_id, rng) or policies_by_agent[agent_id].clone()
            for agent_id in agent_ids
        }
        if trainer_cfg.update_mode == "alternating":
            learner = agent_ids[iteration % len(agent_ids)]
            policies_for_rollout = {
                agent_id: (policies_by_agent[agent_id] if agent_id == learner else opponents[agent_id])
                for agent_id in agent_ids
            }
        else:
            learner = None
            policies_for_rollout = dict(policies_by_agent)

        batch, stats = evaluate_self_play_policies(
            env,
            policies_by_agent=policies_for_rollout,
            horizon=int(trainer_cfg.rollout_horizon),
            seed=int(trainer_cfg.seed + iteration),
        )

        if trainer_cfg.update_mode == "simultaneous":
            for agent_id in agent_ids:
                direction = policies_by_agent[agent_id].mutate(rng, sigma=float(trainer_cfg.mutation_sigma))
                reward_sum = float(stats.get(f"{agent_id}_reward_sum", 0.0))
                policies_by_agent[agent_id].update(direction, scale=np.sign(reward_sum) * float(trainer_cfg.learning_rate))
        else:
            assert learner is not None
            direction = policies_by_agent[learner].mutate(rng, sigma=float(trainer_cfg.mutation_sigma))
            reward_sum = float(stats.get(f"{learner}_reward_sum", 0.0))
            policies_by_agent[learner].update(direction, scale=np.sign(reward_sum) * float(trainer_cfg.learning_rate))

        if int(trainer_cfg.snapshot_interval) > 0 and ((iteration + 1) % int(trainer_cfg.snapshot_interval) == 0):
            for agent_id in agent_ids:
                opponent_pool.add(agent_id, policies_by_agent[agent_id])

        row = {"iteration": float(iteration), **stats}
        history.append(row)
        if log_fn is not None:
            leader = "all" if learner is None else learner
            log_fn(
                f"iter={iteration:03d} mode={trainer_cfg.update_mode} learner={leader} "
                + " ".join(f"{k}={v:.6f}" for k, v in stats.items())
            )

    return {
        "policies_by_agent": policies_by_agent,
        "opponent_pool": opponent_pool,
        "history": history,
        "last_batch": batch,
    }
