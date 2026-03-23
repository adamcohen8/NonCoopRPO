from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from machine_learning.gym_env import EnvFactory, VectorEnvConfig, make_env_fn


@dataclass(frozen=True)
class RolloutBatch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_obs: np.ndarray
    infos: dict[str, list[Any]]


@dataclass(frozen=True)
class MultiAgentRolloutBatch:
    obs_by_agent: dict[str, np.ndarray]
    actions_by_agent: dict[str, np.ndarray]
    rewards_by_agent: dict[str, np.ndarray]
    terminated_by_agent: dict[str, np.ndarray]
    truncated_by_agent: dict[str, np.ndarray]
    next_obs_by_agent: dict[str, np.ndarray]
    infos_by_agent: dict[str, dict[str, list[Any]]]


def collect_vector_rollout(
    vec_env: Any,
    *,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    horizon: int,
    initial_obs: np.ndarray | None = None,
    reset_kwargs: dict[str, Any] | None = None,
) -> RolloutBatch:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if initial_obs is None:
        obs, _ = vec_env.reset(**dict(reset_kwargs or {}))
    else:
        obs = np.array(initial_obs, dtype=np.float32)
    obs_hist = []
    action_hist = []
    reward_hist = []
    terminated_hist = []
    truncated_hist = []
    info_hist: list[dict[str, list[Any]]] = []
    next_obs = obs
    for _ in range(int(horizon)):
        actions = np.array(policy_fn(next_obs), dtype=np.float32)
        step_obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        obs_hist.append(np.array(next_obs, dtype=np.float32))
        action_hist.append(np.array(actions, dtype=np.float32))
        reward_hist.append(np.array(rewards, dtype=np.float32))
        terminated_hist.append(np.array(terminated, dtype=bool))
        truncated_hist.append(np.array(truncated, dtype=bool))
        info_hist.append(dict(infos))
        next_obs = np.array(step_obs, dtype=np.float32)
    return RolloutBatch(
        obs=np.stack(obs_hist, axis=0),
        actions=np.stack(action_hist, axis=0),
        rewards=np.stack(reward_hist, axis=0),
        terminated=np.stack(terminated_hist, axis=0),
        truncated=np.stack(truncated_hist, axis=0),
        next_obs=next_obs,
        infos=_merge_rollout_infos(info_hist),
    )


def make_sb3_vec_env(vector_cfg: VectorEnvConfig) -> Any:
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "stable_baselines3 is not installed. Install it to use make_sb3_vec_env()."
        ) from exc
    if int(vector_cfg.num_envs) <= 0:
        raise ValueError("num_envs must be positive.")
    env_fns = [make_env_fn(vector_cfg.env_cfg) for _ in range(int(vector_cfg.num_envs))]
    vec_cls = SubprocVecEnv if bool(vector_cfg.parallel) else DummyVecEnv
    return vec_cls(env_fns)


def build_sb3_env_fns(vector_cfg: VectorEnvConfig) -> list[EnvFactory]:
    if int(vector_cfg.num_envs) <= 0:
        raise ValueError("num_envs must be positive.")
    return [make_env_fn(vector_cfg.env_cfg) for _ in range(int(vector_cfg.num_envs))]


def _merge_rollout_infos(info_hist: list[dict[str, list[Any]]]) -> dict[str, list[Any]]:
    if not info_hist:
        return {}
    out: dict[str, list[Any]] = {}
    for info in info_hist:
        for key, value in info.items():
            out.setdefault(key, []).append(value)
    return out


def collect_multi_agent_rollout(
    env: Any,
    *,
    policy_fns_by_agent: dict[str, Callable[[np.ndarray], np.ndarray]],
    horizon: int,
    initial_obs_by_agent: dict[str, np.ndarray] | None = None,
    reset_kwargs: dict[str, Any] | None = None,
) -> MultiAgentRolloutBatch:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if initial_obs_by_agent is None:
        obs_by_agent, _ = env.reset(**dict(reset_kwargs or {}))
    else:
        obs_by_agent = {
            agent_id: np.array(obs, dtype=np.float32)
            for agent_id, obs in dict(initial_obs_by_agent).items()
        }
    agent_ids = tuple(sorted(obs_by_agent.keys()))
    obs_hist = {agent_id: [] for agent_id in agent_ids}
    action_hist = {agent_id: [] for agent_id in agent_ids}
    reward_hist = {agent_id: [] for agent_id in agent_ids}
    terminated_hist = {agent_id: [] for agent_id in agent_ids}
    truncated_hist = {agent_id: [] for agent_id in agent_ids}
    info_hist = {agent_id: [] for agent_id in agent_ids}
    next_obs_by_agent = {
        agent_id: np.array(obs_by_agent[agent_id], dtype=np.float32)
        for agent_id in agent_ids
    }

    for _ in range(int(horizon)):
        actions = {
            agent_id: np.array(policy_fns_by_agent[agent_id](next_obs_by_agent[agent_id]), dtype=np.float32)
            for agent_id in agent_ids
        }
        step_obs, rewards, terminated, truncated, infos = env.step(actions)
        for agent_id in agent_ids:
            obs_hist[agent_id].append(np.array(next_obs_by_agent[agent_id], dtype=np.float32))
            action_hist[agent_id].append(np.array(actions[agent_id], dtype=np.float32))
            reward_hist[agent_id].append(float(rewards[agent_id]))
            terminated_hist[agent_id].append(bool(terminated[agent_id]))
            truncated_hist[agent_id].append(bool(truncated[agent_id]))
            info_hist[agent_id].append(dict(infos[agent_id]))
            next_obs_by_agent[agent_id] = np.array(step_obs[agent_id], dtype=np.float32)

    return MultiAgentRolloutBatch(
        obs_by_agent={agent_id: np.stack(hist, axis=0) for agent_id, hist in obs_hist.items()},
        actions_by_agent={agent_id: np.stack(hist, axis=0) for agent_id, hist in action_hist.items()},
        rewards_by_agent={agent_id: np.array(hist, dtype=np.float32) for agent_id, hist in reward_hist.items()},
        terminated_by_agent={agent_id: np.array(hist, dtype=bool) for agent_id, hist in terminated_hist.items()},
        truncated_by_agent={agent_id: np.array(hist, dtype=bool) for agent_id, hist in truncated_hist.items()},
        next_obs_by_agent=next_obs_by_agent,
        infos_by_agent={
            agent_id: _merge_multi_agent_infos(hist)
            for agent_id, hist in info_hist.items()
        },
    )


def _merge_multi_agent_infos(info_hist: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if not info_hist:
        return {}
    out: dict[str, list[Any]] = {}
    for info in info_hist:
        for key, value in info.items():
            out.setdefault(key, []).append(value)
    return out
