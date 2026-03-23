from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machine_learning import (
    ActionField,
    GymEnvConfig,
    ObservationField,
    RelativeDistanceReward,
    RangeTermination,
    ThrustVectorToPointingAdapter,
    VectorEnvConfig,
    collect_vector_rollout,
    make_sb3_vec_env,
    make_vector_env,
)
from sim.config import MonteCarloVariation


def build_demo_scenario(duration_s: float, dt_s: float) -> dict:
    return {
        "scenario_name": "generic_rl_training_demo",
        "rocket": {"enabled": False},
        "target": {
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
            },
        },
        "chaser": {
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "relative_to_target_ric": {"frame": "rect", "state": [1.0, -2.0, 0.0, 0.0, 0.0, 0.0]},
                "attitude_quat_bn": [1.0, 0.0, 0.0, 0.0],
            },
            "mission_execution": {
                "module": "sim.mission.modules",
                "class_name": "ControllerPointingExecution",
                "params": {"alignment_tolerance_deg": 180.0},
            },
            "attitude_control": {
                "module": "sim.control.attitude.baseline",
                "class_name": "ReactionWheelPDController",
                "params": {
                    "kp": [0.1, 0.1, 0.1],
                    "kd": [0.2, 0.2, 0.2],
                },
            },
        },
        "simulator": {
            "duration_s": float(duration_s),
            "dt_s": float(dt_s),
            "termination": {"earth_impact_enabled": False},
            "dynamics": {
                "attitude": {
                    "enabled": True,
                    "attitude_substep_s": min(0.1, float(dt_s)),
                }
            },
        },
        "outputs": {"output_dir": "outputs/generic_rl_env", "mode": "save"},
        "metadata": {"seed": 7},
    }


def build_env_config(duration_s: float, dt_s: float) -> GymEnvConfig:
    return GymEnvConfig(
        scenario=build_demo_scenario(duration_s=duration_s, dt_s=dt_s),
        controlled_agent_id="chaser",
        observation_fields=(
            ObservationField("truth.chaser.position_eci_km"),
            ObservationField("truth.target.position_eci_km"),
            ObservationField("truth.chaser.velocity_eci_km_s"),
            ObservationField("truth.target.velocity_eci_km_s"),
            ObservationField("truth.chaser.attitude_quat_bn"),
            ObservationField("metrics.range_km"),
            ObservationField("metrics.closest_range_km"),
        ),
        action_fields=(
            ActionField("thrust_direction_eci[0]", -1.0, 1.0),
            ActionField("thrust_direction_eci[1]", -1.0, 1.0),
            ActionField("thrust_direction_eci[2]", -1.0, 1.0),
            ActionField("throttle", 0.0, 1.0),
        ),
        episode_variations=(
            MonteCarloVariation(
                parameter_path="chaser.initial_state.relative_to_target_ric.state[0]",
                mode="uniform",
                low=0.5,
                high=2.5,
            ),
            MonteCarloVariation(
                parameter_path="chaser.initial_state.relative_to_target_ric.state[1]",
                mode="uniform",
                low=-4.0,
                high=-1.0,
            ),
        ),
        action_adapter=ThrustVectorToPointingAdapter(),
        reward_fn=RelativeDistanceReward(scale=1000.0, terminal_bonus=5.0),
        termination_fn=RangeTermination(capture_radius_km=0.05),
    )


def random_policy(obs: np.ndarray) -> np.ndarray:
    batch = int(obs.shape[0])
    acts = np.random.uniform(low=-1.0, high=1.0, size=(batch, 4)).astype(np.float32)
    acts[:, 3] = np.random.uniform(low=0.0, high=1.0, size=batch).astype(np.float32)
    return acts


def zero_policy(obs: np.ndarray) -> np.ndarray:
    return np.zeros((int(obs.shape[0]), 4), dtype=np.float32)


def run_rollout_backend(args: argparse.Namespace) -> None:
    env_cfg = build_env_config(duration_s=args.duration_s, dt_s=args.dt)
    vec_env = make_vector_env(
        VectorEnvConfig(
            env_cfg=env_cfg,
            num_envs=int(args.num_envs),
            parallel=bool(args.parallel),
            auto_reset=True,
        )
    )
    try:
        policy = random_policy if args.policy == "random" else zero_policy
        batch = collect_vector_rollout(
            vec_env,
            policy_fn=policy,
            horizon=int(args.rollout_horizon),
            reset_kwargs={"seed": int(args.seed)},
        )
        print("Collected rollout batch.")
        print(f"obs shape: {batch.obs.shape}")
        print(f"actions shape: {batch.actions.shape}")
        print(f"rewards shape: {batch.rewards.shape}")
        print(f"mean reward: {float(np.mean(batch.rewards)):.6f}")
        print(f"terminal count: {int(np.sum(batch.terminated))}")
        print(f"truncated count: {int(np.sum(batch.truncated))}")
    finally:
        vec_env.close()


def run_sb3_backend(args: argparse.Namespace) -> None:
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable_baselines3 is not installed. Install it to use --backend sb3."
        ) from exc

    env_cfg = build_env_config(duration_s=args.duration_s, dt_s=args.dt)
    vec_env = make_sb3_vec_env(
        VectorEnvConfig(
            env_cfg=env_cfg,
            num_envs=int(args.num_envs),
            parallel=bool(args.parallel),
        )
    )
    outdir = REPO_ROOT / "outputs" / "ml" / "generic_rl_env"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=max(8, int(args.rollout_horizon)),
            batch_size=max(8, int(min(args.rollout_horizon * args.num_envs, 256))),
            seed=int(args.seed),
        )
        model.learn(total_timesteps=int(args.total_timesteps))
        save_path = outdir / "sb3_ppo_generic_rl_env"
        model.save(str(save_path))
        print(f"Saved SB3 model to {save_path}.zip")
    finally:
        vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic RL training example for the configurable simulator Gym env.")
    parser.add_argument("--backend", choices=("rollout", "sb3"), default="rollout")
    parser.add_argument("--policy", choices=("zero", "random"), default="random", help="Used by rollout backend only.")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--parallel", action="store_true", help="Use subprocess vector rollout workers.")
    parser.add_argument("--rollout-horizon", type=int, default=32)
    parser.add_argument("--total-timesteps", type=int, default=4096, help="Used by SB3 backend only.")
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    if args.backend == "sb3":
        run_sb3_backend(args)
        return
    run_rollout_backend(args)


if __name__ == "__main__":
    main()
