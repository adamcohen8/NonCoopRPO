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
    MultiAgentEnvConfig,
    MultiAgentSimulationEnv,
    ObservationField,
    RelativeDistanceReward,
    SelfPlayTrainerConfig,
    ThrustVectorToPointingAdapter,
    LinearPolicy,
    run_self_play_training,
)
from sim.config import MonteCarloVariation


def build_multi_agent_scenario(duration_s: float, dt_s: float) -> dict:
    return {
        "scenario_name": "multi_agent_self_play_demo",
        "rocket": {"enabled": False},
        "target": {
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
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
                "params": {"kp": [0.1, 0.1, 0.1], "kd": [0.2, 0.2, 0.2]},
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
                "params": {"kp": [0.1, 0.1, 0.1], "kd": [0.2, 0.2, 0.2]},
            },
        },
        "simulator": {
            "duration_s": float(duration_s),
            "dt_s": float(dt_s),
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": True, "attitude_substep_s": min(0.1, float(dt_s))}},
        },
        "outputs": {"output_dir": "outputs/multi_agent_self_play_demo", "mode": "save"},
        "metadata": {"seed": 17},
    }


def build_multi_agent_env_config(duration_s: float, dt_s: float) -> MultiAgentEnvConfig:
    shared_obs = (
        ObservationField("truth.chaser.position_eci_km"),
        ObservationField("truth.target.position_eci_km"),
        ObservationField("truth.chaser.velocity_eci_km_s"),
        ObservationField("truth.target.velocity_eci_km_s"),
        ObservationField("metrics.range_km"),
        ObservationField("metrics.closest_range_km"),
    )
    shared_actions = (
        ActionField("thrust_direction_eci[0]", -1.0, 1.0),
        ActionField("thrust_direction_eci[1]", -1.0, 1.0),
        ActionField("thrust_direction_eci[2]", -1.0, 1.0),
        ActionField("throttle", 0.0, 1.0),
    )
    return MultiAgentEnvConfig(
        scenario=build_multi_agent_scenario(duration_s=duration_s, dt_s=dt_s),
        controlled_agent_ids=("chaser", "target"),
        observation_fields_by_agent={
            "chaser": shared_obs,
            "target": shared_obs,
        },
        action_fields_by_agent={
            "chaser": shared_actions,
            "target": shared_actions,
        },
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
        action_adapters_by_agent={
            "chaser": ThrustVectorToPointingAdapter(),
            "target": ThrustVectorToPointingAdapter(),
        },
        reward_fns_by_agent={
            "chaser": RelativeDistanceReward(controlled_agent_id="chaser", target_id="target", sign=1.0, scale=1000.0),
            "target": RelativeDistanceReward(controlled_agent_id="target", target_id="chaser", sign=-1.0, scale=1000.0),
        },
    )


def run_self_play(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(int(args.seed))
    env = MultiAgentSimulationEnv(build_multi_agent_env_config(duration_s=args.duration_s, dt_s=args.dt))
    obs, _ = env.reset(seed=int(args.seed))
    chaser_policy = LinearPolicy.random(
        obs_dim=int(obs["chaser"].size),
        action_dim=4,
        rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
    )
    target_policy = LinearPolicy.random(
        obs_dim=int(obs["target"].size),
        action_dim=4,
        rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
    )
    run_self_play_training(
        env,
        policies_by_agent={
            "chaser": chaser_policy,
            "target": target_policy,
        },
        trainer_cfg=SelfPlayTrainerConfig(
            update_mode=str(args.update_mode),
            iterations=int(args.iterations),
            rollout_horizon=int(args.rollout_horizon),
            learning_rate=float(args.learning_rate),
            mutation_sigma=float(args.mutation_sigma),
            snapshot_interval=int(args.snapshot_interval),
            max_opponents=int(args.max_opponents),
            seed=int(args.seed),
        ),
        log_fn=print,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent self-play demo using the shared simulator environment.")
    parser.add_argument("--update-mode", choices=("alternating", "simultaneous"), default="alternating")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rollout-horizon", type=int, default=32)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--mutation-sigma", type=float, default=0.01)
    parser.add_argument("--snapshot-interval", type=int, default=1)
    parser.add_argument("--max-opponents", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    run_self_play(args)


if __name__ == "__main__":
    main()
