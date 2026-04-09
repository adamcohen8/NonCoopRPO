import importlib

from machine_learning.gym_env import (
    ActionField,
    AsyncVectorSimulationEnv,
    DirectActionAdapter,
    EnvFactory,
    GymEnvConfig,
    GymSimulationEnv,
    MultiAgentEnvConfig,
    MultiAgentSimulationEnv,
    ObservationField,
    RangeTermination,
    RelativeDistanceReward,
    SyncVectorSimulationEnv,
    ThrustVectorToPointingAdapter,
    VectorEnvConfig,
    make_env_fn,
    make_vector_env,
)
from machine_learning.training_adapter import (
    MultiAgentRolloutBatch,
    RolloutBatch,
    build_sb3_env_fns,
    collect_multi_agent_rollout,
    collect_vector_rollout,
    make_sb3_vec_env,
)
from machine_learning.self_play import (
    LinearPolicy,
    OpponentPool,
    SelfPlayTrainerConfig,
    evaluate_self_play_policies,
    run_self_play_training,
    summarize_multi_agent_batch,
)

__all__ = [
    "RLRendezvousConfig",
    "RLRendezvousEnv",
    "PPOConfig",
    "PPOLightningModule",
    "AttitudeRICRLConfig",
    "AttitudeRICRLEnv",
    "AttitudeRICPPOConfig",
    "AttitudeRICPPOLightningModule",
    "ObservationField",
    "ActionField",
    "GymEnvConfig",
    "MultiAgentEnvConfig",
    "VectorEnvConfig",
    "EnvFactory",
    "GymSimulationEnv",
    "MultiAgentSimulationEnv",
    "SyncVectorSimulationEnv",
    "AsyncVectorSimulationEnv",
    "DirectActionAdapter",
    "ThrustVectorToPointingAdapter",
    "RelativeDistanceReward",
    "RangeTermination",
    "make_env_fn",
    "make_vector_env",
    "RolloutBatch",
    "MultiAgentRolloutBatch",
    "collect_vector_rollout",
    "collect_multi_agent_rollout",
    "build_sb3_env_fns",
    "make_sb3_vec_env",
    "LinearPolicy",
    "OpponentPool",
    "SelfPlayTrainerConfig",
    "summarize_multi_agent_batch",
    "evaluate_self_play_policies",
    "run_self_play_training",
]


def _load_optional_attr(*, module_name: str, attr_names: dict[str, str], requested_name: str) -> object:
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Optional ML dependency is missing while loading '{requested_name}'. "
            f"Install ML dependencies with `python -m pip install -r requirements-ml.txt`."
        ) from exc
    return getattr(mod, attr_names[requested_name])


def __getattr__(name: str):
    if name in {"RLRendezvousConfig", "RLRendezvousEnv"}:
        mod = importlib.import_module("machine_learning.rendezvous_env")
        return getattr(mod, name)
    if name in {"PPOConfig", "PPOLightningModule"}:
        return _load_optional_attr(
            module_name="machine_learning.ppo_lightning",
            attr_names={
                "PPOConfig": "PPOConfig",
                "PPOLightningModule": "PPOLightningModule",
            },
            requested_name=name,
        )
    if name in {"AttitudeRICRLConfig", "AttitudeRICRLEnv"}:
        mod = importlib.import_module("machine_learning.attitude_ric_env")
        return getattr(mod, name)
    if name in {"AttitudeRICPPOConfig", "AttitudeRICPPOLightningModule"}:
        return _load_optional_attr(
            module_name="machine_learning.attitude_ric_ppo",
            attr_names={
                "AttitudeRICPPOConfig": "AttitudeRICPPOConfig",
                "AttitudeRICPPOLightningModule": "AttitudeRICPPOLightningModule",
            },
            requested_name=name,
        )
    raise AttributeError(name)
