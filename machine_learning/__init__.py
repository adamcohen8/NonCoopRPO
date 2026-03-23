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


def __getattr__(name: str):
    if name in {"RLRendezvousConfig", "RLRendezvousEnv"}:
        from machine_learning.rendezvous_env import RLRendezvousConfig, RLRendezvousEnv

        return {
            "RLRendezvousConfig": RLRendezvousConfig,
            "RLRendezvousEnv": RLRendezvousEnv,
        }[name]
    if name in {"PPOConfig", "PPOLightningModule"}:
        from machine_learning.ppo_lightning import PPOConfig, PPOLightningModule

        return {
            "PPOConfig": PPOConfig,
            "PPOLightningModule": PPOLightningModule,
        }[name]
    if name in {"AttitudeRICRLConfig", "AttitudeRICRLEnv"}:
        from machine_learning.attitude_ric_env import AttitudeRICRLConfig, AttitudeRICRLEnv

        return {
            "AttitudeRICRLConfig": AttitudeRICRLConfig,
            "AttitudeRICRLEnv": AttitudeRICRLEnv,
        }[name]
    if name in {"AttitudeRICPPOConfig", "AttitudeRICPPOLightningModule"}:
        from machine_learning.attitude_ric_ppo import AttitudeRICPPOConfig, AttitudeRICPPOLightningModule

        return {
            "AttitudeRICPPOConfig": AttitudeRICPPOConfig,
            "AttitudeRICPPOLightningModule": AttitudeRICPPOLightningModule,
        }[name]
    raise AttributeError(name)
