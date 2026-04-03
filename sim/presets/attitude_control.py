from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReactionWheelPreset:
    name: str
    axis_body: np.ndarray
    max_torque_nm: float
    max_momentum_nms: float


@dataclass(frozen=True)
class ReactionWheelAssemblyPreset:
    name: str
    wheels: tuple[ReactionWheelPreset, ...]


BASIC_REACTION_WHEEL_X = ReactionWheelPreset(
    name="RW-X",
    axis_body=np.array([1.0, 0.0, 0.0]),
    max_torque_nm=0.05,
    max_momentum_nms=0.20,
)

BASIC_REACTION_WHEEL_Y = ReactionWheelPreset(
    name="RW-Y",
    axis_body=np.array([0.0, 1.0, 0.0]),
    max_torque_nm=0.05,
    max_momentum_nms=0.20,
)

BASIC_REACTION_WHEEL_Z = ReactionWheelPreset(
    name="RW-Z",
    axis_body=np.array([0.0, 0.0, 1.0]),
    max_torque_nm=0.05,
    max_momentum_nms=0.20,
)

BASIC_REACTION_WHEEL_TRIAD = ReactionWheelAssemblyPreset(
    name="Basic Principal-Axis Reaction Wheel Triad",
    wheels=(BASIC_REACTION_WHEEL_X, BASIC_REACTION_WHEEL_Y, BASIC_REACTION_WHEEL_Z),
)
