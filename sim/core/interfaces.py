from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Command, Measurement, StateBelief, StateTruth


class DynamicsModel(ABC):
    @abstractmethod
    def step(self, state: StateTruth, command: Command, env: dict, dt_s: float) -> StateTruth:
        raise NotImplementedError


class SensorModel(ABC):
    @abstractmethod
    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement | None:
        raise NotImplementedError


class Estimator(ABC):
    @abstractmethod
    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        raise NotImplementedError


class Controller(ABC):
    @abstractmethod
    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        raise NotImplementedError


class Actuator(ABC):
    @abstractmethod
    def apply(self, command: Command, limits: dict, dt_s: float) -> Command:
        raise NotImplementedError
