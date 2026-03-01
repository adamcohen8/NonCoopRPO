from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import SensorModel
from sim.core.models import Measurement, StateTruth
from sim.sensors.access import AccessModel


@dataclass
class SensorNoiseConfig:
    sigma: np.ndarray
    bias: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dropout_prob: float = 0.0
    latency_s: float = 0.0


@dataclass
class OwnStateSensor(SensorModel):
    noise: SensorNoiseConfig
    rng: np.random.Generator
    access_model: AccessModel | None = None
    _latency_queue: list[tuple[float, np.ndarray]] = field(default_factory=list)

    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement | None:
        if self.access_model is not None and not self.access_model.can_update(truth.position_eci_km, truth.position_eci_km, t_s):
            return None
        if self.rng.random() < self.noise.dropout_prob:
            return None

        x = np.hstack((truth.position_eci_km, truth.velocity_eci_km_s))
        bias = self._bias(len(x))
        sigma = self._sigma(len(x))
        z = x + bias + self.rng.normal(0.0, sigma, size=len(x))

        if self.noise.latency_s <= 0.0:
            return Measurement(vector=z, t_s=t_s)

        release_t = t_s + self.noise.latency_s
        self._latency_queue.append((release_t, z))
        for i, (ready_t, vec) in enumerate(self._latency_queue):
            if ready_t <= t_s:
                self._latency_queue.pop(i)
                return Measurement(vector=vec, t_s=t_s)
        return None

    def _sigma(self, n: int) -> np.ndarray:
        if self.noise.sigma.size == n:
            return self.noise.sigma
        return np.full(n, float(self.noise.sigma[0]))

    def _bias(self, n: int) -> np.ndarray:
        if self.noise.bias.size == n:
            return self.noise.bias
        if self.noise.bias.size == 0:
            return np.zeros(n)
        return np.full(n, float(self.noise.bias[0]))


@dataclass
class RelativeSensor(SensorModel):
    target_id: str
    mode: str
    noise: SensorNoiseConfig
    rng: np.random.Generator
    access_model: AccessModel | None = None

    def measure(self, truth: StateTruth, env: dict, t_s: float) -> Measurement | None:
        world = env.get("world_truth", {})
        target = world.get(self.target_id)
        if target is None:
            return None

        if self.access_model is not None and not self.access_model.can_update(truth.position_eci_km, target.position_eci_km, t_s):
            return None
        if self.rng.random() < self.noise.dropout_prob:
            return None

        rel_r = target.position_eci_km - truth.position_eci_km
        rel_v = target.velocity_eci_km_s - truth.velocity_eci_km_s
        rng_km = np.linalg.norm(rel_r)
        if rng_km == 0.0:
            los = np.zeros(3)
        else:
            los = rel_r / rng_km

        if self.mode == "angle_only":
            az = np.arctan2(los[1], los[0])
            el = np.arcsin(np.clip(los[2], -1.0, 1.0))
            z = np.array([az, el])
        elif self.mode == "range":
            z = np.array([rng_km])
        elif self.mode == "range_rate":
            rr = 0.0 if rng_km == 0.0 else np.dot(rel_v, los)
            z = np.array([rng_km, rr])
        else:
            raise ValueError(f"unsupported relative sensor mode: {self.mode}")

        sigma = self._sigma(len(z))
        bias = self._bias(len(z))
        z = z + bias + self.rng.normal(0.0, sigma, size=len(z))
        return Measurement(vector=z, t_s=t_s)

    def _sigma(self, n: int) -> np.ndarray:
        if self.noise.sigma.size == n:
            return self.noise.sigma
        return np.full(n, float(self.noise.sigma[0]))

    def _bias(self, n: int) -> np.ndarray:
        if self.noise.bias.size == n:
            return self.noise.bias
        if self.noise.bias.size == 0:
            return np.zeros(n)
        return np.full(n, float(self.noise.bias[0]))
