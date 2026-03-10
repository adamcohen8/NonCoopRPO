from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.optimization.base import ObjectiveFn, OptimizationResult, ParameterBound


@dataclass(frozen=True)
class PSOConfig:
    particles: int = 24
    iterations: int = 50
    inertia: float = 0.72
    cognitive: float = 1.49
    social: float = 1.49

    def __post_init__(self) -> None:
        if self.particles <= 0:
            raise ValueError("particles must be positive.")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive.")
        if self.inertia < 0.0:
            raise ValueError("inertia must be non-negative.")
        if self.cognitive < 0.0 or self.social < 0.0:
            raise ValueError("cognitive and social terms must be non-negative.")


class ParticleSwarmOptimizer:
    """Simple bounded PSO for continuous parameter vectors."""

    def __init__(self, cfg: PSOConfig):
        self.cfg = cfg

    def optimize(self, objective: ObjectiveFn, bounds: list[ParameterBound], seed: int = 0) -> OptimizationResult:
        if not bounds:
            raise ValueError("At least one parameter bound is required.")

        rng = np.random.default_rng(seed)
        dim = len(bounds)
        lb = np.array([b.lower for b in bounds], dtype=float)
        ub = np.array([b.upper for b in bounds], dtype=float)
        span = ub - lb

        x = lb + rng.random((self.cfg.particles, dim)) * span
        v = rng.uniform(-0.2, 0.2, size=(self.cfg.particles, dim)) * span

        pbest_x = x.copy()
        pbest_f = np.array([objective(xi) for xi in x], dtype=float)
        g_idx = int(np.argmin(pbest_f))
        gbest_x = pbest_x[g_idx].copy()
        gbest_f = float(pbest_f[g_idx])
        history = [gbest_f]

        for _ in range(self.cfg.iterations):
            r1 = rng.random((self.cfg.particles, dim))
            r2 = rng.random((self.cfg.particles, dim))
            v = (
                self.cfg.inertia * v
                + self.cfg.cognitive * r1 * (pbest_x - x)
                + self.cfg.social * r2 * (gbest_x[None, :] - x)
            )
            x = x + v
            x = np.clip(x, lb, ub)

            f = np.array([objective(xi) for xi in x], dtype=float)
            improved = f < pbest_f
            if np.any(improved):
                pbest_x[improved, :] = x[improved, :]
                pbest_f[improved] = f[improved]
                g_idx = int(np.argmin(pbest_f))
                gbest_x = pbest_x[g_idx].copy()
                gbest_f = float(pbest_f[g_idx])
            history.append(gbest_f)

        return OptimizationResult(
            best_x=gbest_x,
            best_cost=gbest_f,
            history_best_cost=history,
            metadata={
                "optimizer": "pso",
                "particles": self.cfg.particles,
                "iterations": self.cfg.iterations,
                "inertia": self.cfg.inertia,
                "cognitive": self.cfg.cognitive,
                "social": self.cfg.social,
            },
        )
