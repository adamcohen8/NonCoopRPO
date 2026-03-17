from __future__ import annotations

import importlib
from typing import Any


def run_free_tumble_one_orbit(*args: Any, **kwargs: Any):
    from sim.scenarios.free_tumble_one_orbit import run_free_tumble_one_orbit as _run

    return _run(*args, **kwargs)


def run_free_tumble_one_orbit_ric(*args: Any, **kwargs: Any):
    from sim.scenarios.free_tumble_one_orbit_ric import run_free_tumble_one_orbit_ric as _run

    return _run(*args, **kwargs)


def run_full_stack_demo(*args: Any, **kwargs: Any):
    from sim.scenarios.full_stack_demo import run_full_stack_demo as _run

    return _run(*args, **kwargs)


def run_monte_carlo(*args: Any, **kwargs: Any):
    from sim.scenarios.monte_carlo import run_monte_carlo as _run

    return _run(*args, **kwargs)


def run_asat_phased_engagement(*args: Any, **kwargs: Any):
    from sim.scenarios.asat_phased_engagement import run_asat_phased_engagement as _run

    return _run(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name in {"ASATPhasedScenarioConfig", "AgentStrategyConfig", "KnowledgeGateConfig"}:
        _asat = importlib.import_module("sim.scenarios.asat_phased_engagement")

        return getattr(_asat, name)
    if name == "MonteCarloConfig":
        _mc = importlib.import_module("sim.scenarios.monte_carlo")

        return getattr(_mc, name)
    raise AttributeError(f"module 'sim.scenarios' has no attribute '{name}'")

__all__ = [
    "run_free_tumble_one_orbit",
    "run_free_tumble_one_orbit_ric",
    "run_full_stack_demo",
    "MonteCarloConfig",
    "run_monte_carlo",
    "ASATPhasedScenarioConfig",
    "AgentStrategyConfig",
    "KnowledgeGateConfig",
    "run_asat_phased_engagement",
]
