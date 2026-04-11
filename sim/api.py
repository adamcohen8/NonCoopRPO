from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig, load_simulation_yaml, scenario_config_from_dict
from sim.execution import create_single_run_engine, run_simulation_scenario


def _closest_approach_metric(payload: dict[str, Any]) -> float:
    from sim.master_simulator import _closest_approach_from_run_payload

    return _closest_approach_from_run_payload(payload)


def _as_array_map(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, np.ndarray] = {}
    for key, arr in value.items():
        try:
            out[str(key)] = np.array(arr, dtype=float)
        except (TypeError, ValueError):
            continue
    return out


def _as_nested_array_map(value: Any) -> dict[str, dict[str, np.ndarray]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, np.ndarray]] = {}
    for key, inner in value.items():
        if not isinstance(inner, dict):
            continue
        out[str(key)] = _as_array_map(inner)
    return out


@dataclass(frozen=True)
class SimulationConfig:
    scenario: SimulationScenarioConfig
    source_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationConfig":
        resolved = Path(path).expanduser().resolve()
        return cls(scenario=load_simulation_yaml(resolved), source_path=resolved)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        return cls(scenario=scenario_config_from_dict(dict(data)))

    @property
    def scenario_name(self) -> str:
        return str(self.scenario.scenario_name)

    def to_dict(self) -> dict[str, Any]:
        return self.scenario.to_dict()

    def to_scenario_config(self) -> SimulationScenarioConfig:
        return self.scenario

    def with_seed(self, seed: int) -> "SimulationConfig":
        root = self.to_dict()
        root.setdefault("metadata", {})["seed"] = int(seed)
        return SimulationConfig(
            scenario=scenario_config_from_dict(root),
            source_path=self.source_path,
        )


@dataclass(frozen=True)
class SimulationSnapshot:
    step_index: int
    time_s: float
    truth: dict[str, np.ndarray]
    belief: dict[str, np.ndarray]
    applied_thrust: dict[str, np.ndarray]
    applied_torque: dict[str, np.ndarray]

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.truth.keys()))


@dataclass
class SimulationResult:
    config: SimulationConfig
    payload: dict[str, Any]

    @property
    def analysis(self) -> dict[str, Any]:
        return dict(self.payload.get("analysis", {}) or {})

    @property
    def analysis_study_type(self) -> str:
        analysis = self.analysis
        if bool(analysis.get("enabled", False)):
            return str(analysis.get("study_type", "unknown"))
        if self.is_monte_carlo:
            return "monte_carlo"
        return "single_run"

    @property
    def is_batch_analysis(self) -> bool:
        return self.analysis_study_type in {"monte_carlo", "sensitivity"}

    @property
    def is_monte_carlo(self) -> bool:
        return bool(dict(self.payload.get("monte_carlo", {}) or {}).get("enabled", False))

    @property
    def summary(self) -> dict[str, Any]:
        if isinstance(self.payload.get("summary"), dict):
            return dict(self.payload["summary"])
        if isinstance(self.payload.get("run"), dict):
            return dict(self.payload["run"])
        return {}

    @property
    def time_s(self) -> np.ndarray:
        return np.array(self.payload.get("time_s", []), dtype=float).reshape(-1)

    @property
    def truth(self) -> dict[str, np.ndarray]:
        return _as_array_map(self.payload.get("truth_by_object", {}))

    @property
    def target_reference_orbit(self) -> np.ndarray:
        arr = np.array(self.payload.get("target_reference_orbit_truth", []), dtype=float)
        if arr.size == 0:
            return np.empty((0, 6), dtype=float)
        if arr.ndim == 1:
            return arr.reshape(-1, 6)
        return arr

    @property
    def belief(self) -> dict[str, np.ndarray]:
        return _as_array_map(self.payload.get("belief_by_object", {}))

    @property
    def applied_thrust(self) -> dict[str, np.ndarray]:
        return _as_array_map(self.payload.get("applied_thrust_by_object", {}))

    @property
    def applied_torque(self) -> dict[str, np.ndarray]:
        return _as_array_map(self.payload.get("applied_torque_by_object", {}))

    @property
    def knowledge(self) -> dict[str, dict[str, np.ndarray]]:
        return _as_nested_array_map(self.payload.get("knowledge_by_observer", {}))

    @property
    def artifacts(self) -> dict[str, Any]:
        if self.is_batch_analysis:
            return dict(self.payload.get("artifacts", {}) or {})
        summary = self.summary
        return {
            "plots": dict(summary.get("plot_outputs", {}) or {}),
            "animations": dict(summary.get("animation_outputs", {}) or {}),
        }

    @property
    def metrics(self) -> dict[str, Any]:
        if self.is_monte_carlo:
            return dict(self.payload.get("aggregate_stats", {}) or {})
        if self.analysis_study_type == "sensitivity":
            return {
                "parameter_count": int(self.analysis.get("parameter_count", 0)),
                "run_count": int(self.analysis.get("run_count", 0)),
                "metrics": list(self.analysis.get("metrics", []) or []),
            }
        out = dict(self.summary)
        closest_approach_km = _closest_approach_metric(self.payload)
        if np.isfinite(closest_approach_km):
            out["closest_approach_km"] = float(closest_approach_km)
        return out

    @property
    def num_steps(self) -> int:
        return int(self.time_s.size)

    def snapshot(self, step_index: int) -> SimulationSnapshot:
        if self.is_batch_analysis:
            raise RuntimeError("Snapshots are only available for single-run results.")
        if step_index < 0 or step_index >= self.num_steps:
            raise IndexError(f"step_index {step_index} is out of range for {self.num_steps} samples.")

        truth = {oid: np.array(hist[step_index], dtype=float) for oid, hist in self.truth.items() if hist.shape[0] > step_index}
        belief = {oid: np.array(hist[step_index], dtype=float) for oid, hist in self.belief.items() if hist.shape[0] > step_index}
        thrust = {
            oid: np.array(hist[step_index], dtype=float)
            for oid, hist in self.applied_thrust.items()
            if hist.shape[0] > step_index
        }
        torque = {
            oid: np.array(hist[step_index], dtype=float)
            for oid, hist in self.applied_torque.items()
            if hist.shape[0] > step_index
        }
        return SimulationSnapshot(
            step_index=int(step_index),
            time_s=float(self.time_s[step_index]),
            truth=truth,
            belief=belief,
            applied_thrust=thrust,
            applied_torque=torque,
        )


class SimulationSession:
    def __init__(self, config: SimulationConfig | SimulationScenarioConfig | dict[str, Any]):
        self._base_config = self._coerce_config(config)
        self._active_config = self._base_config
        self._result: SimulationResult | None = None
        self._step_index = 0
        self._done = False
        self._engine: Any | None = None

    @classmethod
    def from_config(cls, config: SimulationConfig | SimulationScenarioConfig | dict[str, Any]) -> "SimulationSession":
        return cls(config)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationSession":
        return cls(SimulationConfig.from_yaml(path))

    @staticmethod
    def _coerce_config(config: SimulationConfig | SimulationScenarioConfig | dict[str, Any]) -> SimulationConfig:
        if isinstance(config, SimulationConfig):
            return config
        if isinstance(config, SimulationScenarioConfig):
            return SimulationConfig(config)
        if isinstance(config, dict):
            return SimulationConfig.from_dict(config)
        raise TypeError(f"Unsupported config type: {type(config)!r}")

    @property
    def config(self) -> SimulationConfig:
        return self._active_config

    @property
    def result(self) -> SimulationResult | None:
        return self._result

    @property
    def done(self) -> bool:
        if self._engine is not None:
            return bool(self._engine.done)
        return bool(self._done)

    def reset(self, seed: int | None = None) -> SimulationSnapshot | None:
        self._active_config = self._base_config.with_seed(seed) if seed is not None else self._base_config
        self._result = None
        self._step_index = 0
        self._done = False
        self._engine = None
        if self._is_batch_analysis(self._active_config.scenario):
            return None
        self._ensure_engine()
        assert self._engine is not None
        snap = self._engine.snapshot(0)
        return SimulationSnapshot(
            step_index=int(snap["step_index"]),
            time_s=float(snap["time_s"]),
            truth=dict(snap["truth"]),
            belief=dict(snap["belief"]),
            applied_thrust=dict(snap["applied_thrust"]),
            applied_torque=dict(snap["applied_torque"]),
        )

    def run(self, *, step_callback: Any | None = None) -> SimulationResult:
        if self._is_batch_analysis(self._active_config.scenario):
            payload = self._run_batch_analysis(self._active_config)
            self._result = SimulationResult(config=self._active_config, payload=payload)
            self._done = True
            return self._result

        self._ensure_engine(step_callback=step_callback)
        assert self._engine is not None
        payload = self._engine.run()
        self._result = SimulationResult(config=self._active_config, payload=payload)
        self._step_index = max(self._result.num_steps - 1, 0)
        self._done = True
        return self._result

    def step(self) -> SimulationSnapshot:
        if self._is_batch_analysis(self._active_config.scenario):
            raise RuntimeError("SimulationSession.step() is only available for single-run scenarios.")
        self._ensure_engine()
        assert self._engine is not None
        snap = self._engine.step()
        self._step_index = int(snap["step_index"])
        self._done = bool(self._engine.done)
        return SimulationSnapshot(
            step_index=int(snap["step_index"]),
            time_s=float(snap["time_s"]),
            truth=dict(snap["truth"]),
            belief=dict(snap["belief"]),
            applied_thrust=dict(snap["applied_thrust"]),
            applied_torque=dict(snap["applied_torque"]),
        )

    def _ensure_engine(self, *, step_callback: Any | None = None) -> None:
        if self._engine is not None:
            if step_callback is not None:
                self._engine.active_step_callback = step_callback
                emit = getattr(self._engine, "_emit_step_callback", None)
                if callable(emit):
                    emit(getattr(self._engine, "current_index", 0))
            return
        self._engine = create_single_run_engine(self._active_config.to_scenario_config(), step_callback=step_callback)

    @staticmethod
    def _is_batch_analysis(config: SimulationScenarioConfig) -> bool:
        return bool(config.monte_carlo.enabled or config.analysis.enabled)

    @staticmethod
    def _run_batch_analysis(config: SimulationConfig) -> dict[str, Any]:
        return run_simulation_scenario(config.to_scenario_config(), source_path=config.source_path)
