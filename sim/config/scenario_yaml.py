from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlgorithmPointer:
    kind: str = "python"
    module: str | None = None
    class_name: str | None = None
    function: str | None = None
    file: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BridgePointer:
    enabled: bool = False
    mode: str = "sil"
    endpoint: str | None = None
    module: str | None = None
    class_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSection:
    enabled: bool = True
    role: str = "agent"
    specs: dict[str, Any] = field(default_factory=dict)
    initial_state: dict[str, Any] = field(default_factory=dict)
    guidance: AlgorithmPointer | None = None
    orbit_control: AlgorithmPointer | None = None
    attitude_control: AlgorithmPointer | None = None
    mission_objectives: list[AlgorithmPointer] = field(default_factory=list)
    bridge: BridgePointer | None = None
    knowledge: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulatorSection:
    duration_s: float = 3600.0
    dt_s: float = 1.0
    initial_jd_utc: float | None = None
    dynamics: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    termination: dict[str, Any] = field(default_factory=lambda: {"earth_impact_enabled": True, "earth_radius_km": 6378.137})


@dataclass(frozen=True)
class MonteCarloVariation:
    parameter_path: str
    mode: str = "choice"
    options: list[Any] = field(default_factory=list)
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None


@dataclass(frozen=True)
class MonteCarloSection:
    enabled: bool = False
    iterations: int = 1
    base_seed: int = 0
    variations: list[MonteCarloVariation] = field(default_factory=list)


@dataclass(frozen=True)
class SimulationScenarioConfig:
    scenario_name: str = "unnamed_scenario"
    rocket: AgentSection = field(default_factory=lambda: AgentSection(enabled=False, role="rocket"))
    chaser: AgentSection = field(default_factory=lambda: AgentSection(enabled=False, role="chaser"))
    target: AgentSection = field(default_factory=lambda: AgentSection(enabled=True, role="target"))
    simulator: SimulatorSection = field(default_factory=SimulatorSection)
    monte_carlo: MonteCarloSection = field(default_factory=MonteCarloSection)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_dict(value: Any, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section_name}' must be a mapping/object.")
    return dict(value)


def _parse_algorithm_pointer(value: Any) -> AlgorithmPointer | None:
    if value is None:
        return None
    if isinstance(value, str):
        return AlgorithmPointer(module=value)
    d = _as_dict(value, "algorithm_pointer")
    return AlgorithmPointer(
        kind=str(d.get("kind", "python")),
        module=d.get("module"),
        class_name=d.get("class_name"),
        function=d.get("function"),
        file=d.get("file"),
        params=dict(d.get("params", {}) or {}),
    )


def _parse_bridge_pointer(value: Any) -> BridgePointer | None:
    if value is None:
        return None
    d = _as_dict(value, "bridge")
    return BridgePointer(
        enabled=bool(d.get("enabled", False)),
        mode=str(d.get("mode", "sil")),
        endpoint=d.get("endpoint"),
        module=d.get("module"),
        class_name=d.get("class_name"),
        params=dict(d.get("params", {}) or {}),
    )


def _parse_agent_section(value: Any, role: str) -> AgentSection:
    d = _as_dict(value, role)
    objectives = d.get("mission_objectives", []) or []
    if not isinstance(objectives, list):
        raise ValueError(f"Section '{role}.mission_objectives' must be a list.")
    return AgentSection(
        enabled=bool(d.get("enabled", True)),
        role=str(d.get("role", role)),
        specs=dict(d.get("specs", {}) or {}),
        initial_state=dict(d.get("initial_state", {}) or {}),
        guidance=_parse_algorithm_pointer(d.get("guidance")),
        orbit_control=_parse_algorithm_pointer(d.get("orbit_control")),
        attitude_control=_parse_algorithm_pointer(d.get("attitude_control")),
        mission_objectives=[p for p in (_parse_algorithm_pointer(x) for x in objectives) if p is not None],
        bridge=_parse_bridge_pointer(d.get("bridge")),
        knowledge=dict(d.get("knowledge", {}) or {}),
    )


def _parse_simulator_section(value: Any) -> SimulatorSection:
    d = _as_dict(value, "simulator")
    out = SimulatorSection(
        duration_s=float(d.get("duration_s", 3600.0)),
        dt_s=float(d.get("dt_s", 1.0)),
        initial_jd_utc=float(d["initial_jd_utc"]) if d.get("initial_jd_utc") is not None else None,
        dynamics=dict(d.get("dynamics", {}) or {}),
        environment=dict(d.get("environment", {}) or {}),
        termination=dict(d.get("termination", {}) or {}),
    )
    if out.dt_s <= 0.0:
        raise ValueError("simulator.dt_s must be positive.")
    if out.duration_s <= 0.0:
        raise ValueError("simulator.duration_s must be positive.")
    return out


def _parse_mc_variation(value: Any) -> MonteCarloVariation:
    d = _as_dict(value, "monte_carlo.variation")
    path = d.get("parameter_path")
    if not isinstance(path, str) or not path:
        raise ValueError("monte_carlo.variations[*].parameter_path must be a non-empty string.")
    return MonteCarloVariation(
        parameter_path=path,
        mode=str(d.get("mode", "choice")),
        options=list(d.get("options", []) or []),
        low=float(d["low"]) if d.get("low") is not None else None,
        high=float(d["high"]) if d.get("high") is not None else None,
        mean=float(d["mean"]) if d.get("mean") is not None else None,
        std=float(d["std"]) if d.get("std") is not None else None,
    )


def _parse_monte_carlo_section(value: Any) -> MonteCarloSection:
    d = _as_dict(value, "monte_carlo")
    vars_raw = d.get("variations", []) or []
    if not isinstance(vars_raw, list):
        raise ValueError("monte_carlo.variations must be a list.")
    out = MonteCarloSection(
        enabled=bool(d.get("enabled", False)),
        iterations=int(d.get("iterations", 1)),
        base_seed=int(d.get("base_seed", 0)),
        variations=[_parse_mc_variation(v) for v in vars_raw],
    )
    if out.iterations <= 0:
        raise ValueError("monte_carlo.iterations must be positive.")
    return out


def scenario_config_from_dict(data: dict[str, Any]) -> SimulationScenarioConfig:
    root = _as_dict(data, "root")
    return SimulationScenarioConfig(
        scenario_name=str(root.get("scenario_name", "unnamed_scenario")),
        rocket=_parse_agent_section(root.get("rocket"), role="rocket"),
        chaser=_parse_agent_section(root.get("chaser"), role="chaser"),
        target=_parse_agent_section(root.get("target"), role="target"),
        simulator=_parse_simulator_section(root.get("simulator")),
        monte_carlo=_parse_monte_carlo_section(root.get("monte_carlo")),
        metadata=dict(root.get("metadata", {}) or {}),
    )


def load_simulation_yaml(path: str | Path) -> SimulationScenarioConfig:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load simulation YAML configs. Install with `pip install pyyaml`.") from exc
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Simulation YAML root must be a mapping/object.")
    return scenario_config_from_dict(raw)
