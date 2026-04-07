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
    reference_orbit: dict[str, Any] = field(default_factory=dict)
    guidance: AlgorithmPointer | None = None
    base_guidance: AlgorithmPointer | None = None
    guidance_modifiers: list[AlgorithmPointer] = field(default_factory=list)
    orbit_control: AlgorithmPointer | None = None
    attitude_control: AlgorithmPointer | None = None
    mission_strategy: AlgorithmPointer | None = None
    mission_execution: AlgorithmPointer | None = None
    mission_objectives: list[AlgorithmPointer] = field(default_factory=list)
    bridge: BridgePointer | None = None
    knowledge: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulatorSection:
    scenario_type: str = "auto"
    duration_s: float = 3600.0
    dt_s: float = 1.0
    initial_jd_utc: float | None = None
    dynamics: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    plugin_validation: dict[str, Any] = field(default_factory=lambda: {"strict": True})
    termination: dict[str, Any] = field(default_factory=lambda: {"earth_impact_enabled": True, "earth_radius_km": 6378.137})


@dataclass(frozen=True)
class OutputsSection:
    output_dir: str = "outputs"
    mode: str = "interactive"
    stats: dict[str, Any] = field(default_factory=dict)
    plots: dict[str, Any] = field(default_factory=dict)
    animations: dict[str, Any] = field(default_factory=dict)
    monte_carlo: dict[str, Any] = field(default_factory=dict)


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
    parallel_enabled: bool = False
    parallel_workers: int = 0
    variations: list[MonteCarloVariation] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisExecutionSection:
    parallel_enabled: bool = False
    parallel_workers: int = 0


@dataclass(frozen=True)
class AnalysisBaselineSection:
    enabled: bool = False
    summary_json: str = ""


@dataclass(frozen=True)
class AnalysisMonteCarloSection:
    iterations: int = 1
    base_seed: int = 0
    variations: list[MonteCarloVariation] = field(default_factory=list)


@dataclass(frozen=True)
class SensitivityParameter:
    parameter_path: str
    values: list[Any] = field(default_factory=list)
    distribution: str = "uniform"
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None


@dataclass(frozen=True)
class SensitivitySection:
    method: str = "one_at_a_time"
    samples: int = 0
    seed: int = 0
    parameters: list[SensitivityParameter] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisSection:
    enabled: bool = False
    study_type: str = "monte_carlo"
    execution: AnalysisExecutionSection = field(default_factory=AnalysisExecutionSection)
    metrics: list[str] = field(default_factory=list)
    baseline: AnalysisBaselineSection = field(default_factory=AnalysisBaselineSection)
    monte_carlo: AnalysisMonteCarloSection = field(default_factory=AnalysisMonteCarloSection)
    sensitivity: SensitivitySection = field(default_factory=SensitivitySection)


@dataclass(frozen=True)
class SimulationScenarioConfig:
    scenario_name: str = "unnamed_scenario"
    scenario_description: str = ""
    rocket: AgentSection = field(default_factory=lambda: AgentSection(enabled=False, role="rocket"))
    chaser: AgentSection = field(default_factory=lambda: AgentSection(enabled=False, role="chaser"))
    target: AgentSection = field(default_factory=lambda: AgentSection(enabled=True, role="target"))
    simulator: SimulatorSection = field(default_factory=SimulatorSection)
    outputs: OutputsSection = field(default_factory=OutputsSection)
    monte_carlo: MonteCarloSection = field(default_factory=MonteCarloSection)
    analysis: AnalysisSection = field(default_factory=AnalysisSection)
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
    if d.get("file") not in (None, ""):
        raise ValueError("Algorithm pointers do not support 'file'; use importable 'module' paths instead.")
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
    guidance_modifiers = d.get("guidance_modifiers", []) or []
    if not isinstance(guidance_modifiers, list):
        raise ValueError(f"Section '{role}.guidance_modifiers' must be a list.")
    if role != "rocket" and d.get("guidance") is not None:
        raise ValueError(
            f"Section '{role}.guidance' is no longer supported. "
            "Use mission_objectives for mission logic and orbit_control/attitude_control for controllers."
        )
    base_guidance = d.get("base_guidance")
    legacy_guidance = d.get("guidance")
    if role == "rocket" and base_guidance is None and legacy_guidance is not None:
        base_guidance = legacy_guidance
    default_enabled_by_role = {
        "rocket": False,
        "chaser": False,
        "target": True,
    }
    default_enabled = bool(default_enabled_by_role.get(role, True))
    return AgentSection(
        enabled=bool(d.get("enabled", default_enabled)),
        role=str(d.get("role", role)),
        specs=dict(d.get("specs", {}) or {}),
        initial_state=dict(d.get("initial_state", {}) or {}),
        reference_orbit=dict(d.get("reference_orbit", {}) or {}),
        guidance=_parse_algorithm_pointer(legacy_guidance),
        base_guidance=_parse_algorithm_pointer(base_guidance),
        guidance_modifiers=[p for p in (_parse_algorithm_pointer(x) for x in guidance_modifiers) if p is not None],
        orbit_control=_parse_algorithm_pointer(d.get("orbit_control")),
        attitude_control=_parse_algorithm_pointer(d.get("attitude_control")),
        mission_strategy=_parse_algorithm_pointer(d.get("mission_strategy")),
        mission_execution=_parse_algorithm_pointer(d.get("mission_execution")),
        mission_objectives=[p for p in (_parse_algorithm_pointer(x) for x in objectives) if p is not None],
        bridge=_parse_bridge_pointer(d.get("bridge")),
        knowledge=dict(d.get("knowledge", {}) or {}),
    )


def _parse_simulator_section(value: Any) -> SimulatorSection:
    d = _as_dict(value, "simulator")
    plugin_validation = {"strict": True}
    plugin_validation.update(dict(d.get("plugin_validation", {}) or {}))
    termination = {"earth_impact_enabled": True, "earth_radius_km": 6378.137}
    termination.update(dict(d.get("termination", {}) or {}))
    out = SimulatorSection(
        scenario_type=str(d.get("scenario_type", "auto")),
        duration_s=float(d.get("duration_s", 3600.0)),
        dt_s=float(d.get("dt_s", 1.0)),
        initial_jd_utc=float(d["initial_jd_utc"]) if d.get("initial_jd_utc") is not None else None,
        dynamics=dict(d.get("dynamics", {}) or {}),
        environment=dict(d.get("environment", {}) or {}),
        plugin_validation=plugin_validation,
        termination=termination,
    )
    if out.dt_s <= 0.0:
        raise ValueError("simulator.dt_s must be positive.")
    if out.duration_s <= 0.0:
        raise ValueError("simulator.duration_s must be positive.")
    if not out.scenario_type.strip():
        raise ValueError("simulator.scenario_type must be non-empty.")
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
        parallel_enabled=bool(d.get("parallel_enabled", False)),
        parallel_workers=int(d.get("parallel_workers", 0)),
        variations=[_parse_mc_variation(v) for v in vars_raw],
    )
    if out.iterations <= 0:
        raise ValueError("monte_carlo.iterations must be positive.")
    if out.parallel_workers < 0:
        raise ValueError("monte_carlo.parallel_workers must be >= 0.")
    return out


def _parse_analysis_execution_section(value: Any, *, fallback: MonteCarloSection | None = None) -> AnalysisExecutionSection:
    d = _as_dict(value, "analysis.execution")
    default_parallel_enabled = bool(fallback.parallel_enabled) if fallback is not None else False
    default_parallel_workers = int(fallback.parallel_workers) if fallback is not None else 0
    out = AnalysisExecutionSection(
        parallel_enabled=bool(d.get("parallel_enabled", default_parallel_enabled)),
        parallel_workers=int(d.get("parallel_workers", default_parallel_workers)),
    )
    if out.parallel_workers < 0:
        raise ValueError("analysis.execution.parallel_workers must be >= 0.")
    return out


def _parse_analysis_baseline_section(value: Any) -> AnalysisBaselineSection:
    d = _as_dict(value, "analysis.baseline")
    return AnalysisBaselineSection(
        enabled=bool(d.get("enabled", False)),
        summary_json=str(d.get("summary_json", "") or ""),
    )


def _parse_analysis_monte_carlo_section(value: Any, *, fallback: MonteCarloSection | None = None) -> AnalysisMonteCarloSection:
    d = _as_dict(value, "analysis.monte_carlo")
    vars_raw = d.get("variations")
    if vars_raw is None:
        variations = list(fallback.variations) if fallback is not None else []
    else:
        if not isinstance(vars_raw, list):
            raise ValueError("analysis.monte_carlo.variations must be a list.")
        variations = [_parse_mc_variation(v) for v in vars_raw]
    default_iterations = int(fallback.iterations) if fallback is not None else 1
    default_base_seed = int(fallback.base_seed) if fallback is not None else 0
    out = AnalysisMonteCarloSection(
        iterations=int(d.get("iterations", default_iterations)),
        base_seed=int(d.get("base_seed", default_base_seed)),
        variations=variations,
    )
    if out.iterations <= 0:
        raise ValueError("analysis.monte_carlo.iterations must be positive.")
    return out


def _parse_sensitivity_parameter(value: Any) -> SensitivityParameter:
    d = _as_dict(value, "analysis.sensitivity.parameter")
    path = d.get("parameter_path", d.get("path"))
    if not isinstance(path, str) or not path:
        raise ValueError("analysis.sensitivity.parameters[*].parameter_path must be a non-empty string.")
    values = d.get("values", [])
    if not isinstance(values, list):
        raise ValueError("analysis.sensitivity.parameters[*].values must be a list.")
    distribution = str(d.get("distribution", "uniform")).strip().lower()
    if distribution not in {"uniform", "normal"}:
        raise ValueError("analysis.sensitivity.parameters[*].distribution must be one of: uniform, normal.")
    return SensitivityParameter(
        parameter_path=path,
        values=list(values),
        distribution=distribution,
        low=float(d["low"]) if d.get("low") is not None else None,
        high=float(d["high"]) if d.get("high") is not None else None,
        mean=float(d["mean"]) if d.get("mean") is not None else None,
        std=float(d["std"]) if d.get("std") is not None else None,
    )


def _parse_sensitivity_section(value: Any) -> SensitivitySection:
    d = _as_dict(value, "analysis.sensitivity")
    params_raw = d.get("parameters", []) or []
    if not isinstance(params_raw, list):
        raise ValueError("analysis.sensitivity.parameters must be a list.")
    out = SensitivitySection(
        method=str(d.get("method", "one_at_a_time")),
        samples=int(d.get("samples", 0)),
        seed=int(d.get("seed", 0)),
        parameters=[_parse_sensitivity_parameter(v) for v in params_raw],
    )
    if out.method not in {"one_at_a_time", "lhs"}:
        raise ValueError("analysis.sensitivity.method must be one of: one_at_a_time, lhs.")
    if out.samples < 0:
        raise ValueError("analysis.sensitivity.samples must be >= 0.")
    return out


def _parse_analysis_section(value: Any, *, legacy_mc: MonteCarloSection) -> AnalysisSection:
    d = _as_dict(value, "analysis")
    metrics = d.get("metrics", []) or []
    if not isinstance(metrics, list):
        raise ValueError("analysis.metrics must be a list.")
    out = AnalysisSection(
        enabled=bool(d.get("enabled", False)),
        study_type=str(d.get("study_type", "monte_carlo")).strip().lower(),
        execution=_parse_analysis_execution_section(d.get("execution"), fallback=legacy_mc),
        metrics=[str(x) for x in metrics],
        baseline=_parse_analysis_baseline_section(d.get("baseline")),
        monte_carlo=_parse_analysis_monte_carlo_section(d.get("monte_carlo"), fallback=legacy_mc),
        sensitivity=_parse_sensitivity_section(d.get("sensitivity")),
    )
    if out.study_type not in {"monte_carlo", "sensitivity"}:
        raise ValueError("analysis.study_type must be one of: monte_carlo, sensitivity.")
    return out


def _analysis_from_legacy_monte_carlo(mc: MonteCarloSection) -> AnalysisSection:
    return AnalysisSection(
        enabled=bool(mc.enabled),
        study_type="monte_carlo",
        execution=AnalysisExecutionSection(
            parallel_enabled=bool(mc.parallel_enabled),
            parallel_workers=int(mc.parallel_workers),
        ),
        monte_carlo=AnalysisMonteCarloSection(
            iterations=int(mc.iterations),
            base_seed=int(mc.base_seed),
            variations=list(mc.variations),
        ),
    )


def _normalize_analysis_and_monte_carlo(
    legacy_mc: MonteCarloSection,
    analysis: AnalysisSection,
) -> tuple[MonteCarloSection, AnalysisSection]:
    if analysis.enabled and analysis.study_type == "monte_carlo":
        normalized_mc = MonteCarloSection(
            enabled=True,
            iterations=int(analysis.monte_carlo.iterations),
            base_seed=int(analysis.monte_carlo.base_seed),
            parallel_enabled=bool(analysis.execution.parallel_enabled),
            parallel_workers=int(analysis.execution.parallel_workers),
            variations=list(analysis.monte_carlo.variations),
        )
        return normalized_mc, analysis
    if analysis.enabled and analysis.study_type == "sensitivity":
        normalized_mc = MonteCarloSection(
            enabled=False,
            iterations=max(int(legacy_mc.iterations), 1),
            base_seed=int(legacy_mc.base_seed),
            parallel_enabled=False,
            parallel_workers=0,
            variations=[],
        )
        return normalized_mc, analysis
    if legacy_mc.enabled:
        return legacy_mc, _analysis_from_legacy_monte_carlo(legacy_mc)
    return legacy_mc, analysis


def _parse_outputs_section(value: Any) -> OutputsSection:
    d = _as_dict(value, "outputs")
    out = OutputsSection(
        output_dir=str(d.get("output_dir", "outputs")),
        mode=str(d.get("mode", "interactive")),
        stats=dict(d.get("stats", {}) or {}),
        plots=dict(d.get("plots", {}) or {}),
        animations=dict(d.get("animations", {}) or {}),
        monte_carlo=dict(d.get("monte_carlo", {}) or {}),
    )
    if out.mode not in ("interactive", "save", "both"):
        raise ValueError("outputs.mode must be one of: interactive, save, both.")
    if not out.output_dir.strip():
        raise ValueError("outputs.output_dir must be non-empty.")
    return out


def scenario_config_from_dict(data: dict[str, Any]) -> SimulationScenarioConfig:
    root = _as_dict(data, "root")
    legacy_mc = _parse_monte_carlo_section(root.get("monte_carlo"))
    analysis = _parse_analysis_section(root.get("analysis"), legacy_mc=legacy_mc)
    normalized_mc, normalized_analysis = _normalize_analysis_and_monte_carlo(legacy_mc, analysis)
    cfg = SimulationScenarioConfig(
        scenario_name=str(root.get("scenario_name", "unnamed_scenario")),
        scenario_description=str(root.get("scenario_description", "") or ""),
        rocket=_parse_agent_section(root.get("rocket"), role="rocket"),
        chaser=_parse_agent_section(root.get("chaser"), role="chaser"),
        target=_parse_agent_section(root.get("target"), role="target"),
        simulator=_parse_simulator_section(root.get("simulator")),
        outputs=_parse_outputs_section(root.get("outputs")),
        monte_carlo=normalized_mc,
        analysis=normalized_analysis,
        metadata=dict(root.get("metadata", {}) or {}),
    )
    if bool(dict(cfg.target.reference_orbit or {}).get("enabled", False)) and (not bool(cfg.target.enabled)):
        raise ValueError("target.reference_orbit.enabled requires target.enabled to be true.")
    return cfg


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
