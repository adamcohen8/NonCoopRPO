from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConfigSummary:
    scenario_name: str
    scenario_type: str
    duration_s: float
    dt_s: float
    objects: list[str] = field(default_factory=list)
    output_dir: str = "outputs"
    output_mode: str = "interactive"
    analysis_enabled: bool = False
    analysis_study_type: str = "single_run"
    monte_carlo_enabled: bool = False
    mc_iterations: int = 1


@dataclass(frozen=True)
class RunRequest:
    config_path: Path
    mode: str = "cli"


@dataclass(frozen=True)
class RunResult:
    command: list[str] = field(default_factory=list)
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float = 0.0
    output_dir: str | None = None
    scenario_name: str | None = None


@dataclass(frozen=True)
class AnalysisUiProfile:
    count_label: str
    seed_label: str
    inputs_title: str
    editor_title: str
    help_text: str
    mode_label: str


@dataclass(frozen=True)
class GuiCapabilities:
    output_modes: list[str] = field(default_factory=list)
    orbit_integrators: list[str] = field(default_factory=list)
    analysis_study_types: list[tuple[str, str]] = field(default_factory=list)
    sensitivity_methods: list[tuple[str, str]] = field(default_factory=list)
    monte_carlo_modes: list[str] = field(default_factory=list)
    monte_carlo_lhs_modes: list[str] = field(default_factory=list)
    chaser_init_modes: list[str] = field(default_factory=list)
    satellite_presets: list[str] = field(default_factory=list)
    rocket_preset_stacks: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)
    animation_types: list[str] = field(default_factory=list)
    base_guidance_options: dict[str, list[tuple[str, dict[str, Any] | None]]] = field(default_factory=dict)
    guidance_modifier_options: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)
    orbit_control_options: dict[str, list[tuple[str, dict[str, Any] | None]]] = field(default_factory=dict)
    attitude_control_options: dict[str, list[tuple[str, dict[str, Any] | None]]] = field(default_factory=dict)
    mission_strategy_options: dict[str, list[tuple[str, dict[str, Any] | None]]] = field(default_factory=dict)
    mission_execution_options: dict[str, list[tuple[str, dict[str, Any] | None]]] = field(default_factory=dict)
    monte_carlo_parameter_categories: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    parameter_form_schemas: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    analysis_ui_profiles: dict[str, AnalysisUiProfile] = field(default_factory=dict)
