from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
