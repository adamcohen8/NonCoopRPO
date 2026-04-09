from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ControllerBenchTarget:
    object_id: str
    slot: str


@dataclass(frozen=True)
class ControllerVariant:
    name: str
    pointer: dict[str, Any]
    description: str = ""


@dataclass(frozen=True)
class ControllerBenchMetric:
    name: str
    source_path: str = ""
    kind: str = ""
    object_id: str = ""
    reference_object_id: str = ""
    desired_quat_bn: tuple[float, float, float, float] | None = None
    keepout_radius_km: float | None = None


@dataclass(frozen=True)
class ControllerBenchPassCriterion:
    metric: str
    op: str
    value: Any


@dataclass(frozen=True)
class ControllerBenchObjective:
    kind: str
    name: str = ""
    object_id: str = ""
    reference_object_id: str = ""
    desired_quat_bn: tuple[float, float, float, float] | None = None
    keepout_radius_km: float | None = None
    max_final_attitude_error_deg: float | None = None
    max_rms_attitude_error_deg: float | None = None
    max_final_body_rate_norm_rad_s: float | None = None
    max_final_relative_distance_km: float | None = None
    max_rms_relative_distance_km: float | None = None
    max_final_relative_speed_km_s: float | None = None
    max_time_inside_keepout_s: float | None = None
    max_total_dv_m_s: float | None = None
    max_fuel_used_kg: float | None = None
    require_not_terminated_early: bool = False


@dataclass(frozen=True)
class ControllerBenchCase:
    name: str
    config_path: Path
    description: str = ""
    metrics: tuple[ControllerBenchMetric, ...] = ()
    pass_criteria: tuple[ControllerBenchPassCriterion, ...] = ()
    objectives: tuple[ControllerBenchObjective, ...] = ()


@dataclass(frozen=True)
class ControllerBenchConfig:
    suite_name: str
    description: str = ""
    output_dir: Path = Path("outputs/controller_bench")
    plot_mode: str = "save"
    controller_target: ControllerBenchTarget = field(
        default_factory=lambda: ControllerBenchTarget(object_id="target", slot="attitude_control")
    )
    variants: tuple[ControllerVariant, ...] = ()
    cases: tuple[ControllerBenchCase, ...] = ()
    metrics: tuple[ControllerBenchMetric, ...] = ()
    pass_criteria: tuple[ControllerBenchPassCriterion, ...] = ()
    objectives: tuple[ControllerBenchObjective, ...] = ()
    save_run_payloads: bool = True
    disable_plots: bool = True
    disable_animations: bool = True
    print_individual_run_summaries: bool = False
    parallel_enabled: bool = False
    parallel_workers: int = 0
