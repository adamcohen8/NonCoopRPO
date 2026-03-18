from __future__ import annotations

from pathlib import Path
from typing import Any

from sim.app.models import ConfigSummary
from sim.app.io import (
    CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    build_run_command,
    dump_yaml_text,
    ensure_sections,
    list_config_files,
    list_output_files,
    load_config_dict,
    parse_yaml_text,
    read_yaml_file,
    run_simulation_cli,
    save_config_dict,
    validate_config_dict,
)
from sim.config.scenario_yaml import SimulationScenarioConfig


def get_repo_root() -> Path:
    return REPO_ROOT


def get_default_config_path() -> Path:
    return DEFAULT_CONFIG_PATH


def get_config_dir() -> Path:
    return CONFIG_DIR


def list_available_configs() -> list[Path]:
    return list_config_files()


def load_config(path: str | Path) -> dict[str, Any]:
    return ensure_sections(load_config_dict(path))


def load_config_text(path: str | Path) -> str:
    return read_yaml_file(path)


def parse_config_text(yaml_text: str) -> dict[str, Any]:
    return ensure_sections(parse_yaml_text(yaml_text))


def validate_config(data: dict[str, Any]) -> SimulationScenarioConfig:
    return validate_config_dict(ensure_sections(data))


def dump_config_text(data: dict[str, Any]) -> str:
    return dump_yaml_text(ensure_sections(data))


def save_config(path: str | Path, data: dict[str, Any]) -> Path:
    cfg = validate_config(data)
    return save_config_dict(path, cfg.to_dict())


def summarize_config(cfg: SimulationScenarioConfig) -> ConfigSummary:
    objects = [
        object_id
        for object_id, section in (("rocket", cfg.rocket), ("chaser", cfg.chaser), ("target", cfg.target))
        if bool(section.enabled)
    ]
    return ConfigSummary(
        scenario_name=cfg.scenario_name,
        scenario_type=cfg.simulator.scenario_type,
        duration_s=float(cfg.simulator.duration_s),
        dt_s=float(cfg.simulator.dt_s),
        objects=objects,
        output_dir=cfg.outputs.output_dir,
        output_mode=cfg.outputs.mode,
        monte_carlo_enabled=bool(cfg.monte_carlo.enabled),
        mc_iterations=int(cfg.monte_carlo.iterations),
    )


def build_cli_run_command(config_path: str | Path) -> list[str]:
    return build_run_command(config_path)


def run_config_via_cli(config_path: str | Path) -> dict[str, Any]:
    return run_simulation_cli(config_path)


def get_output_files(output_dir: str | Path, limit: int = 200) -> list[Path]:
    return list_output_files(output_dir, limit=limit)
