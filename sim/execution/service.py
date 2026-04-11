from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Callable

import yaml

from sim.config import SimulationScenarioConfig, load_simulation_yaml, validate_scenario_plugins
from . import campaigns
from sim.single_run import _SingleRunEngine, _coerce_noninteractive_for_automation, _run_single_config

StepCallback = Callable[[int, int], None]
BatchCallback = Callable[[int, int], None]
BatchProgressCallback = Callable[[dict[str, Any]], None]


class SimulationExecutionService:
    """Canonical entrypoint for executing simulation configs.

    This service is deliberately thin at first: single-run execution goes through
    the newer in-process engine, while batch analysis remains delegated to the
    legacy master simulator until those paths are migrated behind this facade.
    """

    def load_config(self, config_path: str | Path) -> SimulationScenarioConfig:
        return _coerce_noninteractive_for_automation(load_simulation_yaml(config_path))

    def validate_config(self, cfg: SimulationScenarioConfig) -> list[str]:
        strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
        if not strict_plugins:
            return []
        return list(validate_scenario_plugins(cfg))

    def study_type(self, cfg: SimulationScenarioConfig) -> str:
        if bool(cfg.analysis.enabled):
            return str(cfg.analysis.study_type or "monte_carlo").strip().lower()
        if bool(cfg.monte_carlo.enabled):
            return "monte_carlo"
        return "single_run"

    def is_batch_analysis(self, cfg: SimulationScenarioConfig) -> bool:
        return self.study_type(cfg) in {"monte_carlo", "sensitivity"}

    def create_engine(
        self,
        cfg: SimulationScenarioConfig,
        *,
        step_callback: StepCallback | None = None,
    ) -> _SingleRunEngine:
        return _SingleRunEngine(cfg, step_callback=step_callback)

    def run_single(
        self,
        cfg: SimulationScenarioConfig,
        *,
        step_callback: StepCallback | None = None,
    ) -> dict[str, Any]:
        return _run_single_config(cfg, step_callback=step_callback)

    def run_session_payload(
        self,
        cfg: SimulationScenarioConfig,
        *,
        source_path: str | Path | None = None,
        step_callback: StepCallback | None = None,
    ) -> dict[str, Any]:
        if self.is_batch_analysis(cfg):
            return self.run_batch_from_config(cfg, source_path=source_path, step_callback=step_callback)
        return self.run_single(cfg, step_callback=step_callback)

    def run_config_file(
        self,
        config_path: str | Path,
        *,
        step_callback: StepCallback | None = None,
        batch_callback: BatchCallback | None = None,
        batch_progress_callback: BatchProgressCallback | None = None,
    ) -> dict[str, Any]:
        path = Path(config_path).expanduser().resolve()
        cfg = self.load_config(path)
        errors = self.validate_config(cfg)
        if errors:
            msg = "Plugin validation failed:\n- " + "\n- ".join(errors)
            raise ValueError(msg)

        if self.is_batch_analysis(cfg):
            if self.study_type(cfg) == "monte_carlo" and campaigns.can_run_monte_carlo_campaign(cfg):
                return campaigns.run_monte_carlo_campaign(
                    config_path=path,
                    cfg=cfg,
                    step_callback=step_callback,
                    batch_callback=batch_callback,
                    batch_progress_callback=batch_progress_callback,
                )
            return self._run_legacy_master(
                path,
                step_callback=step_callback,
                batch_callback=batch_callback,
                batch_progress_callback=batch_progress_callback,
            )

        payload = self.run_single(cfg, step_callback=step_callback)
        return self.wrap_single_file_payload(payload=payload, cfg=cfg, config_path=path)

    def run_batch_from_config(
        self,
        cfg: SimulationScenarioConfig,
        *,
        source_path: str | Path | None = None,
        step_callback: StepCallback | None = None,
    ) -> dict[str, Any]:
        temp_dir = str(Path(source_path).expanduser().resolve().parent) if source_path is not None else str(Path.cwd())
        with tempfile.NamedTemporaryFile(
            suffix=".yaml",
            mode="w",
            delete=False,
            encoding="utf-8",
            dir=temp_dir,
        ) as tmp:
            yaml.safe_dump(cfg.to_dict(), tmp, sort_keys=False)
            tmp_path = Path(tmp.name)
        try:
            if self.study_type(cfg) == "monte_carlo" and campaigns.can_run_monte_carlo_campaign(cfg):
                payload = campaigns.run_monte_carlo_campaign(
                    config_path=tmp_path,
                    cfg=cfg,
                    step_callback=step_callback,
                )
            else:
                payload = self._run_legacy_master(tmp_path, step_callback=step_callback)
            if source_path is not None and isinstance(payload, dict):
                payload["config_path"] = str(Path(source_path).expanduser().resolve())
            return payload
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def wrap_single_file_payload(
        self,
        *,
        payload: dict[str, Any],
        cfg: SimulationScenarioConfig,
        config_path: str | Path,
    ) -> dict[str, Any]:
        return {
            "config_path": str(Path(config_path).expanduser().resolve()),
            "scenario_name": cfg.scenario_name,
            "scenario_description": cfg.scenario_description,
            "monte_carlo": {"enabled": False},
            "run": dict(payload.get("summary", {}) or {}),
        }

    def _run_legacy_master(
        self,
        config_path: str | Path,
        *,
        step_callback: StepCallback | None = None,
        batch_callback: BatchCallback | None = None,
        batch_progress_callback: BatchProgressCallback | None = None,
    ) -> dict[str, Any]:
        from sim.master_simulator import run_master_simulation

        return run_master_simulation(
            config_path=config_path,
            step_callback=step_callback,
            mc_callback=batch_callback,
            mc_progress_callback=batch_progress_callback,
        )


_DEFAULT_SERVICE = SimulationExecutionService()


def create_single_run_engine(
    cfg: SimulationScenarioConfig,
    *,
    step_callback: StepCallback | None = None,
) -> _SingleRunEngine:
    return _DEFAULT_SERVICE.create_engine(cfg, step_callback=step_callback)


def run_simulation_scenario(
    cfg: SimulationScenarioConfig,
    *,
    source_path: str | Path | None = None,
    step_callback: StepCallback | None = None,
) -> dict[str, Any]:
    return _DEFAULT_SERVICE.run_session_payload(cfg, source_path=source_path, step_callback=step_callback)


def run_simulation_config_file(
    config_path: str | Path,
    *,
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    batch_progress_callback: BatchProgressCallback | None = None,
) -> dict[str, Any]:
    return _DEFAULT_SERVICE.run_config_file(
        config_path,
        step_callback=step_callback,
        batch_callback=batch_callback,
        batch_progress_callback=batch_progress_callback,
    )
