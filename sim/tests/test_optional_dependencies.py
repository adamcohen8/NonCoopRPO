from __future__ import annotations

from unittest.mock import patch

import pytest

import machine_learning
import machine_learning.gym_env as gym_env_mod
import run_gui


def _minimal_rl_scenario() -> dict:
    return {
        "scenario_name": "optional_dep_test",
        "rocket": {"enabled": False},
        "target": {
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
            },
        },
        "chaser": {
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "relative_to_target_ric": {"frame": "rect", "state": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            },
        },
        "simulator": {
            "duration_s": 2.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {"output_dir": "outputs/test_optional_deps", "mode": "save"},
        "metadata": {"seed": 1},
    }


def test_run_gui_reports_missing_pyside_with_install_hint() -> None:
    err = ModuleNotFoundError("No module named 'PySide6'")
    err.name = "PySide6"
    with patch.object(run_gui, "import_module", side_effect=err):
        with pytest.raises(SystemExit, match="requirements-gui.txt"):
            run_gui.main()


def test_gym_env_keeps_fallback_behavior_when_gymnasium_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gym_env_mod, "_GYMNASIUM_IMPORT_ERROR", ModuleNotFoundError("No module named 'gymnasium'"))
    cfg = gym_env_mod.GymEnvConfig(scenario=_minimal_rl_scenario())
    env = gym_env_mod.GymSimulationEnv(cfg)
    obs, info = env.reset(seed=2)
    assert obs.ndim == 1
    assert "metrics" in info


def test_machine_learning_lazy_exports_report_missing_ml_stack() -> None:
    real_import_module = machine_learning.importlib.import_module

    def _fake_import_module(name: str, package: str | None = None):
        if name == "machine_learning.ppo_lightning":
            err = ModuleNotFoundError("No module named 'torch'")
            err.name = "torch"
            raise err
        return real_import_module(name, package)

    with patch.object(machine_learning.importlib, "import_module", side_effect=_fake_import_module):
        with pytest.raises(ModuleNotFoundError, match="requirements-ml.txt"):
            machine_learning.__getattr__("PPOConfig")
