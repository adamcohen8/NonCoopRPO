from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import yaml

from sim import SimulationConfig, SimulationResult, SimulationSession, SimulationSnapshot
from sim.master_simulator import run_master_simulation


def _api_config(output_dir: Path, *, monte_carlo: bool = False) -> dict:
    return {
        "scenario_name": "api_smoke",
        "rocket": {"enabled": False},
        "target": {
            "enabled": True,
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
            },
        },
        "chaser": {"enabled": False},
        "simulator": {
            "duration_s": 2.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {
                "print_summary": False,
                "save_json": False,
                "save_full_log": False,
            },
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
            "monte_carlo": {
                "save_iteration_summaries": False,
                "save_histograms": False,
                "display_histograms": False,
                "save_ops_dashboard": False,
                "display_ops_dashboard": False,
            },
        },
        "monte_carlo": {
            "enabled": bool(monte_carlo),
            "iterations": 2 if monte_carlo else 1,
            "base_seed": 7,
            "parallel_enabled": False,
            "variations": [],
        },
        "metadata": {"seed": 123},
    }


class TestSimulationApi:
    def test_session_from_yaml_and_legacy_single_run_callback_still_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "api_smoke.yaml"
            cfg_dict = _api_config(Path(tmpdir))
            cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

            session = SimulationSession.from_yaml(cfg_path)
            result = session.run()
            callback_events: list[tuple[int, int]] = []

            legacy = run_master_simulation(cfg_path, step_callback=lambda step, total: callback_events.append((step, total)))

            assert result.summary["scenario_name"] == "api_smoke"
            assert legacy["run"]["samples"] == result.summary["samples"]
            assert callback_events[0] == (0, 2)
            assert callback_events[-1] == (2, 2)

    def test_session_reset_step_and_run_single_scenario(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimulationConfig.from_dict(_api_config(Path(tmpdir)))
            session = SimulationSession.from_config(cfg)

            snap0 = session.reset(seed=42)

            assert isinstance(snap0, SimulationSnapshot)
            assert snap0.step_index == 0
            assert snap0.time_s == 0.0
            assert "target" in snap0.truth
            assert session.done is False

            snap1 = session.step()
            snap2 = session.step()

            assert snap1.step_index == 1
            assert snap1.time_s == 1.0
            assert snap2.step_index == 2
            assert snap2.time_s == 2.0
            assert session.done is True

            result = session.run()

            assert isinstance(result, SimulationResult)
            assert result.is_monte_carlo is False
            assert result.num_steps == 3
            assert result.summary["samples"] == 3
            assert result.metrics["scenario_name"] == "api_smoke"
            assert np.isfinite(result.truth["target"]).all()

    def test_session_run_monte_carlo_scenario(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimulationConfig.from_dict(_api_config(Path(tmpdir), monte_carlo=True))
            session = SimulationSession.from_config(cfg)

            assert session.reset() is None

            result = session.run()

            assert result.is_monte_carlo is True
            assert result.payload["monte_carlo"]["enabled"] is True
            assert result.payload["monte_carlo"]["iterations"] == 2
            assert "pass_rate" in result.metrics
