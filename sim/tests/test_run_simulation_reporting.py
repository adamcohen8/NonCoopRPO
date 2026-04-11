from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile

from run_simulation import _print_config_validation_report, _print_monte_carlo_summary, _print_sensitivity_summary


def test_print_sensitivity_summary_does_not_fall_through_into_monte_carlo_output() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_sensitivity_summary(
            {
                "config_path": "configs/example.yaml",
                "scenario_name": "sensitivity_case",
                "scenario_description": "Parameter sweep",
                "analysis": {
                    "method": "one_at_a_time",
                    "run_count": 2,
                    "parameter_count": 1,
                },
                "baseline": {"source": "generated"},
                "parameter_rankings": [
                    {
                        "parameter_path": "simulator.dt_s",
                        "max_abs_delta_from_baseline": 0.25,
                    }
                ],
            }
        )
    output = buf.getvalue()
    assert "MASTER ANALYSIS COMPLETED" in output
    assert "Top Driver" in output
    assert "MASTER MONTE CARLO COMPLETED" not in output


def test_print_monte_carlo_summary_reports_aggregate_statistics() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_monte_carlo_summary(
            {
                "config_path": "configs/example.yaml",
                "scenario_name": "mc_case",
                "scenario_description": "Monte Carlo smoke",
                "runs": [
                    {"summary": {"duration_s": 10.0, "attitude_guardrail_stats": {"target": 2}}},
                    {"summary": {"duration_s": 12.0, "attitude_guardrail_stats": {"target": 0}}},
                ],
                "aggregate_stats": {
                    "duration_s_min": 10.0,
                    "duration_s_mean": 11.0,
                    "duration_s_max": 12.0,
                    "terminated_early_rate": 0.25,
                    "pass_rate": 0.75,
                },
                "commander_brief": {
                    "p_success": 0.75,
                    "p_keepout_violation": 0.10,
                },
            }
        )
    output = buf.getvalue()
    assert "MASTER MONTE CARLO COMPLETED" in output
    assert "Iterations" in output
    assert "P(success)" in output
    assert "Guardrails" in output


def test_validate_only_report_succeeds_for_valid_config() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "valid.yaml"
        cfg_path.write_text(
            """
scenario_name: validate_smoke
target:
  enabled: true
simulator:
  duration_s: 10.0
  dt_s: 1.0
  dynamics:
    orbit:
      orbit_substep_s: 0.5
    attitude:
      enabled: false
outputs:
  output_dir: outputs/validate_smoke
  mode: save
""".strip()
            + "\n",
            encoding="utf-8",
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = _print_config_validation_report(str(cfg_path))
    output = buf.getvalue()
    assert ok is True
    assert "CONFIG VALIDATION" in output
    assert "validate_smoke" in output
    assert "steps=10" in output
    assert "Plugins" in output
    assert "Result" in output


def test_validate_only_report_fails_for_bad_timing_before_run() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "bad.yaml"
        cfg_path.write_text(
            """
scenario_name: bad_timing
simulator:
  duration_s: 2.5
  dt_s: 1.0
""".strip()
            + "\n",
            encoding="utf-8",
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = _print_config_validation_report(str(cfg_path))
    output = buf.getvalue()
    assert ok is False
    assert "FAILED" in output
    assert "integer multiple" in output
