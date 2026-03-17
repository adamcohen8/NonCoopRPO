from __future__ import annotations

import unittest

from sim.master_simulator import _assess_mc_run, _build_baseline_comparison, _build_parameter_sensitivity_rankings


class TestMonteCarloReporting(unittest.TestCase):
    def test_assess_mc_run_applies_gates(self):
        entry = {
            "summary": {
                "terminated_early": False,
                "termination_reason": None,
                "rocket_insertion_achieved": True,
                "duration_s": 1200.0,
                "attitude_guardrail_stats": {"sanitize": 2},
                "thrust_stats": {"chaser": {"total_dv_m_s": 90.0}},
            },
            "closest_approach_km": 0.12,
        }
        gates = {
            "min_closest_approach_km": 0.2,
            "max_duration_s": 2000.0,
            "max_total_dv_m_s": 80.0,
            "max_guardrail_events": 1,
        }
        out = _assess_mc_run(
            run_entry=entry,
            gates=gates,
            success_termination_reasons={"rocket_orbit_insertion"},
            require_rocket_insertion=False,
        )
        self.assertFalse(out["pass"])
        self.assertIn("gate:min_closest_approach_km", out["fail_reasons"])
        self.assertIn("gate:max_total_dv_m_s", out["fail_reasons"])
        self.assertIn("gate:max_guardrail_events", out["fail_reasons"])

    def test_sensitivity_rankings_returns_numeric_paths(self):
        run_details = [
            {"sampled_parameters": {"simulator.dt_s": 0.5, "mode": "a"}, "pass": True, "closest_approach_km": 1.0, "total_dv_m_s_total": 10.0},
            {"sampled_parameters": {"simulator.dt_s": 1.0, "mode": "b"}, "pass": False, "closest_approach_km": 0.3, "total_dv_m_s_total": 30.0},
            {"sampled_parameters": {"simulator.dt_s": 1.5, "mode": "a"}, "pass": False, "closest_approach_km": 0.2, "total_dv_m_s_total": 35.0},
            {"sampled_parameters": {"simulator.dt_s": 2.0, "mode": "b"}, "pass": False, "closest_approach_km": 0.1, "total_dv_m_s_total": 40.0},
        ]
        out = _build_parameter_sensitivity_rankings(run_details)
        self.assertTrue(any(str(r.get("parameter_path")) == "simulator.dt_s" for r in out))
        self.assertFalse(any(str(r.get("parameter_path")) == "mode" for r in out))

    def test_baseline_comparison_has_deltas(self):
        current = {
            "aggregate_stats": {"closest_approach_km_min": 0.3},
            "commander_brief": {
                "p_success": 0.8,
                "p_fail": 0.2,
                "timeline_confidence_bands_s": {"p95": 2000.0},
                "fuel_confidence_bands_total_dv_m_s": {"p95": 120.0},
            },
        }
        baseline = {
            "aggregate_stats": {"closest_approach_km_min": 0.4},
            "commander_brief": {
                "p_success": 0.7,
                "p_fail": 0.3,
                "timeline_confidence_bands_s": {"p95": 2200.0},
                "fuel_confidence_bands_total_dv_m_s": {"p95": 130.0},
            },
        }
        out = _build_baseline_comparison(current, baseline)
        delta = dict(out.get("delta_current_minus_baseline", {}))
        self.assertAlmostEqual(float(delta["p_success"]), 0.1, places=9)
        self.assertAlmostEqual(float(delta["duration_s_p95"]), -200.0, places=9)


if __name__ == "__main__":
    unittest.main()
