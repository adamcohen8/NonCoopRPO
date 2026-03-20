import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from validation.automated_validation_harness import (
    _build_markdown_report,
    _evaluate_rule,
    _extract_metric,
    load_harness_spec,
    run_harness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestValidationHarnessHelpers(unittest.TestCase):
    def test_extract_metric_supports_nested_paths(self):
        payload = {"run": {"duration_s": 12.0}, "items": [{"value": 3}]}
        self.assertEqual(_extract_metric(payload, "run.duration_s"), 12.0)
        self.assertEqual(_extract_metric(payload, "items[0].value"), 3)

    def test_evaluate_rule_supports_min_max_and_equals(self):
        self.assertEqual(_evaluate_rule(5, {"min": 4, "max": 6})[0], True)
        self.assertEqual(_evaluate_rule(False, {"equals": False})[0], True)
        self.assertEqual(_evaluate_rule(3, {"min": 4})[0], False)

    def test_load_harness_spec_merges_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "spec.yaml"
            spec_path.write_text(
                "\n".join(
                    [
                        'suite_name: "demo"',
                        'output_dir: "outputs/demo"',
                        "tolerance_tables:",
                        "  leo:",
                        "    pos_err_max_m:",
                        "      max: 42.0",
                        "benchmarks: []",
                    ]
                ),
                encoding="utf-8",
            )
            spec = load_harness_spec(spec_path)
            self.assertEqual(spec.tolerance_tables["leo"]["pos_err_max_m"]["max"], 42.0)


class TestValidationHarnessExecution(unittest.TestCase):
    def test_run_harness_plugin_validation_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "spec.yaml"
            spec_path.write_text(
                "\n".join(
                    [
                        'suite_name: "plugin_only"',
                        'output_dir: "outputs/plugin_only"',
                        "benchmarks:",
                        '  - name: "plugin_validation_smoke"',
                        '    kind: "plugin_validation"',
                        f'    config_path: "{str(REPO_ROOT / "configs" / "automation_smoke.yaml")}"',
                        "    checks:",
                        "      valid:",
                        "        equals: true",
                        "      error_count:",
                        "        equals: 0",
                    ]
                ),
                encoding="utf-8",
            )
            spec = load_harness_spec(spec_path)
            report = run_harness(spec, base_dir=spec_path.parent)
            self.assertTrue(report["passed"])
            self.assertEqual(report["benchmarks_total"], 1)

    @patch("validation.automated_validation_harness._invoke_hpop_validation")
    def test_run_harness_hpop_benchmark(self, mock_invoke_hpop_validation):
        mock_invoke_hpop_validation.return_value = {
            "pos_err_rms_m": "1.0",
            "pos_err_max_m": "2.0",
            "vel_err_rms_mm_s": "3.0",
            "vel_err_max_mm_s": "4.0",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "spec.yaml"
            spec_path.write_text(
                "\n".join(
                    [
                        'suite_name: "hpop_only"',
                        'output_dir: "outputs/hpop_only"',
                        "benchmarks:",
                        '  - name: "hpop_case"',
                        '    kind: "hpop"',
                        '    envelope: "leo"',
                        "    params:",
                        '      model: "two_body"',
                        '      plot_mode: "save"',
                    ]
                ),
                encoding="utf-8",
            )
            spec = load_harness_spec(spec_path)
            report = run_harness(spec, base_dir=spec_path.parent)
            self.assertTrue(report["passed"])
            self.assertEqual(report["benchmarks"][0]["evaluations"][0]["passed"], True)

    def test_markdown_report_contains_benchmark_name(self):
        md = _build_markdown_report(
            {
                "suite_name": "demo",
                "generated_utc": "2025-01-01T00:00:00Z",
                "passed": True,
                "benchmarks_passed": 1,
                "benchmarks_total": 1,
                "benchmarks": [{"name": "bench_a", "kind": "plugin_validation", "passed": True, "evaluations": []}],
            }
        )
        self.assertIn("bench_a", md)


if __name__ == "__main__":
    unittest.main()
