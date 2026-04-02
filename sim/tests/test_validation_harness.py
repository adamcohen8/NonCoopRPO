import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml

from validation.automated_validation_harness import (
    _build_markdown_report,
    _default_harness_spec,
    _evaluate_rule,
    _extract_metric,
    filter_harness_spec,
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

    def test_load_harness_spec_reads_tags_and_baseline_checks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "spec.yaml"
            spec_path.write_text(
                "\n".join(
                    [
                        'suite_name: "demo"',
                        'output_dir: "outputs/demo"',
                        "benchmarks:",
                        '  - name: "sim_a"',
                        '    kind: "simulation"',
                        '    tags: ["fast", "rpo"]',
                        '    baseline_path: "baseline.json"',
                        "    baseline_checks:",
                        "      run.duration_s:",
                        "        abs_delta_max: 2.5",
                    ]
                ),
                encoding="utf-8",
            )
            spec = load_harness_spec(spec_path)
            self.assertEqual(spec.benchmarks[0].tags, ("fast", "rpo"))
            self.assertEqual(spec.benchmarks[0].baseline_path, "baseline.json")
            self.assertEqual(spec.benchmarks[0].baseline_checks["run.duration_s"]["abs_delta_max"], 2.5)

    def test_filter_harness_spec_supports_tag_kind_and_name(self):
        spec = _default_harness_spec()
        smoke_only = filter_harness_spec(spec, tags={"smoke"})
        self.assertTrue(all("smoke" in bench.tags for bench in smoke_only.benchmarks))
        hpop_only = filter_harness_spec(spec, kinds={"hpop"})
        self.assertTrue(all(bench.kind == "hpop" for bench in hpop_only.benchmarks))
        named = filter_harness_spec(spec, benchmark_names={"simulation_smoke"})
        self.assertEqual([bench.name for bench in named.benchmarks], ["simulation_smoke"])


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

    @patch("validation.automated_validation_harness._run_simulation_validation")
    def test_run_harness_supports_baseline_metric_checks(self, mock_run_simulation_validation):
        mock_run_simulation_validation.return_value = {
            "run": {"duration_s": 10.5},
            "derived": {"closest_approach_km": 3.0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.yaml"
            baseline_path.write_text(
                yaml.safe_dump({"run": {"duration_s": 10.0}, "derived": {"closest_approach_km": 2.9}}, sort_keys=False),
                encoding="utf-8",
            )
            spec_path = Path(tmpdir) / "spec.yaml"
            spec_path.write_text(
                "\n".join(
                    [
                        'suite_name: "baseline_suite"',
                        'output_dir: "outputs/baseline_suite"',
                        "benchmarks:",
                        '  - name: "baseline_case"',
                        '    kind: "simulation"',
                        '    config_path: "dummy.yaml"',
                        f'    baseline_path: "{baseline_path.name}"',
                        "    baseline_checks:",
                        "      run.duration_s:",
                        "        abs_delta_max: 1.0",
                        "      derived.closest_approach_km:",
                        "        rel_delta_max: 0.05",
                    ]
                ),
                encoding="utf-8",
            )
            spec = load_harness_spec(spec_path)
            report = run_harness(spec, base_dir=spec_path.parent)
            bench = report["benchmarks"][0]
            self.assertTrue(report["passed"])
            self.assertEqual(len(bench["baseline_evaluations"]), 2)
            self.assertEqual(bench["baseline_path"], str(baseline_path.resolve()))
            self.assertTrue(all(row["passed"] for row in bench["baseline_evaluations"]))

    def test_markdown_report_contains_baseline_table(self):
        md = _build_markdown_report(
            {
                "suite_name": "demo",
                "generated_utc": "2025-01-01T00:00:00Z",
                "passed": True,
                "benchmarks_passed": 1,
                "benchmarks_total": 1,
                "benchmarks": [
                    {
                        "name": "bench_a",
                        "kind": "simulation",
                        "passed": True,
                        "baseline_path": "/tmp/baseline.json",
                        "baseline_evaluations": [
                            {
                                "metric": "run.duration_s",
                                "actual": 10.5,
                                "baseline": 10.0,
                                "abs_delta": 0.5,
                                "expectation": "|delta| <= 1.0",
                                "passed": True,
                            }
                        ],
                    }
                ],
            }
        )
        self.assertIn("Baseline Metric", md)
        self.assertIn("/tmp/baseline.json", md)


if __name__ == "__main__":
    unittest.main()
