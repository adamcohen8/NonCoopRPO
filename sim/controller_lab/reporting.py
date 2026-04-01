from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_controller_bench_reports(result: dict[str, Any], outdir: Path) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)

    summary_json = outdir / "controller_bench_summary.json"
    summary_md = outdir / "controller_bench_summary.md"
    comparison_csv = outdir / "controller_bench_comparison.csv"

    summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    runs = list(result.get("runs", []) or [])
    metric_names: list[str] = []
    for run in runs:
        for name in dict(run.get("metrics", {}) or {}).keys():
            if name not in metric_names:
                metric_names.append(str(name))

    with comparison_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "case", "passed", "output_dir", *metric_names],
        )
        writer.writeheader()
        for run in runs:
            row = {
                "variant": run.get("variant_name"),
                "case": run.get("case_name"),
                "passed": run.get("passed"),
                "output_dir": run.get("artifacts", {}).get("output_dir"),
            }
            for name in metric_names:
                row[name] = dict(run.get("metrics", {}) or {}).get(name)
            writer.writerow(row)

    lines = [
        f"# {result.get('suite_name', 'controller_bench')}",
        "",
        str(result.get("description", "") or "").strip(),
        "",
        "## Variants",
        "",
        "| Variant | Pass Rate | Passed | Total |",
        "| --- | ---: | ---: | ---: |",
    ]
    for summary in list(result.get("variant_summaries", []) or []):
        lines.append(
            f"| {summary.get('variant_name')} | "
            f"{100.0 * float(summary.get('pass_rate', 0.0)):.1f}% | "
            f"{int(summary.get('passed_runs', 0))} | "
            f"{int(summary.get('run_count', 0))} |"
        )
    objective_names: list[str] = []
    for summary in list(result.get("variant_summaries", []) or []):
        for name in dict(summary.get("objective_pass_rates", {}) or {}).keys():
            if name not in objective_names:
                objective_names.append(str(name))
    if objective_names:
        lines.extend(
            [
                "",
                "## Objective Pass Rates",
                "",
                "| Variant | Objective | Pass Rate |",
                "| --- | --- | ---: |",
            ]
        )
        for summary in list(result.get("variant_summaries", []) or []):
            objective_pass_rates = dict(summary.get("objective_pass_rates", {}) or {})
            for name in objective_names:
                if name not in objective_pass_rates:
                    continue
                lines.append(
                    f"| {summary.get('variant_name')} | {name} | "
                    f"{100.0 * float(objective_pass_rates.get(name, 0.0)):.1f}% |"
                )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary JSON: `{summary_json}`",
            f"- Comparison CSV: `{comparison_csv}`",
        ]
    )
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return {
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "comparison_csv": str(comparison_csv),
    }
