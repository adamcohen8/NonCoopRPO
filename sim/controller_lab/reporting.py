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
    leaderboard_csv = outdir / "controller_bench_leaderboard.csv"
    artifact_paths = {
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "comparison_csv": str(comparison_csv),
        "leaderboard_csv": str(leaderboard_csv),
    }
    current_artifacts = dict(result.get("artifacts", {}) or {})
    result["artifacts"] = {**current_artifacts, **artifact_paths}

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

    leaderboard_rows: list[dict[str, Any]] = []
    relative_rendezvous = list(dict(result.get("leaderboards", {}) or {}).get("relative_rendezvous", []) or [])
    with leaderboard_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["objective", "metric", "label", "direction", "rank", "variant", "value", "samples"],
        )
        writer.writeheader()
        for objective in relative_rendezvous:
            objective_name = str(objective.get("objective_name", "") or "")
            for ranking in list(objective.get("rankings", []) or []):
                for entry in list(ranking.get("entries", []) or []):
                    row = {
                        "objective": objective_name,
                        "metric": ranking.get("metric"),
                        "label": ranking.get("label"),
                        "direction": ranking.get("direction"),
                        "rank": entry.get("rank"),
                        "variant": entry.get("variant_name"),
                        "value": entry.get("value"),
                        "samples": entry.get("sample_count", entry.get("run_count")),
                    }
                    writer.writerow(row)
                    leaderboard_rows.append(row)

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
    if relative_rendezvous:
        lines.extend(
            [
                "",
                "## Rendezvous Leaderboards",
                "",
            ]
        )
        for objective in relative_rendezvous:
            objective_name = str(objective.get("objective_name", "") or "").strip()
            if not objective_name:
                continue
            lines.extend(
                [
                    f"### {objective_name}",
                    "",
                ]
            )
            for ranking in list(objective.get("rankings", []) or []):
                label = str(ranking.get("label", ranking.get("metric", "metric")) or "metric")
                lines.extend(
                    [
                        f"- {label}",
                        "",
                        "| Rank | Variant | Value | Samples |",
                        "| ---: | --- | ---: | ---: |",
                    ]
                )
                for entry in list(ranking.get("entries", []) or []):
                    value = entry.get("value")
                    if isinstance(value, float):
                        value_text = f"{value:.6g}"
                    else:
                        value_text = str(value)
                    sample_count = entry.get("sample_count", entry.get("run_count", ""))
                    lines.append(
                        f"| {entry.get('rank')} | {entry.get('variant_name')} | {value_text} | {sample_count} |"
                    )
                lines.append("")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary JSON: `{summary_json}`",
            f"- Summary MD: `{summary_md}`",
            f"- Comparison CSV: `{comparison_csv}`",
            f"- Leaderboard CSV: `{leaderboard_csv}`",
        ]
    )
    pass_rate_plot_png = current_artifacts.get("pass_rate_plot_png")
    if pass_rate_plot_png:
        lines.append(f"- Pass Rate Plot: `{pass_rate_plot_png}`")
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return artifact_paths
