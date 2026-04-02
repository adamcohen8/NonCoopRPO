from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def coerce_scalar(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        try:
            if any(ch in s for ch in (".", "e", "E")):
                return float(s)
            return int(s)
        except ValueError:
            return value
    return value


def extract_metric(payload: Any, path: str) -> Any:
    cur = payload
    for token in str(path).split("."):
        if "[" in token and token.endswith("]"):
            key, idx_txt = token[:-1].split("[", 1)
            if key:
                if not isinstance(cur, dict):
                    raise KeyError(path)
                cur = cur[key]
            idx = int(idx_txt)
            if not isinstance(cur, list):
                raise KeyError(path)
            cur = cur[idx]
            continue
        if not isinstance(cur, dict):
            raise KeyError(path)
        cur = cur[token]
    return coerce_scalar(cur)


def evaluate_rule(actual: Any, rule: dict[str, Any]) -> tuple[bool, str]:
    checks: list[tuple[bool, str]] = []
    if "equals" in rule:
        expected = coerce_scalar(rule["equals"])
        checks.append((actual == expected, f"equals {expected!r}"))
    if "not_equals" in rule:
        expected = coerce_scalar(rule["not_equals"])
        checks.append((actual != expected, f"!= {expected!r}"))
    if "min" in rule:
        expected = float(rule["min"])
        checks.append((float(actual) >= expected, f">= {expected}"))
    if "max" in rule:
        expected = float(rule["max"])
        checks.append((float(actual) <= expected, f"<= {expected}"))
    if "truthy" in rule:
        expected = bool(rule["truthy"])
        checks.append((bool(actual) is expected, f"truthy is {expected}"))
    if "contains" in rule:
        target = rule["contains"]
        checks.append((target in actual, f"contains {target!r}"))
    if "one_of" in rule:
        options = [coerce_scalar(item) for item in list(rule.get("one_of", []) or [])]
        checks.append((actual in options, f"in {options!r}"))
    ok = all(item[0] for item in checks) if checks else True
    summary = ", ".join(item[1] for item in checks) if checks else "no-op"
    return ok, summary


def evaluate_checks(payload: dict[str, Any], checks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for metric_path, rule in sorted(checks.items()):
        try:
            actual = extract_metric(payload, metric_path)
            passed, expectation = evaluate_rule(actual, rule)
            evaluations.append(
                {
                    "metric": metric_path,
                    "passed": bool(passed),
                    "actual": actual,
                    "rule": dict(rule),
                    "expectation": expectation,
                    "kind": "absolute",
                }
            )
        except Exception as exc:
            evaluations.append(
                {
                    "metric": metric_path,
                    "passed": False,
                    "actual": None,
                    "rule": dict(rule),
                    "expectation": "metric available",
                    "error": str(exc),
                    "kind": "absolute",
                }
            )
    return evaluations


def load_metric_payload(path: str | Path) -> Any:
    payload_path = Path(path).expanduser().resolve()
    raw_text = payload_path.read_text(encoding="utf-8")
    if payload_path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(raw_text)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return yaml.safe_load(raw_text)


def _relative_delta(actual: float, baseline: float) -> float:
    denom = max(abs(float(baseline)), 1.0e-12)
    return abs(float(actual) - float(baseline)) / denom


def evaluate_baseline_checks(
    payload: dict[str, Any],
    baseline_payload: dict[str, Any],
    checks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for metric_path, rule in sorted(checks.items()):
        try:
            actual = extract_metric(payload, metric_path)
            baseline = extract_metric(baseline_payload, metric_path)
            passed_checks: list[tuple[bool, str]] = []
            abs_delta: float | None = None
            rel_delta: float | None = None
            if "equals_baseline" in rule:
                expected = bool(rule["equals_baseline"])
                comparison_ok = (actual == baseline) if expected else (actual != baseline)
                passed_checks.append((comparison_ok, f"equals baseline is {expected}"))
            if any(key in rule for key in ("abs_delta_max", "abs_delta_min", "rel_delta_max", "rel_delta_min")):
                actual_f = float(actual)
                baseline_f = float(baseline)
                abs_delta = abs(actual_f - baseline_f)
                rel_delta = _relative_delta(actual_f, baseline_f)
                if "abs_delta_max" in rule:
                    expected = float(rule["abs_delta_max"])
                    passed_checks.append((abs_delta <= expected, f"|delta| <= {expected}"))
                if "abs_delta_min" in rule:
                    expected = float(rule["abs_delta_min"])
                    passed_checks.append((abs_delta >= expected, f"|delta| >= {expected}"))
                if "rel_delta_max" in rule:
                    expected = float(rule["rel_delta_max"])
                    passed_checks.append((rel_delta <= expected, f"rel_delta <= {expected}"))
                if "rel_delta_min" in rule:
                    expected = float(rule["rel_delta_min"])
                    passed_checks.append((rel_delta >= expected, f"rel_delta >= {expected}"))
            passed = all(item[0] for item in passed_checks) if passed_checks else True
            expectation = ", ".join(item[1] for item in passed_checks) if passed_checks else "no-op"
            evaluations.append(
                {
                    "metric": metric_path,
                    "passed": bool(passed),
                    "actual": actual,
                    "baseline": baseline,
                    "abs_delta": abs_delta,
                    "rel_delta": rel_delta,
                    "rule": dict(rule),
                    "expectation": expectation,
                    "kind": "baseline",
                }
            )
        except Exception as exc:
            evaluations.append(
                {
                    "metric": metric_path,
                    "passed": False,
                    "actual": None,
                    "baseline": None,
                    "rule": dict(rule),
                    "expectation": "baseline metric available",
                    "error": str(exc),
                    "kind": "baseline",
                }
            )
    return evaluations
