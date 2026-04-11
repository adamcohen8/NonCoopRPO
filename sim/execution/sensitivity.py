from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import logging
import multiprocessing as mp
import os
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable
import queue as queue_mod

import numpy as np

from sim.config import SimulationScenarioConfig, scenario_config_from_dict, validate_scenario_plugins
from sim.single_run import _run_single_config

StepCallback = Callable[[int, int], None]
BatchCallback = Callable[[int, int], None]
BatchProgressCallback = Callable[[dict[str, Any]], None]

logger = logging.getLogger(__name__)


def _deep_set(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = root
    for i, tok in enumerate(parts):
        last = i == len(parts) - 1
        if "[" in tok and tok.endswith("]"):
            key, idx_txt = tok[:-1].split("[", 1)
            idx = int(idx_txt)
            if key:
                cur = cur[key]
            if not isinstance(cur, list):
                raise TypeError(f"'{tok}' is not a list segment in path '{path}'.")
            if last:
                cur[idx] = value
                return
            cur = cur[idx]
            continue
        if last:
            cur[tok] = value
            return
        cur = cur[tok]


def _prepare_analysis_run_config(
    cdict: dict[str, Any],
    *,
    outdir: Path,
    run_prefix: str,
    iteration: int,
) -> dict[str, Any]:
    mode = str(cdict.get("outputs", {}).get("mode", "interactive"))
    if mode == "interactive":
        cdict.setdefault("outputs", {})["mode"] = "save"
    cdict.setdefault("outputs", {})["output_dir"] = str(outdir / f"{run_prefix}_{iteration:04d}")
    return cdict


def _prepare_oaat_sensitivity_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    iteration = 0
    for param in cfg.analysis.sensitivity.parameters:
        parameter_path = str(param.parameter_path)
        if not parameter_path:
            raise ValueError("analysis.sensitivity.parameters[*].parameter_path must be non-empty.")
        values = list(param.values or [])
        if not values:
            raise ValueError(f"analysis.sensitivity.parameters[{parameter_path!r}] must provide at least one value.")
        for value_index, value in enumerate(values):
            cdict = _prepare_analysis_run_config(
                deepcopy(root),
                outdir=outdir,
                run_prefix="sensitivity_run",
                iteration=iteration,
            )
            _deep_set(cdict, parameter_path, value)
            prepared.append(
                {
                    "iteration": iteration,
                    "parameter_path": parameter_path,
                    "parameter_value": value,
                    "value_index": value_index,
                    "sampled_parameters": {parameter_path: value},
                    "config_dict": cdict,
                }
            )
            iteration += 1
    return prepared


def _lhs_parameter_value(param: Any, unit_sample: float) -> float:
    unit = float(np.clip(float(unit_sample), 1.0e-12, 1.0 - 1.0e-12))
    distribution = str(getattr(param, "distribution", "uniform") or "uniform").strip().lower()
    if distribution == "uniform":
        low = getattr(param, "low", None)
        high = getattr(param, "high", None)
        if low is None or high is None:
            raise ValueError(f"LHS parameter '{param.parameter_path}' requires low/high for uniform distribution.")
        return float(low) + unit * (float(high) - float(low))
    if distribution == "normal":
        mean = getattr(param, "mean", None)
        std = getattr(param, "std", None)
        if mean is None or std is None or float(std) <= 0.0:
            raise ValueError(f"LHS parameter '{param.parameter_path}' requires mean/std with std > 0.")
        return float(NormalDist(mu=float(mean), sigma=float(std)).inv_cdf(unit))
    raise ValueError(f"Unsupported LHS distribution '{distribution}' for parameter '{param.parameter_path}'.")


def _prepare_lhs_sensitivity_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
) -> list[dict[str, Any]]:
    params = list(cfg.analysis.sensitivity.parameters or [])
    samples = int(cfg.analysis.sensitivity.samples or 0)
    if samples <= 0:
        raise ValueError("analysis.sensitivity.samples must be > 0 for method='lhs'.")
    if not params:
        raise ValueError("analysis.sensitivity.parameters must contain at least one parameter.")

    rng = np.random.default_rng(int(cfg.analysis.sensitivity.seed))
    sampled_columns: dict[str, np.ndarray] = {}
    for param in params:
        unit_samples = (rng.permutation(samples).astype(float) + rng.random(samples)) / float(samples)
        sampled_columns[str(param.parameter_path)] = np.array(
            [_lhs_parameter_value(param, unit) for unit in unit_samples],
            dtype=float,
        )

    prepared: list[dict[str, Any]] = []
    for iteration in range(samples):
        cdict = _prepare_analysis_run_config(
            deepcopy(root),
            outdir=outdir,
            run_prefix="sensitivity_lhs_run",
            iteration=iteration,
        )
        sampled_parameters: dict[str, Any] = {}
        for param in params:
            parameter_path = str(param.parameter_path)
            value = float(sampled_columns[parameter_path][iteration])
            _deep_set(cdict, parameter_path, value)
            sampled_parameters[parameter_path] = value
        prepared.append(
            {
                "iteration": iteration,
                "parameter_path": None,
                "parameter_value": None,
                "value_index": iteration,
                "sampled_parameters": sampled_parameters,
                "config_dict": cdict,
            }
        )
    return prepared


def prepare_sensitivity_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Any,
    sensitivity_method: str,
) -> list[dict[str, Any]]:
    method = str(sensitivity_method or "one_at_a_time").strip().lower()
    out_path = Path(outdir)
    if method == "lhs":
        return _prepare_lhs_sensitivity_runs(cfg=cfg, root=root, outdir=out_path)
    return _prepare_oaat_sensitivity_runs(cfg=cfg, root=root, outdir=out_path)


def run_sensitivity_runs(
    *,
    cfg: SimulationScenarioConfig,
    prepared: list[dict[str, Any]],
    strict_plugins: bool,
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    batch_progress_callback: BatchProgressCallback | None = None,
) -> dict[str, Any]:
    from sim.master_simulator import (
        _closest_approach_from_run_payload,
        _restore_env_vars,
        _run_mc_iteration_from_dict,
        _set_parallel_worker_thread_limits,
    )

    parallel_enabled = bool(cfg.analysis.execution.parallel_enabled)
    max_workers_cfg = int(cfg.analysis.execution.parallel_workers or 0)
    total_iters = int(len(prepared))
    default_workers = max(1, (os.cpu_count() or 1) - 1)
    parallel_workers = max_workers_cfg if max_workers_cfg > 0 else default_workers
    parallel_workers = max(1, min(parallel_workers, max(total_iters, 1)))
    parallel_active = bool(parallel_enabled and total_iters > 1)
    parallel_fallback_reason: str | None = None
    completed: dict[int, dict[str, Any]] = {}

    if parallel_active:
        manager = None
        progress_queue = None
        thread_env_prev = _set_parallel_worker_thread_limits(default_threads="1")
        try:
            manager = mp.Manager()
            progress_queue = manager.Queue()
            tasks = [
                {
                    "iteration": item["iteration"],
                    "config_dict": item["config_dict"],
                    "strict_plugins": strict_plugins,
                    "progress_queue": progress_queue,
                    "progress_emit_every": 20,
                }
                for item in prepared
            ]
            with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                fut_to_idx = {executor.submit(_run_mc_iteration_from_dict, task): int(task["iteration"]) for task in tasks}
                pending = set(fut_to_idx.keys())
                while pending:
                    done_now, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                    for fut in done_now:
                        idx = fut_to_idx[fut]
                        completed[idx] = fut.result()
                        if batch_callback is not None:
                            try:
                                batch_callback(len(completed), total_iters)
                            except Exception as exc:
                                logger.warning("Disabling analysis callback after runtime error: %s", exc)
                                batch_callback = None
                    if progress_queue is not None:
                        while True:
                            try:
                                evt = progress_queue.get_nowait()
                            except queue_mod.Empty:
                                break
                            except Exception:
                                break
                            if batch_progress_callback is not None:
                                try:
                                    batch_progress_callback(dict(evt or {}))
                                except Exception as exc:
                                    logger.warning("Disabling analysis progress callback after runtime error: %s", exc)
                                    batch_progress_callback = None
        except (OSError, PermissionError, NotImplementedError, EOFError) as exc:
            parallel_active = False
            parallel_fallback_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("Parallel sensitivity analysis unavailable, falling back to serial execution: %s", exc)
        finally:
            if progress_queue is not None:
                try:
                    while True:
                        evt = progress_queue.get_nowait()
                        if batch_progress_callback is not None:
                            batch_progress_callback(dict(evt or {}))
                except Exception:
                    pass
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception:
                    pass
            _restore_env_vars(thread_env_prev)

    if not parallel_active:
        completed_count = 0
        for prepared_run in prepared:
            iteration = int(prepared_run["iteration"])
            run_cfg = scenario_config_from_dict(dict(prepared_run["config_dict"]))
            if strict_plugins:
                errs = validate_scenario_plugins(run_cfg)
                if errs:
                    msg = "Plugin validation failed in sensitivity run {i}:\n- ".format(i=iteration) + "\n- ".join(errs)
                    raise ValueError(msg)
            run_payload = _run_single_config(run_cfg, step_callback=step_callback)
            completed[iteration] = {
                "iteration": iteration,
                "summary": dict(run_payload.get("summary", {}) or {}),
                "closest_approach_km": _closest_approach_from_run_payload(run_payload),
                "payload": run_payload,
            }
            completed_count += 1
            if batch_callback is not None:
                try:
                    batch_callback(completed_count, total_iters)
                except Exception as exc:
                    logger.warning("Disabling analysis callback after runtime error: %s", exc)
                    batch_callback = None

    return {
        "completed": completed,
        "parallel_active": bool(parallel_active),
        "parallel_workers": int(parallel_workers if parallel_active else 1),
        "parallel_fallback_reason": parallel_fallback_reason,
    }
