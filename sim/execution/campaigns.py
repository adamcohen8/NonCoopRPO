from __future__ import annotations

from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue as queue_mod
from typing import Any, Callable

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.config import scenario_config_from_dict, validate_scenario_plugins
from sim.single_run import _run_single_config

StepCallback = Callable[[int, int], None]
BatchCallback = Callable[[int, int], None]
BatchProgressCallback = Callable[[dict[str, Any]], None]

logger = logging.getLogger(__name__)


def can_run_monte_carlo_campaign(cfg: SimulationScenarioConfig) -> bool:
    """Return whether this campaign slice is owned by the execution package."""
    return bool(cfg.monte_carlo.enabled)


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


def sample_monte_carlo_variation(variation: Any, rng: np.random.Generator) -> Any:
    mode = variation.mode.lower()
    if mode == "choice":
        if not variation.options:
            raise ValueError(f"Variation '{variation.parameter_path}' with mode=choice requires options.")
        return variation.options[int(rng.integers(0, len(variation.options)))]
    if mode == "uniform":
        if variation.low is None or variation.high is None:
            raise ValueError(f"Variation '{variation.parameter_path}' with mode=uniform requires low/high.")
        return float(rng.uniform(variation.low, variation.high))
    if mode == "normal":
        if variation.mean is None or variation.std is None:
            raise ValueError(f"Variation '{variation.parameter_path}' with mode=normal requires mean/std.")
        return float(rng.normal(variation.mean, variation.std))
    raise ValueError(f"Unsupported variation mode '{variation.mode}'.")


def prepare_monte_carlo_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.monte_carlo.base_seed))
    varies_metadata_seed = any(str(v.parameter_path) == "metadata.seed" for v in cfg.monte_carlo.variations)
    prepared: list[dict[str, Any]] = []
    for i in range(int(cfg.monte_carlo.iterations)):
        cdict = deepcopy(root)
        sampled = {}
        for variation in cfg.monte_carlo.variations:
            sampled_value = sample_monte_carlo_variation(variation, rng)
            _deep_set(cdict, variation.parameter_path, sampled_value)
            sampled[variation.parameter_path] = sampled_value
        if not varies_metadata_seed:
            md = cdict.setdefault("metadata", {})
            if "seed" not in md:
                md["seed"] = int(cfg.monte_carlo.base_seed) + i
        mode = str(cdict.get("outputs", {}).get("mode", "interactive"))
        if mode == "interactive":
            cdict.setdefault("outputs", {})["mode"] = "save"
        cdict.setdefault("outputs", {})["output_dir"] = str(outdir / f"mc_run_{i:04d}")
        prepared.append(
            {
                "iteration": i,
                "sampled_parameters": sampled,
                "config_dict": cdict,
                "seed": int(cdict.get("metadata", {}).get("seed", int(cfg.monte_carlo.base_seed) + i)),
            }
        )
    return prepared


def run_serial_monte_carlo_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
    strict_plugins: bool,
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
) -> dict[str, Any]:
    from sim.master_simulator import _closest_approach_from_run_payload, _relative_range_series_from_run_payload

    prepared = prepare_monte_carlo_runs(cfg=cfg, root=root, outdir=outdir)
    completed: dict[int, dict[str, Any]] = {}
    total_iters = int(cfg.monte_carlo.iterations)
    completed_count = 0

    for item in prepared:
        iteration = int(item["iteration"])
        ci = scenario_config_from_dict(dict(item["config_dict"]))
        if strict_plugins:
            errs = validate_scenario_plugins(ci)
            if errs:
                msg = "Plugin validation failed in Monte Carlo iteration {i}:\n- ".format(i=iteration) + "\n- ".join(errs)
                raise ValueError(msg)
        run_payload = _run_single_config(ci, step_callback=step_callback)
        completed[iteration] = {
            "iteration": iteration,
            "summary": run_payload["summary"],
            "closest_approach_km": _closest_approach_from_run_payload(run_payload),
            "relative_range_series": _relative_range_series_from_run_payload(run_payload),
        }
        completed_count += 1
        if batch_callback is not None:
            try:
                batch_callback(completed_count, total_iters)
            except Exception as exc:
                logger.warning("Disabling Monte Carlo callback after runtime error: %s", exc)
                batch_callback = None

    return {
        "prepared": prepared,
        "completed": completed,
        "parallel_active": False,
        "parallel_fallback_reason": None,
    }


def run_monte_carlo_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
    strict_plugins: bool,
    mc_out_cfg: dict[str, Any],
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    batch_progress_callback: BatchProgressCallback | None = None,
) -> dict[str, Any]:
    from sim.master_simulator import (
        _restore_env_vars,
        _run_mc_iteration_from_dict,
        _set_parallel_worker_thread_limits,
    )

    total_iters = int(cfg.monte_carlo.iterations)
    parallel_enabled = bool(cfg.monte_carlo.parallel_enabled)
    max_workers_cfg = int(cfg.monte_carlo.parallel_workers or 0)
    default_workers = max(1, (os.cpu_count() or 1) - 1)
    parallel_workers = max_workers_cfg if max_workers_cfg > 0 else default_workers
    parallel_workers = max(1, min(parallel_workers, max(total_iters, 1)))
    parallel_active = bool(parallel_enabled and total_iters > 1)
    parallel_fallback_reason: str | None = None
    prepared = prepare_monte_carlo_runs(cfg=cfg, root=root, outdir=outdir)
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
                    "progress_emit_every": int(mc_out_cfg.get("parallel_progress_emit_every_steps", 20) or 20),
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
                                logger.warning("Disabling Monte Carlo callback after runtime error: %s", exc)
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
                                    logger.warning("Disabling Monte Carlo progress callback after runtime error: %s", exc)
                                    batch_progress_callback = None
        except (OSError, PermissionError, NotImplementedError, EOFError) as exc:
            parallel_active = False
            parallel_fallback_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("Parallel Monte Carlo unavailable, falling back to serial execution: %s", exc)
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
        serial_result = run_serial_monte_carlo_runs(
            cfg=cfg,
            root=root,
            outdir=outdir,
            strict_plugins=strict_plugins,
            step_callback=step_callback,
            batch_callback=batch_callback,
        )
        prepared = list(serial_result.get("prepared", []) or [])
        completed = dict(serial_result.get("completed", {}) or {})

    return {
        "prepared": prepared,
        "completed": completed,
        "parallel_active": bool(parallel_active),
        "parallel_workers": int(parallel_workers if parallel_active else 1),
        "parallel_fallback_reason": parallel_fallback_reason,
    }


def run_monte_carlo_campaign(
    *,
    config_path: str | Path,
    cfg: SimulationScenarioConfig,
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    batch_progress_callback: BatchProgressCallback | None = None,
) -> dict[str, Any]:
    """Run a non-parallel Monte Carlo campaign.

    The execution package now owns the dispatch boundary for serial Monte Carlo.
    The rich report builder still lives in ``sim.master_simulator`` for this
    migration slice so the public payload and artifacts remain unchanged.
    """
    if not can_run_monte_carlo_campaign(cfg):
        raise ValueError("run_monte_carlo_campaign only supports non-parallel Monte Carlo configs.")

    from sim.master_simulator import run_master_simulation

    return run_master_simulation(
        config_path=config_path,
        step_callback=step_callback,
        mc_callback=batch_callback,
        mc_progress_callback=batch_progress_callback,
    )
