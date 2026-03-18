from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from sim.config import load_simulation_yaml
from sim.master_simulator import run_master_simulation

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]


def _fmt_float(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def _print_field(label: str, value: str, label_width: int = 13) -> None:
    print(f"{label:<{label_width}} : {value}")


def _pointer_label(ptr) -> str:
    if ptr is None:
        return "none"
    module = str(getattr(ptr, "module", "") or "").strip()
    cls = str(getattr(ptr, "class_name", "") or "").strip()
    fn = str(getattr(ptr, "function", "") or "").strip()
    if module and cls:
        return f"{module}.{cls}"
    if module and fn:
        return f"{module}.{fn}"
    if module:
        return module
    if cls:
        return cls
    if fn:
        return fn
    return "custom"


def _print_run_header(config_path: str, cfg) -> None:
    objects = []
    for oid, sec in (("target", cfg.target), ("rocket", cfg.rocket), ("chaser", cfg.chaser)):
        if bool(getattr(sec, "enabled", False)):
            objects.append(oid)

    n_steps = int(max(math.floor(float(cfg.simulator.duration_s) / float(cfg.simulator.dt_s)), 0))
    dynamics = dict(cfg.simulator.dynamics or {})
    orbit = dict(dynamics.get("orbit", {}) or {})
    attitude = dict(dynamics.get("attitude", {}) or {})
    att_enabled = bool(attitude.get("enabled", True))
    perturbations: list[str] = []
    if bool(orbit.get("j2", False)):
        perturbations.append("J2")
    if bool(orbit.get("j3", False)):
        perturbations.append("J3")
    if bool(orbit.get("j4", False)):
        perturbations.append("J4")
    if bool(orbit.get("drag", False)):
        perturbations.append("Drag")
    if bool(orbit.get("srp", False)):
        perturbations.append("SRP")
    if bool(orbit.get("third_body_moon", False)):
        perturbations.append("3rd Body Moon")
    if bool(orbit.get("third_body_sun", False)):
        perturbations.append("3rd Body Sun")
    sh = dict(orbit.get("spherical_harmonics", {}) or {})
    if bool(sh.get("enabled", False)):
        deg = sh.get("degree")
        order = sh.get("order")
        if deg is not None and order is not None:
            perturbations.append(f"Spherical Harmonics {deg}x{order}")
        else:
            perturbations.append("Spherical Harmonics")
    orbital_txt = "2 Body" + (" + " + " + ".join(perturbations) if perturbations else "")
    attitude_txt = "Enabled" if att_enabled else "Disabled"
    print("")
    print("=" * 102)
    print("MASTER SIMULATION RUN")
    print("=" * 102)
    print(f"Config     : {Path(config_path).resolve()}")
    print(f"Scenario   : {cfg.scenario_name}")
    print(f"Mode       : {'Monte Carlo' if bool(cfg.monte_carlo.enabled) else 'Single Run'}")
    print(
        f"Timing     : duration={_fmt_float(float(cfg.simulator.duration_s), 1)} s, "
        f"dt={_fmt_float(float(cfg.simulator.dt_s), 3)} s, "
        f"steps={n_steps}"
    )
    if bool(cfg.monte_carlo.enabled):
        if bool(cfg.monte_carlo.parallel_enabled):
            req_workers = int(cfg.monte_carlo.parallel_workers or 0)
            auto_workers = int(max(1, (os.cpu_count() or 1) - 1))
            workers_txt = req_workers if req_workers > 0 else f"auto({auto_workers})"
            print(f"MC         : iterations={int(cfg.monte_carlo.iterations)}, parallel=on, workers={workers_txt}")
        else:
            print(f"MC         : iterations={int(cfg.monte_carlo.iterations)}, parallel=off")
    print(f"Dynamics   : Orbital - {orbital_txt}, Attitude - {attitude_txt}")
    print(f"Objects    : {', '.join(objects) if objects else 'none'}")
    print("=" * 102)


def _print_single_run_summary(out: dict) -> None:
    run = dict(out.get("run", {}) or {})
    thrust = dict(run.get("thrust_stats", {}) or {})
    guardrails = dict(run.get("attitude_guardrail_stats", {}) or {})
    print("")
    print("=" * 102)
    print("MASTER SIMULATION COMPLETED")
    print("=" * 102)
    print(f"Config     : {out.get('config_path', '')}")
    print(f"Scenario   : {out.get('scenario_name', run.get('scenario_name', 'unknown'))}")
    print(f"Objects    : {', '.join(run.get('objects', []))}")
    print(
        f"Timing     : samples={run.get('samples', 0)}, "
        f"dt={_fmt_float(float(run.get('dt_s', 0.0)), 3)} s, "
        f"duration={_fmt_float(float(run.get('duration_s', 0.0)), 1)} s"
    )
    if bool(run.get("terminated_early", False)):
        print(
            "Termination: EARLY "
            f"(reason={run.get('termination_reason')}, "
            f"t={run.get('termination_time_s')}, object={run.get('termination_object_id')})"
        )
    else:
        print("Termination: nominal")
    if "rocket_insertion_achieved" in run:
        if bool(run.get("rocket_insertion_achieved", False)):
            print(f"Insertion  : achieved at t={run.get('rocket_insertion_time_s')}")
        else:
            print("Insertion  : not achieved")

    if thrust:
        print("-" * 102)
        print("Thrust Stats")
        print(f"{'Object':<14}{'Burn Samples':>14}{'Max Accel (km/s^2)':>24}{'Total dV (m/s)':>18}")
        for oid in sorted(thrust.keys()):
            s = dict(thrust.get(oid, {}) or {})
            print(
                f"{oid:<14}"
                f"{int(s.get('burn_samples', 0)):>14d}"
                f"{float(s.get('max_accel_km_s2', 0.0)):>24.3e}"
                f"{float(s.get('total_dv_m_s', 0.0)):>18.3f}"
            )
    if guardrails:
        hits = int(sum(int(v) for v in guardrails.values()))
        print("-" * 72)
        print(f"Guardrails : attitude_events={hits}")
    print("=" * 72)


def _print_monte_carlo_summary(out: dict) -> None:
    runs = list(out.get("runs", []) or [])
    agg_stats = dict(out.get("aggregate_stats", {}) or {})
    brief = dict(out.get("commander_brief", {}) or {})
    guardrail_event_totals = [
        int(sum(int(v) for v in dict(dict(r.get("summary", {}) or {}).get("attitude_guardrail_stats", {})).values()))
        for r in runs
    ]
    print("")
    print("=" * 102)
    print("MASTER MONTE CARLO COMPLETED")
    print("=" * 102)
    _print_field("Config", str(out.get("config_path", "")))
    _print_field("Scenario", str(out.get("scenario_name", "unknown")))
    _print_field("Iterations", str(len(runs)))
    if agg_stats:
        d_min = float(agg_stats.get("duration_s_min", 0.0))
        d_mean = float(agg_stats.get("duration_s_mean", 0.0))
        d_max = float(agg_stats.get("duration_s_max", 0.0))
        t_rate = float(agg_stats.get("terminated_early_rate", 0.0))
        p_success = float(agg_stats.get("pass_rate", brief.get("p_success", 0.0)))
        _print_field("Duration", f"min={d_min:.1f}s  mean={d_mean:.1f}s  max={d_max:.1f}s")
        _print_field("Early Term", f"{100.0 * t_rate:.1f}%")
        _print_field("P(success)", f"{100.0 * p_success:.1f}%")
        ca_min = agg_stats.get("closest_approach_km_min")
        ca_mean = agg_stats.get("closest_approach_km_mean")
        ca_max = agg_stats.get("closest_approach_km_max")
        if all(v is not None for v in (ca_min, ca_mean, ca_max)):
            try:
                _print_field(
                    "Closest App",
                    f"min={float(ca_min):.3f} km  mean={float(ca_mean):.3f} km  max={float(ca_max):.3f} km",
                )
            except (TypeError, ValueError):
                pass
        p_keepout = brief.get("p_keepout_violation")
        if p_keepout is not None:
            try:
                p_keepout_f = float(p_keepout)
                if not math.isnan(p_keepout_f):
                    _print_field("Keepout Risk", f"{100.0 * p_keepout_f:.1f}%")
            except (TypeError, ValueError):
                pass
        p_cat = brief.get("p_catastrophic_outcome", agg_stats.get("p_catastrophic_outcome"))
        if p_cat is not None:
            try:
                p_cat_f = float(p_cat)
                if not math.isnan(p_cat_f):
                    _print_field("Catastrophic", f"{100.0 * p_cat_f:.1f}%")
            except (TypeError, ValueError):
                pass
        p_dv = brief.get("p_exceed_dv_budget", agg_stats.get("p_exceed_dv_budget"))
        p_time = brief.get("p_exceed_time_budget", agg_stats.get("p_exceed_time_budget"))
        try:
            if p_dv is not None and not math.isnan(float(p_dv)):
                _print_field("DV > Budget", f"{100.0 * float(p_dv):.1f}%")
        except (TypeError, ValueError):
            pass
        try:
            if p_time is not None and not math.isnan(float(p_time)):
                _print_field("Time > Budget", f"{100.0 * float(p_time):.1f}%")
        except (TypeError, ValueError):
            pass
        timeline = dict(brief.get("timeline_confidence_bands_s", {}) or {})
        fuel = dict(brief.get("fuel_confidence_bands_total_dv_m_s", {}) or {})
        if timeline:
            try:
                _print_field(
                    "Timeline",
                    f"P50={float(timeline.get('p50', float('nan'))):.1f}s  "
                    f"P90={float(timeline.get('p90', float('nan'))):.1f}s  "
                    f"P99={float(timeline.get('p99', float('nan'))):.1f}s",
                )
            except (TypeError, ValueError):
                pass
        if fuel:
            try:
                _print_field(
                    "Total dV",
                    f"P50={float(fuel.get('p50', float('nan'))):.2f}m/s  "
                    f"P90={float(fuel.get('p90', float('nan'))):.2f}m/s  "
                    f"P99={float(fuel.get('p99', float('nan'))):.2f}m/s",
                )
            except (TypeError, ValueError):
                pass
        by_obj = dict(agg_stats.get("by_object", {}) or {})
        if by_obj:
            print("-" * 72)
            print("Object Stats")
            print(f"{'Object':<14}{'Mean dV (m/s)':>16}{'Min dV':>12}{'Max dV':>12}{'Mean Burns':>14}")
            for oid in sorted(by_obj.keys()):
                s = dict(by_obj.get(oid, {}) or {})
                print(
                    f"{oid:<14}"
                    f"{float(s.get('total_dv_m_s_mean', 0.0)):>16.3f}"
                    f"{float(s.get('total_dv_m_s_min', 0.0)):>12.3f}"
                    f"{float(s.get('total_dv_m_s_max', 0.0)):>12.3f}"
                    f"{float(s.get('burn_samples_mean', 0.0)):>14.1f}"
                )
        top_fail = list(brief.get("top_failure_modes", []) or [])
        if top_fail:
            print("-" * 102)
            print("Top Failure Modes")
            for row in top_fail:
                try:
                    print(f"{str(row.get('reason', 'unknown')):<40}{int(row.get('count', 0)):>8d}{100.0*float(row.get('rate', 0.0)):>10.1f}%")
                except (TypeError, ValueError):
                    continue
    elif runs:
        durations = [float(dict(r.get("summary", {}) or {}).get("duration_s", 0.0)) for r in runs]
        _print_field("Duration", f"min={min(durations):.1f}s  max={max(durations):.1f}s")
    if guardrail_event_totals:
        _print_field("Guardrails", f"mean={sum(guardrail_event_totals)/len(guardrail_event_totals):.1f}  max={max(guardrail_event_totals)}")
    print("=" * 102)


def _physical_cpu_count() -> int | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True, stderr=subprocess.DEVNULL).strip()
        n = int(out)
        if n > 0:
            return n
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return None


def _available_memory_bytes() -> int | None:
    names = ["SC_AVPHYS_PAGES", "SC_PAGE_SIZE"]
    if all(hasattr(os, "sysconf") and n in os.sysconf_names for n in names):
        try:
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return pages * page_size
        except (OSError, ValueError):
            pass
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
            page_size = 4096
            for line in out.splitlines():
                if "page size of" in line:
                    parts = line.split("page size of", 1)[1].strip().split(" ", 1)
                    page_size = int(parts[0])
                    break
            free = 0
            inactive = 0
            speculative = 0
            for line in out.splitlines():
                txt = line.strip()
                if txt.startswith("Pages free:"):
                    free = int(txt.split(":")[1].strip().rstrip("."))
                elif txt.startswith("Pages inactive:"):
                    inactive = int(txt.split(":")[1].strip().rstrip("."))
                elif txt.startswith("Pages speculative:"):
                    speculative = int(txt.split(":")[1].strip().rstrip("."))
            total_pages = free + inactive + speculative
            if total_pages > 0:
                return total_pages * page_size
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None
    return None


def _maxrss_bytes() -> int | None:
    if resource is None:
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF)
    v = int(getattr(ru, "ru_maxrss", 0))
    if v <= 0:
        return None
    # Linux reports KB, macOS reports bytes.
    if sys.platform == "darwin":
        return v
    return v * 1024


def _cpu_time_seconds_including_children() -> float:
    if resource is None:
        return float(time.process_time())
    try:
        self_ru = resource.getrusage(resource.RUSAGE_SELF)
        child_ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        return float(self_ru.ru_utime + self_ru.ru_stime + child_ru.ru_utime + child_ru.ru_stime)
    except Exception:
        return float(time.process_time())


def _recommend_workers(
    *,
    logical_cores: int,
    physical_cores: int | None,
    available_mem_bytes: int | None,
    per_worker_mem_bytes: int | None,
) -> dict:
    cpu_base = max(1, (physical_cores if (physical_cores and physical_cores > 0) else logical_cores) - 1)
    mem_base = None
    if available_mem_bytes is not None and per_worker_mem_bytes is not None and per_worker_mem_bytes > 0:
        mem_base = max(1, int((0.7 * float(available_mem_bytes)) // float(per_worker_mem_bytes)))
    rec = cpu_base if mem_base is None else max(1, min(cpu_base, mem_base))
    return {
        "recommended_workers": int(rec),
        "cpu_limited_workers": int(cpu_base),
        "memory_limited_workers": int(mem_base) if mem_base is not None else None,
    }


def _print_serial_benchmark(config_path: str, benchmark_runs: int) -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for benchmark mode.") from exc

    cfg = load_simulation_yaml(config_path)
    root = cfg.to_dict()
    root.setdefault("monte_carlo", {})
    root["monte_carlo"]["enabled"] = True
    root["monte_carlo"]["iterations"] = int(max(benchmark_runs, 1))

    outputs = root.setdefault("outputs", {})
    outputs["mode"] = "save"
    stats = outputs.setdefault("stats", {})
    stats["enabled"] = True
    stats["print_summary"] = False
    stats["save_json"] = False
    stats["save_csv"] = False
    stats["save_full_log"] = False
    plots = outputs.setdefault("plots", {})
    plots["enabled"] = False
    plots["figure_ids"] = []
    animations = outputs.setdefault("animations", {})
    animations["enabled"] = False
    animations["types"] = []
    mc_out = outputs.setdefault("monte_carlo", {})
    mc_out["save_iteration_summaries"] = False
    mc_out["save_aggregate_summary"] = False
    mc_out["save_histograms"] = False
    mc_out["display_histograms"] = False
    mc_out["save_ops_dashboard"] = False
    mc_out["display_ops_dashboard"] = False
    mc_out["save_raw_runs"] = False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        yaml.safe_dump(root, tf, sort_keys=False)
        tmp_cfg_path = tf.name

    rss_before = _maxrss_bytes()
    cpu_before = _cpu_time_seconds_including_children()
    t0 = time.perf_counter()
    try:
        out = run_master_simulation(config_path=tmp_cfg_path)
    finally:
        try:
            os.unlink(tmp_cfg_path)
        except OSError:
            pass
    wall_s = max(time.perf_counter() - t0, 1e-9)
    cpu_s = max(_cpu_time_seconds_including_children() - cpu_before, 0.0)
    rss_after = _maxrss_bytes()

    agg = dict(out.get("aggregate_stats", {}) or {})
    runs_executed = int(out.get("monte_carlo", {}).get("iterations", benchmark_runs))
    sec_per_run = float(wall_s / max(runs_executed, 1))
    runs_per_hour = float(3600.0 / sec_per_run) if sec_per_run > 0 else float("inf")
    cpu_pct = float(100.0 * cpu_s / wall_s) if wall_s > 0 else 0.0
    cpu_cores = float(cpu_s / wall_s) if wall_s > 0 else 0.0
    peak_rss = int(rss_after) if rss_after is not None else None
    delta_rss = None
    if rss_before is not None and rss_after is not None:
        delta_rss = max(0, int(rss_after - rss_before))

    logical = int(os.cpu_count() or 1)
    physical = _physical_cpu_count()
    avail_mem = _available_memory_bytes()
    mem_per_worker = peak_rss if peak_rss is not None else None
    rec = _recommend_workers(
        logical_cores=logical,
        physical_cores=physical,
        available_mem_bytes=avail_mem,
        per_worker_mem_bytes=mem_per_worker,
    )

    print("")
    print("=" * 72)
    print("SERIAL MONTE CARLO BENCHMARK")
    print("=" * 72)
    print(f"Config            : {Path(config_path).resolve()}")
    print(f"Benchmark Runs    : {runs_executed}")
    print(f"Wall Time         : {wall_s:.2f} s")
    print(f"Seconds / Run     : {sec_per_run:.2f} s")
    print(f"Runs / Hour       : {runs_per_hour:.1f}")
    print(f"CPU Utilization   : {cpu_pct:.1f}% (~{cpu_cores:.2f} cores)")
    print(f"Logical Cores     : {logical}")
    print(f"Physical Cores    : {physical if physical is not None else 'unknown'}")
    if peak_rss is not None:
        print(f"Peak RSS          : {peak_rss / (1024**2):.1f} MiB")
    if delta_rss is not None:
        print(f"RSS Increase      : {delta_rss / (1024**2):.1f} MiB")
    if avail_mem is not None:
        print(f"Avail Memory      : {avail_mem / (1024**3):.2f} GiB")
    print("-" * 72)
    print(f"Recommended Workers (future parallel): {rec['recommended_workers']}")
    print(f"CPU-limited Workers               : {rec['cpu_limited_workers']}")
    if rec["memory_limited_workers"] is not None:
        print(f"Memory-limited Workers            : {rec['memory_limited_workers']}")
    print("-" * 72)
    print(
        "MC Summary        : "
        f"P(success)={100.0 * float(agg.get('pass_rate', 0.0)):.1f}%  "
        f"P(fail)={100.0 * float(agg.get('fail_rate', 0.0)):.1f}%"
    )
    print("=" * 72)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Master simulation runner: one YAML config, no other inputs required.")
    parser.add_argument(
        "--config",
        default=str(repo_root / "configs" / "simulation_template.yaml"),
        help="Path to simulation YAML config.",
    )
    parser.add_argument(
        "--benchmark-serial",
        action="store_true",
        help="Run a serial Monte Carlo benchmark (plots/saves disabled) and print max serial throughput and worker recommendation.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=10,
        help="Number of Monte Carlo iterations to run for serial benchmark mode.",
    )
    args = parser.parse_args()
    if args.benchmark_serial:
        _print_serial_benchmark(config_path=args.config, benchmark_runs=int(max(args.benchmark_runs, 1)))
        return
    cfg = load_simulation_yaml(args.config)
    _print_run_header(args.config, cfg)

    # Show progress only for the integration loop; config parsing and plotting are excluded.
    if cfg.monte_carlo.enabled:
        mc_total = int(max(cfg.monte_carlo.iterations, 0))
        if bool(cfg.monte_carlo.parallel_enabled):
            mc_bar = None
            worker_bars = []
            pid_to_slot: dict[int, int] = {}
            slot_state: dict[int, dict] = {}
            try:
                from tqdm.auto import tqdm  # type: ignore

                if mc_total > 0:
                    mc_bar = tqdm(total=mc_total, desc="Monte Carlo", unit="run", position=0)
                last_done = 0
                max_workers_cfg = int(cfg.monte_carlo.parallel_workers or 0)
                default_workers = int(max(1, (os.cpu_count() or 1) - 1))
                display_workers = int(max_workers_cfg if max_workers_cfg > 0 else default_workers)
                display_workers = max(1, min(display_workers, mc_total))
                for i in range(display_workers):
                    wb = tqdm(
                        total=1,
                        desc=f"Worker {i+1}",
                        unit="step",
                        position=i + 1,
                        leave=False,
                        dynamic_ncols=True,
                    )
                    worker_bars.append(wb)
                    slot_state[i] = {"iteration": None, "last_step": 0}

                def _on_mc_done(done: int, total: int) -> None:
                    nonlocal last_done, mc_bar
                    if mc_bar is None:
                        return
                    d = max(int(done), 0)
                    t = max(int(total), 0)
                    if t > 0 and int(mc_bar.total) != t:
                        mc_bar.total = t
                    if d > last_done:
                        mc_bar.update(d - last_done)
                    last_done = d

                def _on_worker_progress(evt: dict) -> None:
                    nonlocal pid_to_slot, slot_state, worker_bars, mc_total
                    if not worker_bars:
                        return
                    event = str(evt.get("event", ""))
                    pid = int(evt.get("pid", -1))
                    iteration = int(evt.get("iteration", -1))
                    if pid <= 0:
                        return
                    if pid not in pid_to_slot:
                        used = set(pid_to_slot.values())
                        free_slots = [i for i in range(len(worker_bars)) if i not in used]
                        pid_to_slot[pid] = free_slots[0] if free_slots else (len(pid_to_slot) % len(worker_bars))
                    slot = int(pid_to_slot[pid])
                    bar = worker_bars[slot]
                    state = slot_state.setdefault(slot, {"iteration": None, "last_step": 0})
                    if event == "done":
                        state["iteration"] = None
                        state["last_step"] = 0
                        bar.set_description(f"Worker {slot+1} (idle)")
                        return
                    if event != "step":
                        return
                    step = max(int(evt.get("step", 0)), 0)
                    total = max(int(evt.get("total", 0)), 0)
                    if state.get("iteration") != iteration:
                        state["iteration"] = iteration
                        state["last_step"] = 0
                        bar.reset(total=max(total, 1))
                        bar.set_description(f"Worker {slot+1} (run {iteration+1}/{max(mc_total,1)})")
                    if total > 0 and int(bar.total) != total:
                        bar.total = total
                    last_step = int(state.get("last_step", 0))
                    if step > last_step:
                        bar.update(step - last_step)
                        state["last_step"] = step

                out = run_master_simulation(
                    config_path=args.config,
                    step_callback=None,
                    mc_callback=_on_mc_done,
                    mc_progress_callback=_on_worker_progress,
                )
            finally:
                for wb in worker_bars:
                    wb.close()
                if mc_bar is not None:
                    if mc_bar.n < mc_total:
                        mc_bar.update(mc_total - mc_bar.n)
                    mc_bar.close()
        else:
            mc_bar = None
            sim_bar = None
            started_runs = 0
            last_step = 0
            run_done = True
            try:
                from tqdm.auto import tqdm  # type: ignore

                if mc_total > 0:
                    mc_bar = tqdm(total=mc_total, desc="Monte Carlo", unit="run")

                def _on_mc_step(step: int, total: int) -> None:
                    nonlocal sim_bar, started_runs, last_step, run_done, mc_bar
                    s = max(int(step), 0)
                    t = max(int(total), 0)
                    if s == 0:
                        started_runs += 1
                        run_done = False
                        last_step = 0
                        if sim_bar is not None:
                            sim_bar.close()
                        sim_bar = tqdm(
                            total=t,
                            desc=f"Simulation {started_runs}/{max(mc_total, 1)}",
                            unit="step",
                            leave=False,
                        )
                        if t == 0:
                            if mc_bar is not None:
                                mc_bar.update(1)
                            run_done = True
                            if sim_bar is not None:
                                sim_bar.close()
                                sim_bar = None
                        return

                    if sim_bar is None:
                        sim_bar = tqdm(total=t, desc=f"Simulation {started_runs}/{max(mc_total, 1)}", unit="step", leave=False)
                    if t > 0 and int(sim_bar.total) != t:
                        sim_bar.total = t
                    if s > last_step:
                        sim_bar.update(s - last_step)
                    last_step = s

                    if t > 0 and s >= t and not run_done:
                        if mc_bar is not None:
                            mc_bar.update(1)
                        run_done = True
                        if sim_bar is not None:
                            sim_bar.close()
                            sim_bar = None

                out = run_master_simulation(config_path=args.config, step_callback=_on_mc_step)
            finally:
                if sim_bar is not None:
                    sim_bar.close()
                if mc_bar is not None:
                    if mc_bar.n < mc_total:
                        mc_bar.update(mc_total - mc_bar.n)
                    mc_bar.close()
    else:
        total_steps = int(max(math.floor(float(cfg.simulator.duration_s) / float(cfg.simulator.dt_s)), 0))
        pbar = None
        last_step = 0
        if total_steps > 0:
            try:
                from tqdm.auto import tqdm  # type: ignore

                pbar = tqdm(total=total_steps, desc="Simulation", unit="step")
            except ImportError:
                pbar = None

        def _on_step(step: int, total: int) -> None:
            nonlocal last_step, pbar
            if pbar is None:
                return
            if int(total) > 0 and int(total) != int(pbar.total):
                pbar.total = int(total)
            s = max(int(step), 0)
            if s > last_step:
                pbar.update(s - last_step)
            last_step = s

        try:
            out = run_master_simulation(config_path=args.config, step_callback=_on_step)
        finally:
            if pbar is not None:
                pbar.close()
    if bool(out.get("monte_carlo", {}).get("enabled", False)):
        _print_monte_carlo_summary(out)
    else:
        _print_single_run_summary(out)


if __name__ == "__main__":
    main()
