from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.propagator import (
    OrbitPropagator,
    drag_plugin,
    j2_plugin,
    j3_plugin,
    j4_plugin,
    spherical_harmonics_plugin,
    srp_plugin,
)
from sim.dynamics.orbit.spherical_harmonics import load_hpop_ggm03_terms


_LINE_RE = re.compile(
    r"^\s*(\d{4})/(\d{2})/(\d{2})\s+(\d{1,2}):\s*(\d{1,2}):\s*(\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*$"
)


@dataclass
class HPOPSeries:
    t_s: np.ndarray
    x_eci_km_km_s: np.ndarray


@dataclass
class HPOPSatelliteProps:
    mass_kg: float | None = None
    drag_area_m2: float | None = None
    solar_area_m2: float | None = None
    cd: float | None = None
    cr: float | None = None


def _parse_hpop_satellite_states(path: Path) -> HPOPSeries:
    if not path.exists():
        raise FileNotFoundError(f"HPOP output file not found: {path}")

    t_dt: list[datetime] = []
    x_rows: list[np.ndarray] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                continue
            m = _LINE_RE.match(s)
            if m is None:
                continue
            yy, mm, dd, hh, minu, sec = (
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                int(m.group(4)),
                int(m.group(5)),
                float(m.group(6)),
            )
            # Build from minute boundary + timedelta so values like sec=60 are handled safely.
            ts0 = datetime(yy, mm, dd, hh, minu, 0, 0)
            ts = ts0 + timedelta(seconds=float(sec))
            t_dt.append(ts)

            x_m = float(m.group(7))
            y_m = float(m.group(8))
            z_m = float(m.group(9))
            vx_m_s = float(m.group(10))
            vy_m_s = float(m.group(11))
            vz_m_s = float(m.group(12))
            x_rows.append(np.array([x_m, y_m, z_m, vx_m_s, vy_m_s, vz_m_s], dtype=float))

    if not x_rows:
        raise ValueError(f"No parsable state rows found in {path}")
    if len(t_dt) != len(x_rows):
        raise RuntimeError("Parsed timestamps and state rows have mismatched lengths.")

    t0 = t_dt[0]
    t_s = np.array([(ti - t0).total_seconds() for ti in t_dt], dtype=float)
    x_si = np.vstack(x_rows)
    x_km = x_si[:, :3] / 1e3
    v_km_s = x_si[:, 3:] / 1e3
    x = np.hstack((x_km, v_km_s))
    return HPOPSeries(t_s=t_s, x_eci_km_km_s=x)


def _parse_hpop_initial_state_props(path: Path) -> HPOPSatelliteProps:
    if not path.exists():
        return HPOPSatelliteProps()
    props = HPOPSatelliteProps()
    for raw in path.open("r", encoding="utf-8", errors="ignore"):
        s = raw.strip().lower()
        if not s:
            continue
        m = re.search(r"satellite's effective area for drag:\s*([-+]?\d+(?:\.\d+)?)", s)
        if m is not None:
            props.drag_area_m2 = float(m.group(1))
            continue
        m = re.search(r"satellite's effective area for solar radiation:\s*([-+]?\d+(?:\.\d+)?)", s)
        if m is not None:
            props.solar_area_m2 = float(m.group(1))
            continue
        m = re.search(r"satellite's mass:\s*([-+]?\d+(?:\.\d+)?)", s)
        if m is not None:
            props.mass_kg = float(m.group(1))
            continue
        m = re.search(r"\bcd:\s*([-+]?\d+(?:\.\d+)?)", s)
        if m is not None:
            props.cd = float(m.group(1))
            continue
        m = re.search(r"\bcr:\s*([-+]?\d+(?:\.\d+)?)", s)
        if m is not None:
            props.cr = float(m.group(1))
            continue
    return props


def _default_hpop_root() -> Path:
    return REPO_ROOT / "validation" / "High Precision Orbit Propagator_4-2" / "High Precision Orbit Propagator_4.2.2"


def _build_plugins(
    model: str,
    hpop_root: Path,
    atmosphere_model: str = "exponential",
    density_override_kg_m3: float | None = None,
    sun_dir_eci: np.ndarray | None = None,
) -> tuple[list, dict]:
    model_l = model.strip().lower()
    env: dict = {}
    if model_l == "two_body":
        return [], env
    if model_l == "drag":
        env["atmosphere_model"] = str(atmosphere_model).lower()
        if density_override_kg_m3 is not None:
            env["density_kg_m3"] = float(density_override_kg_m3)
        return [drag_plugin], env
    if model_l == "srp":
        s = np.array([1.0, 0.0, 0.0], dtype=float) if sun_dir_eci is None else np.array(sun_dir_eci, dtype=float).reshape(3)
        n = float(np.linalg.norm(s))
        if n > 0.0:
            s = s / n
        env["sun_dir_eci"] = s
        return [srp_plugin], env
    if model_l == "j2":
        return [j2_plugin], env
    if model_l == "j3":
        return [j3_plugin], env
    if model_l == "j4":
        return [j4_plugin], env
    if model_l in ("j2j3", "j2_j3"):
        return [j2_plugin, j3_plugin], env
    if model_l in ("j2j3j4", "j2_j3_j4"):
        return [j2_plugin, j3_plugin, j4_plugin], env
    if model_l in ("sh8x8", "spherical_8x8"):
        ggm03_path = REPO_ROOT / "validation" / "data" / "GGM03C.txt"
        terms = load_hpop_ggm03_terms(
            coeff_path=ggm03_path,
            max_degree=8,
            max_order=8,
            normalized=True,
        )
        env = {
            "spherical_harmonics_terms": terms,
            "spherical_harmonics_source": str(ggm03_path),
        }
        return [spherical_harmonics_plugin], env
    raise ValueError(f"Unsupported model '{model}'. Use: two_body, drag, srp, j2, j3, j4, j2j3, j2j3j4, sh8x8.")


def _propagate_on_time_grid(x0: np.ndarray, t_grid_s: np.ndarray, plugins: list, env: dict, ctx: OrbitContext) -> np.ndarray:
    if t_grid_s.ndim != 1 or t_grid_s.size < 2:
        raise ValueError("t_grid_s must be 1D with at least 2 samples.")
    x = np.zeros((t_grid_s.size, 6), dtype=float)
    x[0, :] = x0
    prop = OrbitPropagator(integrator="rk4", plugins=list(plugins))
    for k in range(t_grid_s.size - 1):
        dt = float(t_grid_s[k + 1] - t_grid_s[k])
        if dt <= 0.0:
            raise ValueError("Non-increasing time grid in HPOP states.")
        x[k + 1, :] = prop.propagate(
            x_eci=x[k, :],
            dt_s=dt,
            t_s=float(t_grid_s[k]),
            command_accel_eci_km_s2=np.zeros(3),
            env=env,
            ctx=ctx,
        )
    return x


def _resample_states_linear(t_src_s: np.ndarray, x_src: np.ndarray, t_query_s: np.ndarray) -> np.ndarray:
    if t_src_s.ndim != 1 or t_src_s.size < 2:
        raise ValueError("t_src_s must be 1D with at least 2 samples.")
    if x_src.ndim != 2 or x_src.shape[0] != t_src_s.size or x_src.shape[1] != 6:
        raise ValueError("x_src must have shape (N,6) matching t_src_s.")
    if np.any(np.diff(t_src_s) <= 0.0):
        raise ValueError("t_src_s must be strictly increasing for interpolation.")
    xq = np.zeros((t_query_s.size, 6), dtype=float)
    for i in range(6):
        xq[:, i] = np.interp(t_query_s, t_src_s, x_src[:, i])
    return xq


def _normalize_plot_mode(plot_mode: str) -> str:
    mode = str(plot_mode).strip().lower()
    if mode not in {"interactive", "save", "both", "none"}:
        raise ValueError("plot_mode must be one of: interactive, save, both, none.")
    return mode


def _import_pyplot():
    import matplotlib.pyplot as plt

    return plt


def compare_state_histories(
    *,
    sim_t_s: np.ndarray,
    sim_x_eci_km_km_s: np.ndarray,
    ref_t_s: np.ndarray,
    ref_x_eci_km_km_s: np.ndarray,
    model: str,
    plot_mode: str = "interactive",
    output_dir: Path | None = None,
    ref_label: str = "HPOP",
) -> dict[str, str]:
    plot_mode_norm = _normalize_plot_mode(plot_mode)
    sim_t = np.array(sim_t_s, dtype=float).reshape(-1)
    ref_t = np.array(ref_t_s, dtype=float).reshape(-1)
    sim_x = np.array(sim_x_eci_km_km_s, dtype=float)
    ref_x = np.array(ref_x_eci_km_km_s, dtype=float)

    if sim_t.ndim != 1 or sim_t.size < 2:
        raise ValueError("sim_t_s must be 1D with at least 2 samples.")
    if ref_t.ndim != 1 or ref_t.size < 2:
        raise ValueError("ref_t_s must be 1D with at least 2 samples.")
    if sim_x.shape != (sim_t.size, 6):
        raise ValueError("sim_x_eci_km_km_s must have shape (N,6) matching sim_t_s.")
    if ref_x.shape != (ref_t.size, 6):
        raise ValueError("ref_x_eci_km_km_s must have shape (N,6) matching ref_t_s.")
    if np.any(np.diff(sim_t) <= 0.0):
        raise ValueError("sim_t_s must be strictly increasing.")
    if np.any(np.diff(ref_t) <= 0.0):
        raise ValueError("ref_t_s must be strictly increasing.")
    if float(ref_t[0]) < float(sim_t[0]) - 1.0e-9 or float(ref_t[-1]) > float(sim_t[-1]) + 1.0e-9:
        raise ValueError("Reference time grid must lie within the simulator time span.")

    if sim_t.size == ref_t.size and np.allclose(sim_t, ref_t, atol=1.0e-6, rtol=0.0):
        sim_x_grid = sim_x.copy()
    else:
        sim_x_grid = _resample_states_linear(sim_t, sim_x, ref_t)

    d = sim_x_grid - ref_x
    pos_err_m = d[:, :3] * 1e3
    vel_err_mm_s = d[:, 3:] * 1e6
    pos_err_norm_m = np.linalg.norm(pos_err_m, axis=1)
    vel_err_norm_mm_s = np.linalg.norm(vel_err_mm_s, axis=1)

    outdir = Path(output_dir) if output_dir is not None else REPO_ROOT / "outputs" / "validation_hpop"
    if plot_mode_norm in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    plot_path = outdir / f"hpop_compare_{model}.png"
    eci_plot_path = outdir / f"hpop_compare_{model}_eci_overlay.png"
    if plot_mode_norm != "none":
        plt = _import_pyplot()
        fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
        pos_lbl = ["dx (m)", "dy (m)", "dz (m)"]
        vel_lbl = ["dvx (mm/s)", "dvy (mm/s)", "dvz (mm/s)"]
        for i in range(3):
            axes[i, 0].plot(ref_t, pos_err_m[:, i], color="tab:blue")
            axes[i, 0].set_ylabel(pos_lbl[i])
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 1].plot(ref_t, vel_err_mm_s[:, i], color="tab:orange")
            axes[i, 1].set_ylabel(vel_lbl[i])
            axes[i, 1].grid(True, alpha=0.3)
        axes[3, 0].plot(ref_t, pos_err_norm_m, color="tab:green")
        axes[3, 0].set_ylabel("|dr| (m)")
        axes[3, 0].set_xlabel("Time (s)")
        axes[3, 0].grid(True, alpha=0.3)
        axes[3, 1].plot(ref_t, vel_err_norm_mm_s, color="tab:red")
        axes[3, 1].set_ylabel("|dv| (mm/s)")
        axes[3, 1].set_xlabel("Time (s)")
        axes[3, 1].grid(True, alpha=0.3)
        axes[0, 0].set_title(f"Position Difference: Simulator ({model}) - {ref_label}")
        axes[0, 1].set_title(f"Velocity Difference: Simulator ({model}) - {ref_label}")
        fig.tight_layout()

        if plot_mode_norm in ("save", "both"):
            fig.savefig(plot_path, dpi=160)
        if plot_mode_norm in ("interactive", "both"):
            plt.show()
        plt.close(fig)

        fig_eci = plt.figure(figsize=(9, 8))
        ax_eci = fig_eci.add_subplot(111, projection="3d")
        ax_eci.plot(ref_x[:, 0], ref_x[:, 1], ref_x[:, 2], "--", linewidth=1.2, label=ref_label)
        ax_eci.plot(sim_x_grid[:, 0], sim_x_grid[:, 1], sim_x_grid[:, 2], linewidth=1.4, label=f"Simulator ({model})")
        ax_eci.scatter([ref_x[0, 0]], [ref_x[0, 1]], [ref_x[0, 2]], s=25, c="tab:green", label="Start")
        ax_eci.set_title(f"3D ECI Orbit Overlay: Simulator vs {ref_label}")
        ax_eci.set_xlabel("X (km)")
        ax_eci.set_ylabel("Y (km)")
        ax_eci.set_zlabel("Z (km)")
        ax_eci.grid(True, alpha=0.3)
        ax_eci.legend(loc="best")
        r_stack = np.vstack((ref_x[:, :3], sim_x_grid[:, :3]))
        span = np.ptp(r_stack, axis=0)
        max_span = float(np.max(span))
        center = np.mean(r_stack, axis=0)
        half = 0.5 * max(max_span, 1e-6)
        ax_eci.set_xlim(center[0] - half, center[0] + half)
        ax_eci.set_ylim(center[1] - half, center[1] + half)
        ax_eci.set_zlim(center[2] - half, center[2] + half)
        try:
            ax_eci.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass
        fig_eci.tight_layout()

        if plot_mode_norm in ("save", "both"):
            fig_eci.savefig(eci_plot_path, dpi=160)
        if plot_mode_norm in ("interactive", "both"):
            plt.show()
        plt.close(fig_eci)

    return {
        "validation_samples": str(int(ref_t.size)),
        "used_duration_s": f"{float(ref_t[-1]):.3f}",
        "pos_err_rms_m": f"{float(np.sqrt(np.mean(pos_err_norm_m**2))):.6f}",
        "pos_err_max_m": f"{float(np.max(pos_err_norm_m)):.6f}",
        "vel_err_rms_mm_s": f"{float(np.sqrt(np.mean(vel_err_norm_mm_s**2))):.6f}",
        "vel_err_max_mm_s": f"{float(np.max(vel_err_norm_mm_s)):.6f}",
        "eci_overlay_plot_path": str(eci_plot_path) if plot_mode_norm in ("save", "both") else "",
        "plot_path": str(plot_path) if plot_mode_norm in ("save", "both") else "",
    }


def run_validation(
    hpop_root: Path,
    model: str,
    validation_dt_s: float = 1.0,
    validation_duration_s: float = 150.0 * 60.0,
    atmosphere_model: str = "exponential",
    density_override_kg_m3: float | None = None,
    sun_dir_eci: np.ndarray | None = None,
    plot_mode: str = "interactive",
    output_dir: Path | None = None,
) -> dict[str, str]:
    sat_states = hpop_root / "SatelliteStates.txt"
    hpop = _parse_hpop_satellite_states(sat_states)
    dt = float(validation_dt_s)
    if dt <= 0.0:
        raise ValueError("validation_dt_s must be > 0.")

    t_end_req = float(validation_duration_s)
    if t_end_req <= 0.0:
        raise ValueError("validation_duration_s must be > 0.")
    t_end = min(t_end_req, float(hpop.t_s[-1]))
    if t_end < dt:
        raise ValueError(f"HPOP output duration ({hpop.t_s[-1]:.3f}s) is shorter than one validation step ({dt:.3f}s).")
    t_grid = np.arange(0.0, t_end + 0.5 * dt, dt, dtype=float)
    # Prefer exact sample alignment when HPOP was produced at the same cadence.
    if hpop.t_s.size == t_grid.size and np.allclose(hpop.t_s, t_grid, atol=1e-6, rtol=0.0):
        hpop_x_grid = hpop.x_eci_km_km_s.copy()
    else:
        hpop_x_grid = _resample_states_linear(hpop.t_s, hpop.x_eci_km_km_s, t_grid)

    props = _parse_hpop_initial_state_props(hpop_root / "InitialState.txt")
    mass_kg = float(props.mass_kg) if props.mass_kg is not None else 1000.0
    drag_area_m2 = float(props.drag_area_m2) if props.drag_area_m2 is not None else 1.0
    solar_area_m2 = float(props.solar_area_m2) if props.solar_area_m2 is not None else drag_area_m2
    area_m2 = solar_area_m2 if str(model).lower() == "srp" else drag_area_m2
    cd = float(props.cd) if props.cd is not None else 2.2
    cr = float(props.cr) if props.cr is not None else 1.2
    ctx = OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=mass_kg, area_m2=area_m2, cd=cd, cr=cr)

    plugins, env = _build_plugins(
        model,
        hpop_root=hpop_root,
        atmosphere_model=atmosphere_model,
        density_override_kg_m3=density_override_kg_m3,
        sun_dir_eci=sun_dir_eci,
    )
    sim_x = _propagate_on_time_grid(
        x0=hpop_x_grid[0, :],
        t_grid_s=t_grid,
        plugins=plugins,
        env=env,
        ctx=ctx,
    )

    comparison = compare_state_histories(
        sim_t_s=t_grid,
        sim_x_eci_km_km_s=sim_x,
        ref_t_s=t_grid,
        ref_x_eci_km_km_s=hpop_x_grid,
        model=model,
        plot_mode=plot_mode,
        output_dir=output_dir,
        ref_label="HPOP",
    )

    return {
        "hpop_root": str(hpop_root),
        "hpop_samples": str(int(hpop.t_s.size)),
        "validation_dt_s": f"{dt:.3f}",
        "model": model,
        "mass_kg": f"{mass_kg:.6f}",
        "solar_area_m2": f"{solar_area_m2:.6f}",
        "drag_area_m2": f"{area_m2:.6f}",
        "cd": f"{cd:.6f}",
        "cr": f"{cr:.6f}",
        "atmosphere_model": str(atmosphere_model),
        "density_override_kg_m3": "" if density_override_kg_m3 is None else f"{float(density_override_kg_m3):.6e}",
        "requested_duration_s": f"{t_end_req:.3f}",
        **comparison,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Python orbit propagation against HPOP SatelliteStates.txt output."
    )
    parser.add_argument(
        "--hpop-root",
        type=str,
        default=str(_default_hpop_root()),
        help="Directory containing HPOP files (SatelliteStates.txt).",
    )
    parser.add_argument(
        "--model",
        choices=["two_body", "drag", "srp", "j2", "j3", "j4", "j2j3", "j2j3j4", "sh8x8"],
        default="sh8x8",
        help="Python comparison model.",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run one-by-one comparison suite: two_body, j2, j3, j4, sh8x8.",
    )
    parser.add_argument("--dt", type=float, default=1.0, help="Validation time step for comparison grid (s).")
    parser.add_argument("--duration-min", type=float, default=150.0, help="Validation duration (minutes).")
    parser.add_argument(
        "--atmosphere-model",
        choices=["exponential", "ussa1976", "nrlmsise00", "jb2008"],
        default="exponential",
        help="Atmosphere model used for drag comparison.",
    )
    parser.add_argument(
        "--density-override-kg-m3",
        type=float,
        default=np.nan,
        help="Optional fixed density override for drag; if omitted uses selected atmosphere model.",
    )
    parser.add_argument("--sun-dir-x", type=float, default=1.0, help="ECI sun direction x component for SRP mode.")
    parser.add_argument("--sun-dir-y", type=float, default=0.0, help="ECI sun direction y component for SRP mode.")
    parser.add_argument("--sun-dir-z", type=float, default=0.0, help="ECI sun direction z component for SRP mode.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both", "none"], default="interactive")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional artifact directory override for saved plots.",
    )
    args = parser.parse_args()

    hpop_root = Path(args.hpop_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else None
    if bool(args.suite):
        suite_models = ["two_body", "j2", "j3", "j4", "sh8x8"]
        all_res: list[dict[str, str]] = []
        print("Running HPOP comparison suite...")
        for m in suite_models:
            r = run_validation(
                hpop_root=hpop_root,
                model=m,
                validation_dt_s=float(args.dt),
                validation_duration_s=float(args.duration_min) * 60.0,
                atmosphere_model=str(args.atmosphere_model),
                density_override_kg_m3=None if not np.isfinite(float(args.density_override_kg_m3)) else float(args.density_override_kg_m3),
                sun_dir_eci=np.array([float(args.sun_dir_x), float(args.sun_dir_y), float(args.sun_dir_z)], dtype=float),
                plot_mode=str(args.plot_mode),
                output_dir=output_dir,
            )
            all_res.append(r)
            print(
                f"  {m:8s} | pos_rms_m={r['pos_err_rms_m']} | pos_max_m={r['pos_err_max_m']} | "
                f"vel_rms_mm_s={r['vel_err_rms_mm_s']} | vel_max_mm_s={r['vel_err_max_mm_s']}"
            )
    else:
        res = run_validation(
            hpop_root=hpop_root,
            model=str(args.model),
            validation_dt_s=float(args.dt),
            validation_duration_s=float(args.duration_min) * 60.0,
            atmosphere_model=str(args.atmosphere_model),
            density_override_kg_m3=None if not np.isfinite(float(args.density_override_kg_m3)) else float(args.density_override_kg_m3),
            sun_dir_eci=np.array([float(args.sun_dir_x), float(args.sun_dir_y), float(args.sun_dir_z)], dtype=float),
            plot_mode=str(args.plot_mode),
            output_dir=output_dir,
        )
        print("HPOP comparison results:")
        for k, v in res.items():
            if v:
                print(f"  {k}: {v}")
