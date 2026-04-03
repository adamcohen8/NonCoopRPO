from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from sim.presets.rockets import RocketStackPreset
from sim.config import SimulationScenarioConfig
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.utils.figure_size import cap_figsize
from sim.utils.frames import ric_dcm_ir_from_rv
from sim.utils.ground_track import ground_track_from_eci_history
from sim.utils.quaternion import quaternion_to_dcm_bn

AVAILABLE_FIGURE_IDS = [
    "orbit_eci",
    "trajectory_ecef",
    "trajectory_ric_rect",
    "trajectory_ric_curv",
    "trajectory_ric_rect_2d",
    "trajectory_ric_curv_2d",
    "trajectory_eci_multi",
    "trajectory_ecef_multi",
    "trajectory_ric_rect_multi",
    "trajectory_ric_curv_multi",
    "trajectory_ric_rect_2d_multi",
    "trajectory_ric_curv_2d_multi",
    "attitude",
    "quaternion_eci",
    "quaternion_ric",
    "rates_eci",
    "rates_ric",
    "relative_range",
    "knowledge_timeline",
    "control_thrust",
    "control_thrust_multi",
    "control_thrust_ric",
    "control_thrust_ric_multi",
    "quaternion_error",
    "rocket_ascent_diagnostics",
    "rocket_orbital_elements",
    "rocket_fuel_remaining",
    "satellite_delta_v_remaining",
    "thrust_alignment_error",
]

AVAILABLE_ANIMATION_TYPES = [
    "ground_track",
    "ground_track_multi",
    "ric_curv_prism_multi",
    "ric_prism_side_by_side",
]


def _load_plotting_functions() -> dict[str, Any]:
    from sim.utils.plotting import plot_attitude_tumble, plot_orbit_eci
    from sim.utils.plotting_capabilities import (
        animate_ground_track,
        animate_multi_ground_track,
        animate_multi_rectangular_prism_ric_curv,
        animate_side_by_side_rectangular_prism_ric_attitude,
        plot_body_rates,
        plot_control_commands,
        plot_multi_control_commands,
        plot_multi_ric_2d_projections,
        plot_multi_trajectory_frame,
        plot_quaternion_components,
        plot_ric_2d_projections,
        plot_trajectory_frame,
    )

    return {
        "plot_orbit_eci": plot_orbit_eci,
        "plot_attitude_tumble": plot_attitude_tumble,
        "plot_body_rates": plot_body_rates,
        "plot_control_commands": plot_control_commands,
        "plot_multi_control_commands": plot_multi_control_commands,
        "plot_multi_ric_2d_projections": plot_multi_ric_2d_projections,
        "plot_multi_trajectory_frame": plot_multi_trajectory_frame,
        "plot_quaternion_components": plot_quaternion_components,
        "plot_ric_2d_projections": plot_ric_2d_projections,
        "plot_trajectory_frame": plot_trajectory_frame,
        "animate_ground_track": animate_ground_track,
        "animate_multi_ground_track": animate_multi_ground_track,
        "animate_multi_rectangular_prism_ric_curv": animate_multi_rectangular_prism_ric_curv,
        "animate_side_by_side_rectangular_prism_ric_attitude": animate_side_by_side_rectangular_prism_ric_attitude,
    }


def _quat_error_angle_deg(q_des: np.ndarray, q_cur: np.ndarray) -> float:
    qd = np.array(q_des, dtype=float).reshape(-1)
    qc = np.array(q_cur, dtype=float).reshape(-1)
    if qd.size != 4 or qc.size != 4:
        return float("nan")
    nd = float(np.linalg.norm(qd))
    nc = float(np.linalg.norm(qc))
    if nd <= 0.0 or nc <= 0.0:
        return float("nan")
    qd /= nd
    qc /= nc
    dot = float(np.clip(np.dot(qd, qc), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(abs(dot))))


def _orbital_elements_basic(
    r_km: np.ndarray,
    v_km_s: np.ndarray,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> tuple[float, float]:
    r = float(np.linalg.norm(r_km))
    v2 = float(np.dot(v_km_s, v_km_s))
    if r <= 0.0:
        return np.inf, np.inf
    eps = 0.5 * v2 - mu_km3_s2 / r
    a = np.inf if abs(eps) < 1e-14 else float(-mu_km3_s2 / (2.0 * eps))
    h = np.cross(r_km, v_km_s)
    e_vec = np.cross(v_km_s, h) / mu_km3_s2 - r_km / r
    e = float(np.linalg.norm(e_vec))
    return a, e


def plot_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    thrust_hist: dict[str, np.ndarray],
    desired_attitude_hist: dict[str, np.ndarray] | None,
    knowledge_hist: dict[str, dict[str, np.ndarray]],
    rocket_metrics: dict[str, np.ndarray] | None,
    outdir: Path,
    resolve_rocket_stack: Callable[[dict[str, Any]], RocketStackPreset],
    resolve_satellite_isp_s: Callable[[dict[str, Any]], float],
) -> dict[str, str]:
    out: dict[str, str] = {}
    if not bool(cfg.outputs.plots.get("enabled", True)):
        return out
    mode = cfg.outputs.mode
    figure_ids = list(cfg.outputs.plots.get("figure_ids", []) or [])
    ric_2d_planes = list(cfg.outputs.plots.get("ric_2d_planes", ["ri", "ic", "rc"]) or ["ri", "ic", "rc"])
    reference_object_id = str(cfg.outputs.plots.get("reference_object_id", "")).strip()
    if reference_object_id and reference_object_id not in truth_hist:
        reference_object_id = ""
    if not reference_object_id and "target" in truth_hist:
        reference_object_id = "target"
    if not reference_object_id and truth_hist:
        reference_object_id = sorted(truth_hist.keys())[0]
    reference_truth = truth_hist.get(reference_object_id) if reference_object_id else None
    ric_truth_hist = (
        {oid: hist for oid, hist in truth_hist.items() if oid != reference_object_id}
        if reference_object_id
        else dict(truth_hist)
    )
    if not figure_ids:
        return out
    plot_fns = _load_plotting_functions()
    plot_orbit_eci = plot_fns["plot_orbit_eci"]
    plot_attitude_tumble = plot_fns["plot_attitude_tumble"]
    plot_body_rates = plot_fns["plot_body_rates"]
    plot_control_commands = plot_fns["plot_control_commands"]
    plot_multi_control_commands = plot_fns["plot_multi_control_commands"]
    plot_multi_ric_2d_projections = plot_fns["plot_multi_ric_2d_projections"]
    plot_multi_trajectory_frame = plot_fns["plot_multi_trajectory_frame"]
    plot_quaternion_components = plot_fns["plot_quaternion_components"]
    plot_ric_2d_projections = plot_fns["plot_ric_2d_projections"]
    plot_trajectory_frame = plot_fns["plot_trajectory_frame"]
    for oid, hist in truth_hist.items():
        if not np.any(np.isfinite(hist[:, 0])):
            continue
        if "orbit_eci" in figure_ids:
            p = outdir / f"{oid}_orbit_eci.png"
            plot_orbit_eci(hist, mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_orbit_eci"] = str(p)
        if "attitude" in figure_ids:
            p = outdir / f"{oid}_attitude.png"
            plot_attitude_tumble(t_s=t_s, truth_hist=hist, mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_attitude"] = str(p)

    if "relative_range" in figure_ids:
        import matplotlib.pyplot as plt

        ids = list(truth_hist.keys())
        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = truth_hist[ids[i]][:, :3]
                b = truth_hist[ids[j]][:, :3]
                mask = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
                if not np.any(mask):
                    continue
                rr = np.linalg.norm(a - b, axis=1)
                ax.plot(t_s[mask], rr[mask], label=f"{ids[i]}-{ids[j]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Range (km)")
        ax.set_title("Relative Range")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        p = outdir / "relative_ranges.png"
        if mode in ("save", "both"):
            fig.savefig(p, dpi=int(cfg.outputs.plots.get("dpi", 150)))
            out["relative_ranges"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    if "quaternion_error" in figure_ids and desired_attitude_hist is not None:
        import matplotlib.pyplot as plt

        for oid, hist in truth_hist.items():
            q_des_hist = desired_attitude_hist.get(oid) if isinstance(desired_attitude_hist, dict) else None
            if q_des_hist is None or q_des_hist.shape[0] == 0:
                continue
            n_s = min(hist.shape[0], q_des_hist.shape[0], t_s.size)
            if n_s <= 0:
                continue
            qd = np.array(q_des_hist[:n_s, :], dtype=float)
            qc = np.array(hist[:n_s, 6:10], dtype=float)
            for k in range(1, n_s):
                if not np.all(np.isfinite(qd[k, :])) and np.all(np.isfinite(qd[k - 1, :])):
                    qd[k, :] = qd[k - 1, :]
            err_deg = np.full(n_s, np.nan, dtype=float)
            for k in range(n_s):
                if not (np.all(np.isfinite(qd[k, :])) and np.all(np.isfinite(qc[k, :]))):
                    continue
                err_deg[k] = _quat_error_angle_deg(qd[k, :], qc[k, :])
            finite = np.isfinite(err_deg)
            if not np.any(finite):
                continue
            fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
            ax.plot(t_s[:n_s][finite], err_deg[finite], linewidth=1.4)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Error Angle (deg)")
            ax.set_title(f"Quaternion Tracking Error ({oid})")
            ax.grid(True, alpha=0.3)
            p = outdir / f"{oid}_quaternion_error.png"
            if mode in ("save", "both"):
                fig.savefig(p, dpi=int(cfg.outputs.plots.get("dpi", 150)))
                out[f"{oid}_quaternion_error"] = str(p)
            if mode in ("interactive", "both"):
                plt.show(block=False)
            else:
                plt.close(fig)

    if "trajectory_eci_multi" in figure_ids:
        p = outdir / "trajectory_eci_multi.png"
        plot_multi_trajectory_frame(t_s, truth_hist, frame="eci", mode=mode, out_path=str(p))
        if mode in ("save", "both"):
            out["trajectory_eci_multi"] = str(p)
    if "trajectory_ecef_multi" in figure_ids:
        p = outdir / "trajectory_ecef_multi.png"
        plot_multi_trajectory_frame(t_s, truth_hist, frame="ecef", mode=mode, out_path=str(p))
        if mode in ("save", "both"):
            out["trajectory_ecef_multi"] = str(p)
    if "trajectory_ric_rect_multi" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_rect_multi.png"
        plot_multi_trajectory_frame(
            t_s,
            ric_truth_hist,
            frame="ric_rect",
            reference_truth_hist=reference_truth,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_rect_multi"] = str(p)
    if "trajectory_ric_curv_multi" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_curv_multi.png"
        plot_multi_trajectory_frame(
            t_s,
            ric_truth_hist,
            frame="ric_curv",
            reference_truth_hist=reference_truth,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_curv_multi"] = str(p)
    if "trajectory_ric_rect_2d_multi" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_rect_2d_multi.png"
        plot_multi_ric_2d_projections(
            t_s,
            ric_truth_hist,
            frame="ric_rect",
            reference_truth_hist=reference_truth,
            planes=ric_2d_planes,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_rect_2d_multi"] = str(p)
    if "trajectory_ric_curv_2d_multi" in figure_ids and reference_truth is not None:
        p = outdir / "trajectory_ric_curv_2d_multi.png"
        plot_multi_ric_2d_projections(
            t_s,
            ric_truth_hist,
            frame="ric_curv",
            reference_truth_hist=reference_truth,
            planes=ric_2d_planes,
            mode=mode,
            out_path=str(p),
        )
        if mode in ("save", "both"):
            out["trajectory_ric_curv_2d_multi"] = str(p)

    for oid, hist in truth_hist.items():
        if not np.any(np.isfinite(hist[:, 0])):
            continue
        if "quaternion_eci" in figure_ids:
            p = outdir / f"{oid}_quat_eci.png"
            plot_quaternion_components(t_s, hist, frame="eci", layout="single", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_quat_eci"] = str(p)
        if "quaternion_ric" in figure_ids:
            p = outdir / f"{oid}_quat_ric.png"
            plot_quaternion_components(t_s, hist, frame="ric", layout="single", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_quat_ric"] = str(p)
        if "rates_eci" in figure_ids:
            p = outdir / f"{oid}_rates_eci.png"
            plot_body_rates(t_s, hist, frame="eci", layout="subplots", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_rates_eci"] = str(p)
        if "rates_ric" in figure_ids:
            p = outdir / f"{oid}_rates_ric.png"
            plot_body_rates(t_s, hist, frame="ric", layout="subplots", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_rates_ric"] = str(p)
        if "trajectory_ecef" in figure_ids:
            p = outdir / f"{oid}_traj_ecef.png"
            plot_trajectory_frame(t_s, hist, frame="ecef", mode=mode, out_path=str(p))
            if mode in ("save", "both"):
                out[f"{oid}_traj_ecef"] = str(p)
        if "trajectory_ric_rect" in figure_ids and reference_truth is not None and oid != reference_object_id:
            p = outdir / f"{oid}_traj_ric_rect.png"
            plot_trajectory_frame(
                t_s,
                hist,
                frame="ric_rect",
                reference_truth_hist=reference_truth,
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_traj_ric_rect"] = str(p)
        if "trajectory_ric_curv" in figure_ids and reference_truth is not None and oid != reference_object_id:
            p = outdir / f"{oid}_traj_ric_curv.png"
            plot_trajectory_frame(
                t_s,
                hist,
                frame="ric_curv",
                reference_truth_hist=reference_truth,
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_traj_ric_curv"] = str(p)
        if "trajectory_ric_rect_2d" in figure_ids and reference_truth is not None and oid != reference_object_id:
            p = outdir / f"{oid}_traj_ric_rect_2d.png"
            plot_ric_2d_projections(
                t_s,
                hist,
                frame="ric_rect",
                reference_truth_hist=reference_truth,
                planes=ric_2d_planes,
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_traj_ric_rect_2d"] = str(p)
        if "trajectory_ric_curv_2d" in figure_ids and reference_truth is not None and oid != reference_object_id:
            p = outdir / f"{oid}_traj_ric_curv_2d.png"
            plot_ric_2d_projections(
                t_s,
                hist,
                frame="ric_curv",
                reference_truth_hist=reference_truth,
                planes=ric_2d_planes,
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_traj_ric_curv_2d"] = str(p)

    if "rocket_ascent_diagnostics" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        r = x[:, 0:3]
        v = x[:, 3:6]
        m = x[:, 13]
        alt_km = np.linalg.norm(r, axis=1) - EARTH_RADIUS_KM
        speed_km_s = np.linalg.norm(v, axis=1)
        q_dyn = np.zeros_like(t_s)
        mach = np.zeros_like(t_s)
        stage = np.zeros_like(t_s)
        throttle = np.zeros_like(t_s)
        if rocket_metrics is not None:
            if "q_dyn_pa" in rocket_metrics:
                q_dyn = np.array(rocket_metrics["q_dyn_pa"], dtype=float).reshape(-1)[: t_s.size]
            if "mach" in rocket_metrics:
                mach = np.array(rocket_metrics["mach"], dtype=float).reshape(-1)[: t_s.size]
            if "stage_index" in rocket_metrics:
                stage = np.array(rocket_metrics["stage_index"], dtype=float).reshape(-1)[: t_s.size]
            if "throttle_cmd" in rocket_metrics:
                throttle = np.array(rocket_metrics["throttle_cmd"], dtype=float).reshape(-1)[: t_s.size]
        a_cmd = np.linalg.norm(np.nan_to_num(thrust_hist.get("rocket", np.zeros((t_s.size, 3))), nan=0.0), axis=1)

        fig, ax = plt.subplots(4, 1, figsize=cap_figsize(11, 11), sharex=True)

        ax0r = ax[0].twinx()
        l00 = ax[0].plot(t_s, alt_km, label="altitude (km)", color="tab:blue")
        l01 = ax0r.plot(t_s, speed_km_s, label="speed (km/s)", color="tab:orange")
        ax[0].set_ylabel("altitude (km)")
        ax0r.set_ylabel("speed (km/s)")
        ax[0].set_title("Rocket Ascent: Altitude and Speed")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(l00 + l01, [ln.get_label() for ln in (l00 + l01)], loc="best")

        ax1r = ax[1].twinx()
        l10 = ax[1].plot(t_s, q_dyn, label="q_dyn (Pa)", color="tab:green")
        l11 = ax1r.plot(t_s, mach, label="Mach", color="tab:red")
        ax[1].set_ylabel("dynamic pressure (Pa)")
        ax1r.set_ylabel("Mach")
        ax[1].set_title("Dynamic Pressure and Mach")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(l10 + l11, [ln.get_label() for ln in (l10 + l11)], loc="best")

        ax2r = ax[2].twinx()
        l20 = ax[2].plot(t_s, m, label="mass (kg)", color="tab:purple")
        l21 = ax2r.step(t_s, stage, where="post", label="stage index", color="tab:brown")
        ax[2].set_ylabel("mass (kg)")
        ax2r.set_ylabel("stage index")
        ax[2].set_title("Mass and Stage")
        ax[2].grid(True, alpha=0.3)
        ax[2].legend(l20 + l21, [ln.get_label() for ln in (l20 + l21)], loc="best")

        ax3r = ax[3].twinx()
        l30 = ax[3].plot(t_s, throttle, label="throttle", color="tab:cyan")
        l31 = ax3r.plot(t_s, a_cmd, label="|a_cmd| (km/s^2)", color="tab:gray")
        ax[3].set_ylabel("throttle")
        ax3r.set_ylabel("|a_cmd| (km/s^2)")
        ax[3].set_xlabel("time (s)")
        ax[3].set_title("Throttle and Commanded Acceleration")
        ax[3].grid(True, alpha=0.3)
        ax[3].legend(l30 + l31, [ln.get_label() for ln in (l30 + l31)], loc="best")
        fig.tight_layout()
        p = outdir / "rocket_ascent_diagnostics.png"
        if mode in ("save", "both"):
            fig.savefig(p, dpi=int(cfg.outputs.plots.get("dpi", 150)))
            out["rocket_ascent_diagnostics"] = str(p)
        if mode == "save":
            plt.close(fig)

    if "rocket_orbital_elements" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        a_km = np.full(t_s.size, np.nan, dtype=float)
        e = np.full(t_s.size, np.nan, dtype=float)
        for k in range(min(t_s.size, x.shape[0])):
            a_km[k], e[k] = _orbital_elements_basic(x[k, 0:3], x[k, 3:6], EARTH_MU_KM3_S2)

        fig, ax = plt.subplots(2, 1, figsize=cap_figsize(10, 7), sharex=True)
        ax[0].plot(t_s, a_km)
        ax[0].set_ylabel("a (km)")
        ax[0].set_title("Rocket Orbital Elements")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(t_s, e)
        ax[1].set_ylabel("e")
        ax[1].set_xlabel("time (s)")
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        p = outdir / "rocket_orbital_elements.png"
        if mode in ("save", "both"):
            fig.savefig(p, dpi=int(cfg.outputs.plots.get("dpi", 150)))
            out["rocket_orbital_elements"] = str(p)
        if mode == "save":
            plt.close(fig)

    if "rocket_fuel_remaining" in figure_ids and "rocket" in truth_hist:
        import matplotlib.pyplot as plt

        x = truth_hist["rocket"]
        m = np.array(x[:, 13], dtype=float).reshape(-1)
        stack = resolve_rocket_stack(dict(cfg.rocket.specs or {}))
        payload_kg = float((cfg.rocket.specs or {}).get("payload_mass_kg", 150.0))
        dry_total_kg = float(sum(float(s.dry_mass_kg) for s in stack.stages) + payload_kg)
        prop0_kg = float(sum(float(s.propellant_mass_kg) for s in stack.stages))
        if prop0_kg > 0.0:
            fuel_rem_kg = np.clip(m - dry_total_kg, 0.0, prop0_kg)
            fuel_pct = 100.0 * fuel_rem_kg / prop0_kg
        else:
            fuel_pct = np.zeros_like(m)

        fig, ax = plt.subplots(figsize=cap_figsize(10, 4.5))
        ax.plot(t_s, fuel_pct, linewidth=1.6)
        ax.set_ylim(-1.0, 101.0)
        ax.set_ylabel("Fuel Remaining (%)")
        ax.set_xlabel("time (s)")
        ax.set_title("Rocket Fuel Remaining")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = outdir / "rocket_fuel_remaining.png"
        if mode in ("save", "both"):
            fig.savefig(p, dpi=int(cfg.outputs.plots.get("dpi", 150)))
            out["rocket_fuel_remaining"] = str(p)
        if mode == "save":
            plt.close(fig)

    if "satellite_delta_v_remaining" in figure_ids:
        import matplotlib.pyplot as plt

        g0_m_s2 = 9.80665
        section_by_id = {"chaser": cfg.chaser, "target": cfg.target}
        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        plotted = False
        for oid in ("chaser", "target"):
            hist = truth_hist.get(oid)
            sec = section_by_id.get(oid)
            if hist is None or sec is None or hist.shape[0] == 0:
                continue
            specs = dict(getattr(sec, "specs", {}) or {})
            dry_mass_kg = float(specs.get("dry_mass_kg", np.nan))
            fuel_mass_kg = float(specs.get("fuel_mass_kg", np.nan))
            if not (np.isfinite(dry_mass_kg) and np.isfinite(fuel_mass_kg)):
                continue
            if dry_mass_kg <= 0.0 or fuel_mass_kg < 0.0:
                continue
            m0 = dry_mass_kg + fuel_mass_kg
            if m0 <= dry_mass_kg:
                continue
            isp_s = resolve_satellite_isp_s(specs)
            if isp_s <= 0.0:
                continue
            dv0_m_s = float(isp_s * g0_m_s2 * np.log(m0 / dry_mass_kg))
            if dv0_m_s <= 0.0:
                continue
            m_hist = np.clip(np.array(hist[:, 13], dtype=float), dry_mass_kg, m0)
            dv_rem_m_s = isp_s * g0_m_s2 * np.log(m_hist / dry_mass_kg)
            pct = np.clip(100.0 * dv_rem_m_s / dv0_m_s, 0.0, 100.0)
            ax.plot(t_s[: pct.size], pct, label=f"{oid}")
            plotted = True

        if plotted:
            ax.set_ylim(-1.0, 101.0)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("Delta-V Remaining (%)")
            ax.set_title("Satellite Delta-V Remaining")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            p = outdir / "satellite_delta_v_remaining.png"
            if mode in ("save", "both"):
                fig.savefig(p, dpi=int(cfg.outputs.plots.get("dpi", 150)))
                out["satellite_delta_v_remaining"] = str(p)
            if mode == "save":
                plt.close(fig)
        else:
            plt.close(fig)

    thrust_hist_ric: dict[str, np.ndarray] = {}
    if ("control_thrust_ric" in figure_ids) or ("control_thrust_ric_multi" in figure_ids):
        for oid, u in thrust_hist.items():
            hist = truth_hist.get(oid)
            if hist is None or hist.size == 0:
                continue
            n_s = min(u.shape[0], hist.shape[0], t_s.size)
            ur = np.full((u.shape[0], 3), np.nan, dtype=float)
            for k in range(n_s):
                a_eci = np.array(u[k, :], dtype=float)
                rv = np.array(hist[k, 0:6], dtype=float)
                if not (np.all(np.isfinite(a_eci)) and np.all(np.isfinite(rv))):
                    continue
                c_ir = ric_dcm_ir_from_rv(rv[:3], rv[3:6])
                ur[k, :] = c_ir.T @ a_eci
            thrust_hist_ric[oid] = ur

    if "control_thrust" in figure_ids:
        for oid, u in thrust_hist.items():
            if not np.any(np.isfinite(u[:, 0])):
                continue
            p = outdir / f"{oid}_control_thrust.png"
            plot_control_commands(
                t_s,
                u,
                layout="subplots",
                input_labels=["ax", "ay", "az"],
                title=f"Thrust Commands ({oid})",
                y_label="km/s^2",
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_control_thrust"] = str(p)

    if "control_thrust_ric" in figure_ids:
        for oid, u in thrust_hist_ric.items():
            if not np.any(np.isfinite(u[:, 0])):
                continue
            p = outdir / f"{oid}_control_thrust_ric.png"
            plot_control_commands(
                t_s,
                u,
                layout="subplots",
                input_labels=["aR", "aI", "aC"],
                title=f"Thrust Commands RIC ({oid})",
                y_label="km/s^2",
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"{oid}_control_thrust_ric"] = str(p)

    if "control_thrust_multi" in figure_ids:
        for i_comp, lbl in enumerate(("ax", "ay", "az")):
            p = outdir / f"control_thrust_multi_{lbl}.png"
            plot_multi_control_commands(
                t_s,
                thrust_hist,
                component_index=i_comp,
                title=f"Thrust Command Overlay ({lbl})",
                y_label="km/s^2",
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"control_thrust_multi_{lbl}"] = str(p)

    if "control_thrust_ric_multi" in figure_ids:
        for i_comp, lbl in enumerate(("aR", "aI", "aC")):
            p = outdir / f"control_thrust_ric_multi_{lbl}.png"
            plot_multi_control_commands(
                t_s,
                thrust_hist_ric,
                component_index=i_comp,
                title=f"Thrust Command Overlay RIC ({lbl})",
                y_label="km/s^2",
                mode=mode,
                out_path=str(p),
            )
            if mode in ("save", "both"):
                out[f"control_thrust_ric_multi_{lbl}"] = str(p)

    if "thrust_alignment_error" in figure_ids:
        import matplotlib.pyplot as plt

        thrust_dir_body = np.array(cfg.outputs.plots.get("thrust_direction_body", [1.0, 0.0, 0.0]), dtype=float).reshape(-1)
        if thrust_dir_body.size != 3:
            thrust_dir_body = np.array([1.0, 0.0, 0.0], dtype=float)
        n_t = float(np.linalg.norm(thrust_dir_body))
        if n_t <= 0.0:
            thrust_dir_body = np.array([1.0, 0.0, 0.0], dtype=float)
            n_t = 1.0
        thrust_dir_body = thrust_dir_body / n_t

        for oid, hist in truth_hist.items():
            u = thrust_hist.get(oid)
            if u is None or hist.size == 0:
                continue
            thrust_norm = np.linalg.norm(np.nan_to_num(u, nan=0.0), axis=1)
            if not np.any(thrust_norm > 1e-15):
                continue
            err_deg = np.full(t_s.shape, np.nan, dtype=float)
            for k in range(min(hist.shape[0], u.shape[0], t_s.size)):
                a_cmd = np.array(u[k, :], dtype=float)
                if not np.all(np.isfinite(a_cmd)):
                    continue
                a_norm = float(np.linalg.norm(a_cmd))
                if a_norm <= 1e-15:
                    continue
                q_bn = np.array(hist[k, 6:10], dtype=float)
                if not np.all(np.isfinite(q_bn)):
                    continue
                c_bn = quaternion_to_dcm_bn(q_bn)
                thrust_axis_eci = c_bn.T @ thrust_dir_body
                burn_dir_eci = -a_cmd / a_norm
                cosang = float(np.clip(np.dot(thrust_axis_eci, burn_dir_eci), -1.0, 1.0))
                if not np.isfinite(cosang):
                    continue
                err_deg[k] = float(np.degrees(np.arccos(cosang)))

            fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
            finite = np.isfinite(err_deg)
            if np.any(finite):
                t_f = np.array(t_s[finite], dtype=float)
                e_f = np.array(err_deg[finite], dtype=float)
                ax.plot(t_f, e_f, linewidth=1.2, marker="o", markersize=2.5)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid burn/alignment samples in this run",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Angle Error (deg)")
            ax.set_title(f"Attitude vs Thrust Vector Error ({oid})")
            ax.grid(True, alpha=0.3)
            p = outdir / f"{oid}_thrust_alignment_error.png"
            if mode in ("save", "both"):
                fig.savefig(p, dpi=int(cfg.outputs.plots.get("dpi", 150)))
                out[f"{oid}_thrust_alignment_error"] = str(p)
            if mode in ("interactive", "both"):
                plt.show(block=False)
            else:
                plt.close(fig)

    if "knowledge_timeline" in figure_ids:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=cap_figsize(10, 5))
        i = 0
        for obs, by_tgt in knowledge_hist.items():
            for tgt, hist in by_tgt.items():
                known = np.any(np.isfinite(hist), axis=1).astype(float)
                ax.plot(t_s, known + i * 1.2, label=f"{obs}->{tgt}")
                i += 1
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Known (offset)")
        ax.set_title("Knowledge Timeline")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        p = outdir / "knowledge_timeline.png"
        if mode in ("save", "both"):
            fig.savefig(p, dpi=int(cfg.outputs.plots.get("dpi", 150)))
            out["knowledge_timeline"] = str(p)
        if mode in ("interactive", "both"):
            plt.show(block=False)
        else:
            plt.close(fig)

    return out


def animate_outputs(
    *,
    cfg: SimulationScenarioConfig,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    outdir: Path,
) -> dict[str, str]:
    out: dict[str, str] = {}
    anim_cfg = dict(cfg.outputs.animations or {})
    if not bool(anim_cfg.get("enabled", False)):
        return out

    mode = cfg.outputs.mode
    fps = float(anim_cfg.get("fps", 30.0))
    speed_multiple = float(anim_cfg.get("speed_multiple", 10.0))
    frame_stride = int(anim_cfg.get("frame_stride", 1))
    draw_earth_map = bool(anim_cfg.get("draw_earth_map", True))
    types = list(anim_cfg.get("types", []) or [])
    if not types:
        return out
    plot_fns = _load_plotting_functions()
    animate_ground_track = plot_fns["animate_ground_track"]
    animate_multi_ground_track = plot_fns["animate_multi_ground_track"]
    animate_multi_rectangular_prism_ric_curv = plot_fns["animate_multi_rectangular_prism_ric_curv"]
    animate_side_by_side_rectangular_prism_ric_attitude = plot_fns["animate_side_by_side_rectangular_prism_ric_attitude"]

    if "ground_track_multi" in types:
        p = outdir / "ground_track_multi.mp4"
        animate_multi_ground_track(
            t_s=t_s,
            truth_hist_by_object=truth_hist,
            jd_utc_start=cfg.simulator.initial_jd_utc,
            mode=mode,
            out_path=str(p),
            fps=fps,
            speed_multiple=speed_multiple,
            draw_earth_map=draw_earth_map,
            frame_stride=frame_stride,
        )
        if mode in ("save", "both"):
            out["ground_track_multi"] = str(p)

    if "ground_track" in types:
        for oid, hist in truth_hist.items():
            if hist.size == 0 or not np.any(np.isfinite(hist[:, 0])):
                continue
            lat_deg, lon_deg, _ = ground_track_from_eci_history(
                hist[:, :3],
                t_s=t_s,
                jd_utc_start=cfg.simulator.initial_jd_utc,
            )
            p = outdir / f"{oid}_ground_track.mp4"
            animate_ground_track(
                lon_deg=lon_deg,
                lat_deg=lat_deg,
                t_s=t_s,
                jd_utc_start=cfg.simulator.initial_jd_utc,
                mode=mode,
                out_path=str(p),
                fps=fps,
                speed_multiple=speed_multiple,
                draw_earth_map=draw_earth_map,
                frame_stride=frame_stride,
            )
            if mode in ("save", "both"):
                out[f"{oid}_ground_track"] = str(p)

    if "ric_curv_prism_multi" in types:
        p = outdir / "ric_curv_prism_multi.mp4"
        target_object_id = str(anim_cfg.get("target_object_id", "target"))
        prism_obj_ids = anim_cfg.get("ric_curv_prism_object_ids")
        if not isinstance(prism_obj_ids, list):
            prism_obj_ids = None
        dims_map_raw = anim_cfg.get("ric_curv_prism_dims_m", {})
        dims_map = dict(dims_map_raw) if isinstance(dims_map_raw, dict) else {}
        animate_multi_rectangular_prism_ric_curv(
            t_s=t_s,
            truth_hist_by_object=truth_hist,
            target_object_id=target_object_id,
            object_ids=prism_obj_ids,
            prism_dims_m_by_object=dims_map,
            mode=mode,
            out_path=str(p),
            fps=fps,
            speed_multiple=speed_multiple,
            frame_stride=frame_stride,
        )
        if mode in ("save", "both"):
            out["ric_curv_prism_multi"] = str(p)

    if "ric_prism_side_by_side" in types:
        p = outdir / "ric_prism_side_by_side.mp4"
        left_object_id = str(anim_cfg.get("ric_side_by_side_left_object_id", "target"))
        right_object_id = str(anim_cfg.get("ric_side_by_side_right_object_id", "chaser"))
        dims_map_raw = anim_cfg.get("ric_side_by_side_dims_m", {})
        dims_map = dict(dims_map_raw) if isinstance(dims_map_raw, dict) else {}
        animate_side_by_side_rectangular_prism_ric_attitude(
            t_s=t_s,
            truth_hist_by_object=truth_hist,
            left_object_id=left_object_id,
            right_object_id=right_object_id,
            prism_dims_m_by_object=dims_map,
            mode=mode,
            out_path=str(p),
            fps=fps,
            speed_multiple=speed_multiple,
            frame_stride=frame_stride,
        )
        if mode in ("save", "both"):
            out["ric_prism_side_by_side"] = str(p)

    return out
