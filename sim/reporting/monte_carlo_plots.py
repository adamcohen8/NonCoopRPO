from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.utils.figure_size import cap_figsize


def write_monte_carlo_plot_artifacts(
    *,
    cfg: SimulationScenarioConfig,
    outdir: Path,
    agg: dict[str, Any],
    runs: list[dict[str, Any]],
    run_details: list[dict[str, Any]],
    relative_range_series_runs: list[dict[str, np.ndarray] | None],
    durations_s: np.ndarray,
    ca_finite: np.ndarray,
    all_obj_ids: list[str],
    dv_by_object: dict[str, list[float]],
    dv_remaining_m_s_by_object: dict[str, list[float]],
    dv_budget_m_s_by_object: dict[str, float],
    failure_mode_counts: dict[str, int],
    keepout_threshold: float,
    gates: dict[str, Any],
    mc_out_cfg: dict[str, Any],
) -> dict[str, Any]:
    from sim.master_simulator import _mc_initial_relative_ric_curv_samples, _safe_float

    save_hist = bool(cfg.outputs.monte_carlo.get("save_histograms", False))
    show_hist = bool(cfg.outputs.monte_carlo.get("display_histograms", False))
    if (save_hist or show_hist) and runs:
        import matplotlib.pyplot as plt

        plot_series: list[tuple[str, np.ndarray]] = [("Duration (s)", durations_s)]
        if ca_finite.size:
            plot_series.append(("Closest Approach (km)", ca_finite))
        for oid in all_obj_ids:
            dv_arr = np.array(dv_by_object.get(oid, []), dtype=float)
            if dv_arr.size:
                plot_series.append((f"{oid} Total dV (m/s)", dv_arr))
        for oid in ("chaser", "target"):
            rem_arr = np.array(dv_remaining_m_s_by_object.get(oid, []), dtype=float)
            if rem_arr.size:
                plot_series.append((f"{oid} dV Remaining (m/s)", rem_arr))
        nplots = len(plot_series)
        if nplots == 6:
            nrows, ncols = 3, 2
        else:
            ncols = min(3, max(1, nplots))
            nrows = int(np.ceil(nplots / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=cap_figsize(5.2 * ncols, 3.8 * nrows), squeeze=False)
        axes_flat = list(np.ravel(axes))
        for ax, (title, arr) in zip(axes_flat, plot_series):
            bins = int(max(5, min(30, np.sqrt(max(arr.size, 1)))))
            ax.hist(arr, bins=bins, alpha=0.85)
            ax.set_title(title)
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[nplots:]:
            ax.set_visible(False)
        fig.tight_layout()
        if save_hist:
            fig.savefig(str(outdir / "master_monte_carlo_histograms.png"), dpi=int(cfg.outputs.plots.get("dpi", 150)))
        if show_hist:
            plt.show()
        plt.close(fig)

        range_series_available = [s for s in relative_range_series_runs if isinstance(s, dict)]
        if range_series_available:
            fig_rr, ax_rr = plt.subplots(figsize=cap_figsize(10, 6))
            for idx, series in enumerate(relative_range_series_runs):
                if not isinstance(series, dict):
                    continue
                t_rr = np.array(series.get("time_s", []), dtype=float)
                r_rr = np.array(series.get("range_km", []), dtype=float)
                if t_rr.size == 0 or r_rr.size == 0:
                    continue
                ax_rr.plot(t_rr, r_rr, linewidth=1.0, alpha=0.65, label=f"run {idx}")
            ax_rr.set_title("Chaser-Target Relative Range by Iteration")
            ax_rr.set_xlabel("Time (s)")
            ax_rr.set_ylabel("Range (km)")
            ax_rr.grid(True, alpha=0.3)
            fig_rr.tight_layout()
            rr_path = outdir / "master_monte_carlo_relative_range_timeseries.png"
            if save_hist:
                fig_rr.savefig(str(rr_path), dpi=int(cfg.outputs.plots.get("dpi", 150)))
                agg["artifacts"]["relative_range_timeseries_png"] = str(rr_path)
            if show_hist:
                plt.show()
            plt.close(fig_rr)

    save_ops_dashboard = bool(mc_out_cfg.get("save_ops_dashboard", True))
    show_ops_dashboard = bool(mc_out_cfg.get("display_ops_dashboard", False))
    if (save_ops_dashboard or show_ops_dashboard) and run_details:
        import matplotlib.pyplot as plt

        ric_initial_samples = _mc_initial_relative_ric_curv_samples(cfg, run_details)
        pass_color = np.array(["tab:green" if bool(d.get("pass", False)) else "tab:red" for d in run_details], dtype=object)
        idx_arr = np.arange(len(run_details), dtype=float)
        ca_run_arr = np.array([_safe_float(d.get("closest_approach_km")) for d in run_details], dtype=float)
        dur_run_arr = np.array([_safe_float(d.get("duration_s"), default=0.0) for d in run_details], dtype=float)
        dv_run_arr = np.array([_safe_float(d.get("total_dv_m_s_total"), default=0.0) for d in run_details], dtype=float)

        fig, axes = plt.subplots(2, 3, figsize=cap_figsize(14, 8))
        bins = int(max(5, min(30, np.sqrt(max(len(run_details), 1)))))

        finite_ca = ca_run_arr[np.isfinite(ca_run_arr)]
        axes[0, 0].hist(finite_ca, bins=bins, alpha=0.85)
        if np.isfinite(keepout_threshold):
            axes[0, 0].axvline(keepout_threshold, linestyle="--", color="k")
        axes[0, 0].set_title("Closest Approach (km)")
        axes[0, 0].set_ylabel("count")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(dur_run_arr[np.isfinite(dur_run_arr)], bins=bins, alpha=0.85)
        max_duration_s = _safe_float(gates.get("max_duration_s"))
        if np.isfinite(max_duration_s):
            axes[0, 1].axvline(max_duration_s, linestyle="--", color="k")
        axes[0, 1].set_title("Duration (s)")
        axes[0, 1].set_ylabel("count")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].hist(dv_run_arr[np.isfinite(dv_run_arr)], bins=bins, alpha=0.85, color="tab:orange")
        max_total_dv_m_s = _safe_float(gates.get("max_total_dv_m_s"))
        if np.isfinite(max_total_dv_m_s):
            axes[0, 2].axvline(max_total_dv_m_s, linestyle="--", color="k")
        axes[0, 2].set_title("Total dV (m/s)")
        axes[0, 2].set_ylabel("count")
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].scatter(idx_arr, ca_run_arr, c=pass_color, s=22, alpha=0.9)
        if np.isfinite(keepout_threshold):
            axes[1, 0].axhline(keepout_threshold, linestyle="--", color="k")
        axes[1, 0].set_title("Closest Approach by Run")
        axes[1, 0].set_xlabel("run index")
        axes[1, 0].set_ylabel("km")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(idx_arr, dv_run_arr, c=pass_color, s=22, alpha=0.9)
        if np.isfinite(max_total_dv_m_s):
            axes[1, 1].axhline(max_total_dv_m_s, linestyle="--", color="k")
        axes[1, 1].set_title("Total dV by Run")
        axes[1, 1].set_xlabel("run index")
        axes[1, 1].set_ylabel("m/s")
        axes[1, 1].grid(True, alpha=0.3)

        top_fail_pairs = sorted(failure_mode_counts.items(), key=lambda kv: int(kv[1]), reverse=True)[:6]
        if top_fail_pairs:
            labels = [k for k, _ in top_fail_pairs]
            vals = [int(v) for _, v in top_fail_pairs]
            axes[1, 2].bar(np.arange(len(vals)), vals, color="tab:red", alpha=0.85)
            axes[1, 2].set_xticks(np.arange(len(vals)))
            axes[1, 2].set_xticklabels(labels, rotation=30, ha="right")
        axes[1, 2].set_title("Failure Mode Counts")
        axes[1, 2].set_ylabel("count")
        axes[1, 2].grid(True, alpha=0.3)

        fig.suptitle("Monte Carlo Ops Dashboard", fontsize=12)
        fig.tight_layout()
        dashboard_path = outdir / "master_monte_carlo_ops_dashboard.png"
        if save_ops_dashboard:
            fig.savefig(str(dashboard_path), dpi=int(cfg.outputs.plots.get("dpi", 150)))
            agg["artifacts"]["ops_dashboard_png"] = str(dashboard_path)
        if show_ops_dashboard:
            plt.show()
        plt.close(fig)

        if ric_initial_samples:
            fig_ic, axes_ic = plt.subplots(3, 2, figsize=cap_figsize(12, 9), squeeze=False)
            scatter_specs = [
                ("radial_sep_km", "Initial Radial Separation (km)"),
                ("radial_vel_km_s", "Initial Radial Velocity (km/s)"),
                ("in_track_sep_km", "Initial In-Track Separation (km)"),
                ("in_track_vel_km_s", "Initial In-Track Velocity (km/s)"),
                ("cross_track_sep_km", "Initial Cross-Track Separation (km)"),
                ("cross_track_vel_km_s", "Initial Cross-Track Velocity (km/s)"),
            ]
            axes_ic_flat = [
                axes_ic[0, 0],
                axes_ic[0, 1],
                axes_ic[1, 0],
                axes_ic[1, 1],
                axes_ic[2, 0],
                axes_ic[2, 1],
            ]
            for ax, (key, xlabel) in zip(axes_ic_flat, scatter_specs):
                x = np.array(ric_initial_samples.get(key, []), dtype=float)
                finite = np.isfinite(x) & np.isfinite(ca_run_arr)
                ax.scatter(x[finite], ca_run_arr[finite], c=pass_color[finite], s=24, alpha=0.85)
                if np.isfinite(keepout_threshold):
                    ax.axhline(keepout_threshold, linestyle="--", color="k")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Closest Approach (km)")
                ax.grid(True, alpha=0.3)
            fig_ic.suptitle("Initial Relative RIC State vs Closest Approach", fontsize=12)
            fig_ic.tight_layout()
            ic_path = outdir / "master_monte_carlo_initial_relative_state_vs_closest_approach.png"
            if save_ops_dashboard:
                fig_ic.savefig(str(ic_path), dpi=int(cfg.outputs.plots.get("dpi", 150)))
                agg["artifacts"]["initial_relative_state_vs_closest_approach_png"] = str(ic_path)
            if show_ops_dashboard:
                plt.show()
            plt.close(fig_ic)

        rem_obj_ids = [oid for oid in ("chaser", "target") if oid in dv_budget_m_s_by_object]
        if rem_obj_ids:
            fig_rem, axes_rem = plt.subplots(2, len(rem_obj_ids), figsize=cap_figsize(5.0 * len(rem_obj_ids), 7.0), squeeze=False)
            for j, oid in enumerate(rem_obj_ids):
                rem_arr = np.array(
                    [
                        _safe_float(dict(d.get("delta_v_remaining_m_s_by_object", {}) or {}).get(oid))
                        for d in run_details
                    ],
                    dtype=float,
                )
                finite_rem = rem_arr[np.isfinite(rem_arr)]
                bins_rem = int(max(5, min(30, np.sqrt(max(finite_rem.size, 1)))))
                axes_rem[0, j].hist(finite_rem, bins=bins_rem, alpha=0.85, color="tab:blue")
                axes_rem[0, j].set_title(f"{oid} dV Remaining (m/s)")
                axes_rem[0, j].set_ylabel("count")
                axes_rem[0, j].grid(True, alpha=0.3)

                axes_rem[1, j].scatter(idx_arr, rem_arr, c=pass_color, s=22, alpha=0.9)
                axes_rem[1, j].set_title(f"{oid} dV Remaining by Run")
                axes_rem[1, j].set_xlabel("run index")
                axes_rem[1, j].set_ylabel("m/s")
                axes_rem[1, j].grid(True, alpha=0.3)
            fig_rem.suptitle("Monte Carlo Delta-V Remaining", fontsize=12)
            fig_rem.tight_layout()
            rem_path = outdir / "master_monte_carlo_delta_v_remaining.png"
            if save_ops_dashboard:
                fig_rem.savefig(str(rem_path), dpi=int(cfg.outputs.plots.get("dpi", 150)))
                agg["artifacts"]["delta_v_remaining_png"] = str(rem_path)
            if show_ops_dashboard:
                plt.show()
            plt.close(fig_rem)

    return agg
