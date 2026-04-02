from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from presets import BASIC_CHEMICAL_BOTTOM_Z, BASIC_REACTION_WHEEL_TRIAD, BASIC_SATELLITE, build_sim_object_from_presets
from sim.actuators.attitude import ReactionWheelLimits
from sim.control.attitude import RICFrameLQRController, SmallAngleLQRController
from sim.control.orbit import HCWLQRController, PredictiveBurnConfig, PredictiveBurnScheduler
from sim.core.models import Command, StateBelief
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.sensors.noisy_own_state import NoisyOwnStateSensor
from sim.utils.figure_size import cap_figsize
from sim.utils.io import write_json
from sim.utils.quaternion import quaternion_to_dcm_bn

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    qv = np.array(q, dtype=float).reshape(-1)
    n = float(np.linalg.norm(qv))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return qv / n


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3 = _quat_normalize(q1)
    b0, b1, b2, b3 = _quat_normalize(q2)
    return np.array(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ]
    )


def _quat_from_small_angle(phi: np.ndarray) -> np.ndarray:
    phi = np.array(phi, dtype=float).reshape(3)
    ang = float(np.linalg.norm(phi))
    if ang <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = phi / ang
    s = float(np.sin(0.5 * ang))
    return np.array([float(np.cos(0.5 * ang)), ax[0] * s, ax[1] * s, ax[2] * s])


def _eci_to_ric_rect(x_host_eci: np.ndarray, x_dep_eci: np.ndarray) -> np.ndarray:
    r_host = x_host_eci[:3]
    v_host = x_host_eci[3:]
    r_dep = x_dep_eci[:3]
    v_dep = x_dep_eci[3:]
    dr = r_dep - r_host
    dv = v_dep - v_host

    h = np.cross(r_host, v_host)
    in_vec = np.cross(h, r_host)
    rsw = np.column_stack((_unit(r_host), _unit(in_vec), _unit(h)))
    rtemp = np.cross(h, v_host)
    vtemp = np.cross(h, r_host)
    drsw = np.column_stack((v_host / max(np.linalg.norm(r_host), 1e-12), rtemp / max(np.linalg.norm(vtemp), 1e-12), np.zeros(3)))

    x_r = rsw.T @ dr
    frame_mv = np.array(
        [
            x_r[0] * (r_host @ v_host) / (max(np.linalg.norm(r_host), 1e-12) ** 2),
            x_r[1] * (vtemp @ rtemp) / (max(np.linalg.norm(vtemp), 1e-12) ** 2),
            0.0,
        ]
    )
    x_v = (rsw.T @ dv) + (drsw.T @ dr) - frame_mv
    return np.hstack((x_r, x_v))


def _ric_rect_to_eci(x_host_eci: np.ndarray, x_rel_rect: np.ndarray) -> np.ndarray:
    r_host = x_host_eci[:3]
    v_host = x_host_eci[3:]
    xr = x_rel_rect[:3]
    xv = x_rel_rect[3:]

    h = np.cross(r_host, v_host)
    in_vec = np.cross(h, r_host)
    rsw = np.column_stack((_unit(r_host), _unit(in_vec), _unit(h)))
    dr = np.linalg.inv(rsw.T) @ xr

    rtemp = np.cross(h, v_host)
    vtemp = np.cross(h, r_host)
    drsw = np.column_stack((v_host / max(np.linalg.norm(r_host), 1e-12), rtemp / max(np.linalg.norm(vtemp), 1e-12), np.zeros(3)))
    frame_mv = np.array(
        [
            xr[0] * (r_host @ v_host) / (max(np.linalg.norm(r_host), 1e-12) ** 2),
            xr[1] * (vtemp @ rtemp) / (max(np.linalg.norm(vtemp), 1e-12) ** 2),
            0.0,
        ]
    )
    dv = np.linalg.inv(rsw.T) @ (xv + frame_mv - (drsw.T @ dr))
    return np.hstack((r_host + dr, v_host + dv))


def _capture_time_s(distance_km: np.ndarray, dt_s: float, capture_radius_km: float, hold_s: float) -> float | None:
    hold_steps = max(1, int(np.ceil(hold_s / dt_s)))
    n = distance_km.size
    for i in range(n):
        j = min(n, i + hold_steps)
        if np.all(distance_km[i:j] <= capture_radius_km):
            return float(i * dt_s)
    return None


@dataclass(frozen=True)
class PredictiveRendezvousMCConfig:
    runs: int = 40
    base_seed: int = 1
    dt_s: float = 1.0
    duration_s: float = 5400.0
    lead_steps: int = 100
    align_deg: float = 10.0
    wheel_scale: float = 5.0
    thrust_mode: Literal["attitude", "perfect"] = "attitude"
    enable_orbit_disturbances: bool = False
    init_rel_ric_rect_mean: np.ndarray = field(default_factory=lambda: np.array([2.0, -8.0, 1.2, 0.0008, -0.0012, 0.0004]))
    init_rel_pos_sigma_km: float = 0.25
    init_rel_vel_sigma_km_s: float = 8e-5
    init_att_sigma_deg: float = 8.0
    init_rate_sigma_rad_s: float = 0.01
    capture_radius_km: float = 0.2
    capture_hold_s: float = 60.0
    pass_final_miss_km: float = 0.5
    pass_max_dv_km_s: float = 0.40
    pass_max_alignment_violations: int = 0


def run_predictive_rendezvous_monte_carlo(
    config: PredictiveRendezvousMCConfig,
    output_dir: str,
    plot_mode: Literal["interactive", "save", "both"] = "interactive",
) -> dict[str, str]:
    if config.runs <= 0:
        raise ValueError("runs must be positive.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict] = []
    for run_idx in tqdm(range(config.runs), desc="Predictive MC Runs", unit="run", dynamic_ncols=True):
        seed = config.base_seed + run_idx
        trial = _run_one_trial(seed=seed, config=config)
        trial_rows.append(trial)

    capture_times = [r["capture_time_s"] for r in trial_rows if r["capture_time_s"] is not None]
    final_miss = np.array([r["final_miss_distance_km"] for r in trial_rows], dtype=float)
    min_miss = np.array([r["min_miss_distance_km"] for r in trial_rows], dtype=float)
    dv_used = np.array([r["dv_used_km_s"] for r in trial_rows], dtype=float)
    pass_flags = np.array([1 if r["pass"] else 0 for r in trial_rows], dtype=int)

    fail_reasons: dict[str, int] = {
        "capture_not_achieved": 0,
        "final_miss_exceeded": 0,
        "dv_exceeded": 0,
        "alignment_violations": 0,
    }
    for r in trial_rows:
        for key in fail_reasons:
            if key in r["fail_reasons"]:
                fail_reasons[key] += 1

    summary = {
        "config": _config_to_dict(config),
        "aggregate": {
            "runs": int(config.runs),
            "pass_count": int(np.sum(pass_flags)),
            "pass_rate": float(np.mean(pass_flags)),
            "capture_rate": float(np.mean([1 if r["capture_time_s"] is not None else 0 for r in trial_rows])),
            "capture_time_s_mean_success": float(np.mean(capture_times)) if capture_times else None,
            "capture_time_s_median_success": float(np.median(capture_times)) if capture_times else None,
            "final_miss_km_mean": float(np.mean(final_miss)),
            "final_miss_km_p95": float(np.percentile(final_miss, 95)),
            "min_miss_km_mean": float(np.mean(min_miss)),
            "dv_used_km_s_mean": float(np.mean(dv_used)),
            "dv_used_km_s_p95": float(np.percentile(dv_used, 95)),
            "fail_reasons": fail_reasons,
        },
        "runs": trial_rows,
    }

    summary_path = out / "predictive_rendezvous_mc_summary.json"
    write_json(str(summary_path), summary)

    fig, axes = plt.subplots(2, 2, figsize=cap_figsize(12, 9))
    bins = min(20, max(5, int(np.sqrt(config.runs))))
    axes[0, 0].hist(final_miss, bins=bins, alpha=0.85)
    axes[0, 0].axvline(config.pass_final_miss_km, linestyle="--", color="k")
    axes[0, 0].set_title("Final Miss Distance")
    axes[0, 0].set_xlabel("km")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(dv_used, bins=bins, alpha=0.85, color="tab:orange")
    axes[0, 1].axvline(config.pass_max_dv_km_s, linestyle="--", color="k")
    axes[0, 1].set_title("DV Used")
    axes[0, 1].set_xlabel("km/s")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(True, alpha=0.3)

    x_idx = np.arange(config.runs)
    colors = np.where(pass_flags > 0, "tab:green", "tab:red")
    axes[1, 0].scatter(x_idx, final_miss, c=colors, s=28, alpha=0.9)
    axes[1, 0].axhline(config.pass_final_miss_km, linestyle="--", color="k")
    axes[1, 0].set_title("Final Miss by Run (green=pass)")
    axes[1, 0].set_xlabel("run index")
    axes[1, 0].set_ylabel("km")
    axes[1, 0].grid(True, alpha=0.3)

    capture_scatter = np.array([np.nan if r["capture_time_s"] is None else r["capture_time_s"] for r in trial_rows], dtype=float)
    axes[1, 1].scatter(x_idx, capture_scatter, c=colors, s=28, alpha=0.9)
    axes[1, 1].set_title("Capture Time by Run")
    axes[1, 1].set_xlabel("run index")
    axes[1, 1].set_ylabel("s")
    axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = out / "predictive_rendezvous_mc_metrics.png"
    if plot_mode in ("save", "both"):
        fig.savefig(plot_path, dpi=150)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    return {
        "summary_json": str(summary_path),
        "metrics_plot": str(plot_path) if plot_mode in ("save", "both") else "",
    }


def _run_one_trial(seed: int, config: PredictiveRendezvousMCConfig) -> dict:
    rng = np.random.default_rng(seed)
    dt_s = float(config.dt_s)
    steps = int(np.ceil(float(config.duration_s) / dt_s))

    chief = build_sim_object_from_presets(
        object_id=f"chief_mc_{seed}",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=config.enable_orbit_disturbances,
        enable_attitude_knowledge=False,
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.zeros(3),
    )
    chaser = build_sim_object_from_presets(
        object_id=f"chaser_mc_{seed}",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=config.enable_orbit_disturbances,
        enable_attitude_knowledge=True,
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.zeros(3),
    )

    x_rel0 = np.array(config.init_rel_ric_rect_mean, dtype=float).reshape(6).copy()
    x_rel0[:3] += rng.normal(0.0, float(config.init_rel_pos_sigma_km), size=3)
    x_rel0[3:] += rng.normal(0.0, float(config.init_rel_vel_sigma_km_s), size=3)
    x_chief = np.hstack((chief.truth.position_eci_km, chief.truth.velocity_eci_km_s))
    x_chaser = _ric_rect_to_eci(x_chief, x_rel0)
    chaser.truth.position_eci_km = x_chaser[:3]
    chaser.truth.velocity_eci_km_s = x_chaser[3:]

    dphi = np.deg2rad(float(config.init_att_sigma_deg)) * rng.normal(0.0, 1.0, size=3)
    q_delta = _quat_from_small_angle(dphi)
    chaser.truth.attitude_quat_bn = _quat_normalize(_quat_multiply(q_delta, chaser.truth.attitude_quat_bn))
    chaser.truth.angular_rate_body_rad_s = rng.normal(0.0, float(config.init_rate_sigma_rad_s), size=3)

    n_rad_s = np.sqrt(EARTH_MU_KM3_S2 / (np.linalg.norm(chief.truth.position_eci_km) ** 3))
    orbit_lqr = HCWLQRController(
        mean_motion_rad_s=n_rad_s,
        max_accel_km_s2=5e-5,
        design_dt_s=dt_s,
        q_weights=np.array([8.66, 8.66, 8.66, 1.33, 1.33, 1.33]) * 1e3,
        r_weights=np.ones(3) * 1.94e13,
    )
    wheel_axes = np.vstack([w.axis_body for w in BASIC_REACTION_WHEEL_TRIAD.wheels])
    wheel_limits = np.array([w.max_torque_nm for w in BASIC_REACTION_WHEEL_TRIAD.wheels], dtype=float) * float(config.wheel_scale)
    att_lqr = SmallAngleLQRController.robust_profile(
        inertia_kg_m2=BASIC_SATELLITE.inertia_kg_m2,
        wheel_axes_body=wheel_axes,
        wheel_torque_limits_nm=wheel_limits,
        design_dt_s=dt_s,
    )
    ric_att_ctrl = RICFrameLQRController(lqr=att_lqr)

    if hasattr(chaser.actuator, "attitude") and hasattr(chaser.actuator.attitude, "reaction_wheels"):
        rw = chaser.actuator.attitude.reaction_wheels
        if rw is not None:
            chaser.actuator.attitude.reaction_wheels = ReactionWheelLimits(
                max_torque_nm=rw.max_torque_nm * float(config.wheel_scale),
                max_momentum_nms=rw.max_momentum_nms * float(config.wheel_scale),
            )

    scheduler = PredictiveBurnScheduler(
        orbit_lqr=orbit_lqr,
        thruster_direction_body=BASIC_CHEMICAL_BOTTOM_Z.mount.thrust_direction_body,
        config=PredictiveBurnConfig(
            horizon_steps=int(config.lead_steps),
            attitude_tolerance_rad=np.deg2rad(float(config.align_deg)),
            mu_km3_s2=EARTH_MU_KM3_S2,
        ),
    )

    chaser_sensor = NoisyOwnStateSensor(pos_sigma_km=0.001, vel_sigma_km_s=1e-5, rng=rng)
    chief_sensor = NoisyOwnStateSensor(pos_sigma_km=0.001, vel_sigma_km_s=1e-5, rng=rng)
    chaser_ekf = OrbitEKFEstimator(
        mu_km3_s2=EARTH_MU_KM3_S2,
        dt_s=dt_s,
        process_noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]),
        meas_noise_diag=np.array([1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10]),
    )
    chief_ekf = OrbitEKFEstimator(
        mu_km3_s2=EARTH_MU_KM3_S2,
        dt_s=dt_s,
        process_noise_diag=np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10]),
        meas_noise_diag=np.array([1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10]),
    )
    chaser_belief = StateBelief(state=np.hstack((chaser.truth.position_eci_km, chaser.truth.velocity_eci_km_s)), covariance=np.eye(6), last_update_t_s=0.0)
    chief_belief = StateBelief(state=np.hstack((chief.truth.position_eci_km, chief.truth.velocity_eci_km_s)), covariance=np.eye(6), last_update_t_s=0.0)

    rel_dist = np.zeros(steps + 1)
    x_rel = _eci_to_ric_rect(
        np.hstack((chief.truth.position_eci_km, chief.truth.velocity_eci_km_s)),
        np.hstack((chaser.truth.position_eci_km, chaser.truth.velocity_eci_km_s)),
    )
    rel_dist[0] = np.linalg.norm(x_rel[:3])

    dv_used_km_s = 0.0
    burn_count = 0
    gate_checks = 0
    alignment_violations = 0

    for k in tqdm(range(steps), desc=f"Run {seed} steps", unit="step", leave=False, dynamic_ncols=True):
        t_next = chief.truth.t_s + dt_s
        m_chief = chief_sensor.measure(chief.truth, env={}, t_s=t_next)
        m_chaser = chaser_sensor.measure(chaser.truth, env={}, t_s=t_next)
        chief_belief = chief_ekf.update(chief_belief, m_chief, t_s=t_next)
        chaser_belief = chaser_ekf.update(chaser_belief, m_chaser, t_s=t_next)

        decision = scheduler.step(
            chaser_truth=chaser.truth,
            chief_truth=chief.truth,
            chaser_orbit_belief=chaser_belief,
            chief_orbit_belief=chief_belief,
            dt_s=dt_s,
        )

        e = np.array(decision["desired_ric_euler_rad"], dtype=float)
        ric_att_ctrl.set_desired_ric_state(float(e[0]), float(e[1]), float(e[2]), np.zeros(3))
        belief_att = StateBelief(
            state=np.hstack(
                (
                    chaser.truth.position_eci_km,
                    chaser.truth.velocity_eci_km_s,
                    chaser.truth.attitude_quat_bn,
                    chaser.truth.angular_rate_body_rad_s,
                )
            ),
            covariance=np.eye(13),
            last_update_t_s=chaser.truth.t_s,
        )
        c_att = ric_att_ctrl.act(belief_att, t_s=chaser.truth.t_s, budget_ms=1.0)

        planned = np.array(decision["planned_accel_eci_km_s2"], dtype=float)
        gate_event = bool(decision["countdown"] == -1 and np.linalg.norm(planned) > 0.0)
        if gate_event:
            gate_checks += 1
            if float(decision["alignment_angle_rad"]) > np.deg2rad(float(config.align_deg)):
                alignment_violations += 1

        thrust = np.zeros(3)
        if decision["fire"]:
            burn_count += 1
            if config.thrust_mode == "perfect":
                thrust = np.array(decision["thrust_eci_km_s2"], dtype=float)
            else:
                mag = float(np.linalg.norm(np.array(decision["thrust_eci_km_s2"], dtype=float)))
                c_bn = quaternion_to_dcm_bn(chaser.truth.attitude_quat_bn)
                thrust_axis_eci = c_bn.T @ _unit(BASIC_CHEMICAL_BOTTOM_Z.mount.thrust_direction_body)
                thrust = -mag * thrust_axis_eci

        dv_used_km_s += float(np.linalg.norm(thrust) * dt_s)

        cmd_chaser = Command(
            thrust_eci_km_s2=thrust,
            torque_body_nm=c_att.torque_body_nm,
            mode_flags={
                "mode": "pred_mc",
                "current_mass_kg": float(chaser.truth.mass_kg),
            },
        )
        cmd_chief = Command.zero()
        app_chaser = chaser.actuator.apply(cmd_chaser, chaser.limits, dt_s)
        app_chief = chief.actuator.apply(cmd_chief, chief.limits, dt_s)
        chaser.truth = chaser.dynamics.step(chaser.truth, app_chaser, env={}, dt_s=dt_s)
        chief.truth = chief.dynamics.step(chief.truth, app_chief, env={}, dt_s=dt_s)

        x_rel_k = _eci_to_ric_rect(
            np.hstack((chief.truth.position_eci_km, chief.truth.velocity_eci_km_s)),
            np.hstack((chaser.truth.position_eci_km, chaser.truth.velocity_eci_km_s)),
        )
        rel_dist[k + 1] = np.linalg.norm(x_rel_k[:3])

    cap_time = _capture_time_s(rel_dist, dt_s, float(config.capture_radius_km), float(config.capture_hold_s))
    final_miss = float(rel_dist[-1])
    min_miss = float(np.min(rel_dist))

    fail_reasons: list[str] = []
    if cap_time is None:
        fail_reasons.append("capture_not_achieved")
    if final_miss > float(config.pass_final_miss_km):
        fail_reasons.append("final_miss_exceeded")
    if dv_used_km_s > float(config.pass_max_dv_km_s):
        fail_reasons.append("dv_exceeded")
    if alignment_violations > int(config.pass_max_alignment_violations):
        fail_reasons.append("alignment_violations")

    return {
        "seed": int(seed),
        "capture_time_s": cap_time,
        "final_miss_distance_km": final_miss,
        "min_miss_distance_km": min_miss,
        "dv_used_km_s": float(dv_used_km_s),
        "burn_count": int(burn_count),
        "gate_checks": int(gate_checks),
        "alignment_violations": int(alignment_violations),
        "pass": len(fail_reasons) == 0,
        "fail_reasons": fail_reasons,
    }


def _config_to_dict(config: PredictiveRendezvousMCConfig) -> dict:
    d = asdict(config)
    d["init_rel_ric_rect_mean"] = [float(v) for v in np.array(config.init_rel_ric_rect_mean, dtype=float).reshape(6)]
    return d
