from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from presets import BASIC_CHEMICAL_BOTTOM_Z, BASIC_REACTION_WHEEL_TRIAD, BASIC_SATELLITE, build_sim_object_from_presets
from sim.control.attitude import RICFrameLQRController, SmallAngleLQRController
from sim.control.orbit import HCWLQRController, PredictiveBurnConfig, PredictiveBurnScheduler
from sim.core.models import Command, StateBelief
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.sensors.noisy_own_state import NoisyOwnStateSensor
from sim.utils.quaternion import quaternion_to_dcm_bn
from sim.actuators.attitude import ReactionWheelLimits


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


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


def _quat_error_deg(q_target: np.ndarray, q_current: np.ndarray) -> float:
    qt = np.array(q_target, dtype=float)
    qc = np.array(q_current, dtype=float)
    qt /= max(np.linalg.norm(qt), 1e-12)
    qc /= max(np.linalg.norm(qc), 1e-12)
    qt_conj = np.array([qt[0], -qt[1], -qt[2], -qt[3]])
    qe = np.array(
        [
            qt_conj[0] * qc[0] - qt_conj[1] * qc[1] - qt_conj[2] * qc[2] - qt_conj[3] * qc[3],
            qt_conj[0] * qc[1] + qt_conj[1] * qc[0] + qt_conj[2] * qc[3] - qt_conj[3] * qc[2],
            qt_conj[0] * qc[2] - qt_conj[1] * qc[3] + qt_conj[2] * qc[0] + qt_conj[3] * qc[1],
            qt_conj[0] * qc[3] + qt_conj[1] * qc[2] - qt_conj[2] * qc[1] + qt_conj[3] * qc[0],
        ]
    )
    if qe[0] < 0.0:
        qe *= -1.0
    return float(np.rad2deg(2.0 * np.arccos(np.clip(qe[0], -1.0, 1.0))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predictive EKF coupling demo: predict future state, plan burn, wait horizon, burn if aligned."
    )
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--orbit-dt", type=float, default=None, help="Optional orbital integration substep (s).")
    parser.add_argument("--attitude-dt", type=float, default=None, help="Optional attitude integration substep (s).")
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--lead-time-s", type=float, default=5.0, help="Lead time (seconds) between attitude command and burn check.")
    parser.add_argument("--horizon-steps", type=int, default=None, help="Deprecated alias for --lead-steps.")
    parser.add_argument("--lead-steps", type=int, default=None, help="Deprecated: use --lead-time-s.")
    parser.add_argument("--align-deg", type=float, default=10.0)
    parser.add_argument("--wheel-scale", type=float, default=10.0, help="Scale factor for demo reaction-wheel torque/momentum.")
    parser.add_argument(
        "--thrust-mode",
        choices=["attitude", "perfect"],
        default="attitude",
        help="`perfect`: apply planned inertial thrust vector directly; `attitude`: apply thrust direction from current attitude.",
    )
    args = parser.parse_args()
    if args.horizon_steps is not None:
        lead_steps = int(args.horizon_steps)
    elif args.lead_steps is not None:
        lead_steps = int(args.lead_steps)
    else:
        lead_steps = int(np.ceil(float(args.lead_time_s) / float(args.dt)))
    if lead_steps < 0:
        raise ValueError("lead time must be non-negative.")

    dt_s = float(args.dt)
    steps = int(np.ceil(float(args.duration) / dt_s))

    chief = build_sim_object_from_presets(
        object_id="chief_pred",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=False,
        enable_attitude_knowledge=False,
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.zeros(3),
        orbit_substep_s=args.orbit_dt,
        attitude_substep_s=args.attitude_dt,
    )
    chaser = build_sim_object_from_presets(
        object_id="chaser_pred",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=False,
        enable_attitude_knowledge=True,
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.array([0.01, -0.005, 0.004]),
        orbit_substep_s=args.orbit_dt,
        attitude_substep_s=args.attitude_dt,
    )

    x_rel0_rect = np.array([2.0, -8.0, 1.2, 0.0008, -0.0012, 0.0004], dtype=float)
    x_chief = np.hstack((chief.truth.position_eci_km, chief.truth.velocity_eci_km_s))
    x_chaser = _ric_rect_to_eci(x_chief, x_rel0_rect)
    chaser.truth.position_eci_km = x_chaser[:3]
    chaser.truth.velocity_eci_km_s = x_chaser[3:]

    n_rad_s = np.sqrt(EARTH_MU_KM3_S2 / (np.linalg.norm(chief.truth.position_eci_km) ** 3))
    orbit_lqr = HCWLQRController(
        mean_motion_rad_s=n_rad_s,
        max_accel_km_s2=5e-5,
        design_dt_s=dt_s,
        q_weights=np.array([8.66, 8.66, 8.66, 1.33, 1.33, 1.33]) * 1e3,
        r_weights=np.ones(3) * 1.94e13,
    )

    wheel_axes = np.vstack([w.axis_body for w in BASIC_REACTION_WHEEL_TRIAD.wheels])
    if args.wheel_scale <= 0.0:
        raise ValueError("--wheel-scale must be positive.")
    wheel_limits = np.array([w.max_torque_nm for w in BASIC_REACTION_WHEEL_TRIAD.wheels], dtype=float) * float(
        args.wheel_scale
    )
    att_lqr = SmallAngleLQRController.robust_profile(
        inertia_kg_m2=BASIC_SATELLITE.inertia_kg_m2,
        wheel_axes_body=wheel_axes,
        wheel_torque_limits_nm=wheel_limits,
        design_dt_s=dt_s,
    )
    ric_att_ctrl = RICFrameLQRController(lqr=att_lqr)

    # Increase simulated wheel hardware for this demo as well (torque + momentum).
    if hasattr(chaser.actuator, "attitude") and hasattr(chaser.actuator.attitude, "reaction_wheels"):
        rw = chaser.actuator.attitude.reaction_wheels
        if rw is not None:
            chaser.actuator.attitude.reaction_wheels = ReactionWheelLimits(
                max_torque_nm=rw.max_torque_nm * float(args.wheel_scale),
                max_momentum_nms=rw.max_momentum_nms * float(args.wheel_scale),
            )

    sched = PredictiveBurnScheduler(
        orbit_lqr=orbit_lqr,
        thruster_direction_body=BASIC_CHEMICAL_BOTTOM_Z.mount.thrust_direction_body,
        config=PredictiveBurnConfig(
            horizon_steps=lead_steps,
            attitude_tolerance_rad=np.deg2rad(float(args.align_deg)),
            mu_km3_s2=EARTH_MU_KM3_S2,
        ),
    )

    rng = np.random.default_rng(3)
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

    t = np.zeros(steps + 1)
    rel_pos = np.zeros(steps + 1)
    rel_vel = np.zeros(steps + 1)
    rel_pos_comp = np.zeros((steps + 1, 3))
    rel_vel_comp = np.zeros((steps + 1, 3))
    fire_hist = np.zeros(steps + 1)
    align_deg_hist = np.zeros(steps + 1)
    planned_thrust_norm = np.zeros(steps + 1)
    applied_thrust_norm = np.zeros(steps + 1)
    q_target_hist = np.zeros((steps + 1, 4))
    q_chaser_hist = np.zeros((steps + 1, 4))
    q_err_deg_hist = np.zeros(steps + 1)

    x_rel_init = _eci_to_ric_rect(
        np.hstack((chief.truth.position_eci_km, chief.truth.velocity_eci_km_s)),
        np.hstack((chaser.truth.position_eci_km, chaser.truth.velocity_eci_km_s)),
    )
    rel_pos[0] = np.linalg.norm(x_rel_init[:3])
    rel_vel[0] = np.linalg.norm(x_rel_init[3:])
    rel_pos_comp[0, :] = x_rel_init[:3]
    rel_vel_comp[0, :] = x_rel_init[3:]
    q_target_hist[0, :] = chaser.truth.attitude_quat_bn
    q_chaser_hist[0, :] = chaser.truth.attitude_quat_bn
    q_err_deg_hist[0] = 0.0

    for k in range(steps):
        m_chief = chief_sensor.measure(chief.truth, env={}, t_s=chief.truth.t_s + dt_s)
        m_chaser = chaser_sensor.measure(chaser.truth, env={}, t_s=chaser.truth.t_s + dt_s)
        chief_belief = chief_ekf.update(chief_belief, m_chief, t_s=chief.truth.t_s + dt_s)
        chaser_belief = chaser_ekf.update(chaser_belief, m_chaser, t_s=chaser.truth.t_s + dt_s)

        decision = sched.step(
            chaser_truth=chaser.truth,
            chief_truth=chief.truth,
            chaser_orbit_belief=chaser_belief,
            chief_orbit_belief=chief_belief,
            dt_s=dt_s,
        )
        e = decision["desired_ric_euler_rad"]
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
        thrust = np.zeros(3)
        if decision["fire"]:
            planned = np.array(decision["thrust_eci_km_s2"], dtype=float)
            if args.thrust_mode == "perfect":
                thrust = planned
            else:
                # Attitude-dependent mode: keep planned magnitude, apply along current thruster axis direction.
                # Sign convention follows planner alignment check: thrust axis should align with -planned direction.
                mag = float(np.linalg.norm(planned))
                c_bn = quaternion_to_dcm_bn(chaser.truth.attitude_quat_bn)
                thrust_axis_eci = c_bn.T @ _unit(BASIC_CHEMICAL_BOTTOM_Z.mount.thrust_direction_body)
                thrust = -mag * thrust_axis_eci
        planned_now = np.array(decision["planned_accel_eci_km_s2"], dtype=float)

        cmd_chaser = Command(thrust_eci_km_s2=thrust, torque_body_nm=c_att.torque_body_nm, mode_flags={"mode": "pred_ekf"})
        cmd_chief = Command.zero()
        app_chaser = chaser.actuator.apply(cmd_chaser, chaser.limits, dt_s)
        app_chief = chief.actuator.apply(cmd_chief, chief.limits, dt_s)
        chaser.truth = chaser.dynamics.step(chaser.truth, app_chaser, env={}, dt_s=dt_s)
        chief.truth = chief.dynamics.step(chief.truth, app_chief, env={}, dt_s=dt_s)

        x_rel = _eci_to_ric_rect(
            np.hstack((chief.truth.position_eci_km, chief.truth.velocity_eci_km_s)),
            np.hstack((chaser.truth.position_eci_km, chaser.truth.velocity_eci_km_s)),
        )
        t[k + 1] = chaser.truth.t_s
        rel_pos[k + 1] = np.linalg.norm(x_rel[:3])
        rel_vel[k + 1] = np.linalg.norm(x_rel[3:])
        rel_pos_comp[k + 1, :] = x_rel[:3]
        rel_vel_comp[k + 1, :] = x_rel[3:]
        fire_hist[k + 1] = 1.0 if decision["fire"] else 0.0
        align_deg_hist[k + 1] = np.rad2deg(decision["alignment_angle_rad"])
        planned_thrust_norm[k + 1] = np.linalg.norm(planned_now)
        applied_thrust_norm[k + 1] = np.linalg.norm(thrust)
        q_target_hist[k + 1, :] = np.array(decision["planned_q_target_bn"], dtype=float)
        q_chaser_hist[k + 1, :] = chaser.truth.attitude_quat_bn
        q_err_deg_hist[k + 1] = _quat_error_deg(q_target_hist[k + 1, :], q_chaser_hist[k + 1, :])

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, rel_pos)
    axes[0].set_ylabel("||r_rel|| km")
    axes[0].set_title("Predictive EKF Coupling: Relative Position")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, rel_vel)
    axes[1].set_ylabel("||v_rel|| km/s")
    axes[1].set_title("Relative Velocity")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, planned_thrust_norm, label="planned")
    axes[2].plot(t, applied_thrust_norm, label="applied")
    axes[2].set_ylabel("||a|| km/s^2")
    axes[2].set_title("Planned vs Applied Thrust Magnitude")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(t, align_deg_hist, label="alignment")
    axes[3].axhline(float(args.align_deg), linestyle="--", label="tolerance")
    burn_idx = np.where(fire_hist > 0.5)[0]
    if burn_idx.size > 0:
        axes[3].plot(t[burn_idx], align_deg_hist[burn_idx], "o", markersize=2.5, label="burn samples")
    axes[3].set_ylabel("deg")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Alignment vs Tolerance (Burns Marked)")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)
    fig.tight_layout()

    if args.plot_mode in ("save", "both"):
        outdir = REPO_ROOT / "outputs" / "rendezvous_predictive_ekf_demo"
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "predictive_metrics.png", dpi=150)
    if args.plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    fig_comp, axes_comp = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    pos_labels = ["R (km)", "I (km)", "C (km)"]
    vel_labels = ["dR (km/s)", "dI (km/s)", "dC (km/s)"]
    for i in range(3):
        axes_comp[i, 0].plot(t, rel_pos_comp[:, i])
        axes_comp[i, 0].set_ylabel(pos_labels[i])
        axes_comp[i, 0].grid(True, alpha=0.3)
        axes_comp[i, 1].plot(t, rel_vel_comp[:, i])
        axes_comp[i, 1].set_ylabel(vel_labels[i])
        axes_comp[i, 1].grid(True, alpha=0.3)
    axes_comp[0, 0].set_title("Relative Position Components (RIC)")
    axes_comp[0, 1].set_title("Relative Velocity Components (RIC)")
    axes_comp[2, 0].set_xlabel("Time (s)")
    axes_comp[2, 1].set_xlabel("Time (s)")
    fig_comp.tight_layout()

    if args.plot_mode in ("save", "both"):
        fig_comp.savefig(outdir / "relative_components.png", dpi=150)
    if args.plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_comp)

    fig_q, axes_q = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    q_labels = ["q0", "q1", "q2", "q3"]
    for i in range(4):
        axes_q[i].plot(t, q_target_hist[:, i], label="target thrust quat")
        axes_q[i].plot(t, q_chaser_hist[:, i], "--", label="chaser quat")
        axes_q[i].set_ylabel(q_labels[i])
        axes_q[i].grid(True, alpha=0.3)
        axes_q[i].legend(loc="best")
    axes_q[0].set_title("Quaternion: Thrust-Vector Target vs Chaser Attitude")
    axes_q[-1].set_xlabel("Time (s)")
    fig_q.tight_layout()

    if args.plot_mode in ("save", "both"):
        fig_q.savefig(outdir / "quaternion_target_vs_chaser.png", dpi=150)
    if args.plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_q)

    q_component_error = q_target_hist - q_chaser_hist
    fig_qc, axes_qc = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i in range(4):
        axes_qc[i].plot(t, q_component_error[:, i])
        axes_qc[i].set_ylabel(f"{q_labels[i]} err")
        axes_qc[i].grid(True, alpha=0.3)
    axes_qc[0].set_title("Quaternion Component Error: Target - Chaser")
    axes_qc[-1].set_xlabel("Time (s)")
    fig_qc.tight_layout()

    if args.plot_mode in ("save", "both"):
        fig_qc.savefig(outdir / "quaternion_component_error.png", dpi=150)
    if args.plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_qc)

    fig_qe, ax_qe = plt.subplots(1, 1, figsize=(10, 4.5))
    ax_qe.plot(t, q_err_deg_hist)
    ax_qe.set_title("Quaternion Error: Chaser vs Thrust-Target")
    ax_qe.set_ylabel("Angle Error (deg)")
    ax_qe.set_xlabel("Time (s)")
    ax_qe.grid(True, alpha=0.3)
    fig_qe.tight_layout()

    if args.plot_mode in ("save", "both"):
        fig_qe.savefig(outdir / "quaternion_error_deg.png", dpi=150)
    if args.plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_qe)

    print("Predictive EKF rendezvous demo complete.")
    print("Initial ||r_rel|| km:", float(rel_pos[0]))
    print("Final ||r_rel|| km:", float(rel_pos[-1]))
    print("Initial ||v_rel|| km/s:", float(rel_vel[0]))
    print("Final ||v_rel|| km/s:", float(rel_vel[-1]))
    print("Burn fraction:", float(np.mean(fire_hist)))
    print("Thrust mode:", args.thrust_mode)
    print("Wheel scale:", float(args.wheel_scale))
    print("Lead steps:", int(lead_steps))
    print("Lead time (s):", float(lead_steps) * dt_s)
    print("Orbit substep (s):", args.orbit_dt if args.orbit_dt is not None else dt_s)
    print("Attitude substep (s):", args.attitude_dt if args.attitude_dt is not None else dt_s)
