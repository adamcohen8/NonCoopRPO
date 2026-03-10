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
from sim.control.orbit import HCWLQRController
from sim.control.orbit.impulsive import AttitudeAgnosticImpulsiveManeuverer
from sim.core.models import Command, StateBelief
from sim.utils.frames import dcm_to_euler_321, ric_curv_to_rect, ric_dcm_ir_from_rv, ric_rect_to_curv
from sim.utils.quaternion import quaternion_to_dcm_bn


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-satellite rendezvous demo using HCW orbital LQR + attitude LQR.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument("--align-deg", type=float, default=5.0)
    args = parser.parse_args()

    dt_s = float(args.dt)
    steps = int(np.ceil(float(args.duration) / dt_s))
    align_tol_rad = np.deg2rad(float(args.align_deg))

    chief = build_sim_object_from_presets(
        object_id="chief",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=False,
        enable_attitude_knowledge=False,
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.zeros(3),
    )
    chaser = build_sim_object_from_presets(
        object_id="chaser",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=False,
        enable_attitude_knowledge=True,
        attitude_quat_bn=np.array([0.98, 0.0, 0.18, 0.0]),
        angular_rate_body_rad_s=np.array([0.0, -0.00, 0.00]),
    )

    # Set chaser initial state from chief with relative rectangular RIC offset.
    x_rel0_rect = np.array([0.0, -8.0, 0.0, 0.000, -0.00, 0.000], dtype=float)
    x_chief = np.hstack((chief.truth.position_eci_km, chief.truth.velocity_eci_km_s))
    x_chaser = _ric_rect_to_eci(x_chief, x_rel0_rect)
    chaser.truth.position_eci_km = x_chaser[:3]
    chaser.truth.velocity_eci_km_s = x_chaser[3:]

    n_rad_s = np.sqrt(398600.4418 / (np.linalg.norm(chief.truth.position_eci_km) ** 3))
    orbit_lqr = HCWLQRController(
        mean_motion_rad_s=n_rad_s,
        max_accel_km_s2=5e-5,
        design_dt_s=dt_s,
        q_weights=np.array([8.66, 8.66, 8.66, 1.33, 1.33, 1.33]) * 1e3,
        r_weights=np.ones(3) * 1.94e13,
    )
    maneuverer = AttitudeAgnosticImpulsiveManeuverer()

    # Initialize chaser attitude close to the initial thrust-vector target.
    x_rel0_curv = ric_rect_to_curv(x_rel0_rect, r0_km=np.linalg.norm(chief.truth.position_eci_km))
    belief0_orbit = StateBelief(
        state=np.hstack((x_rel0_curv, np.hstack((chief.truth.position_eci_km, chief.truth.velocity_eci_km_s)))),
        covariance=np.eye(12),
        last_update_t_s=0.0,
    )
    c0_orb = orbit_lqr.act(belief0_orbit, t_s=0.0, budget_ms=1.0)
    dv0_cmd = np.array(c0_orb.thrust_eci_km_s2, dtype=float) * dt_s
    if np.linalg.norm(dv0_cmd) > 0.0:
        q0_target = maneuverer.required_attitude_for_delta_v(
            truth=chaser.truth,
            delta_v_eci_km_s=dv0_cmd,
            thruster_direction_body=BASIC_CHEMICAL_BOTTOM_Z.mount.thrust_direction_body,
        )
        chaser.truth.attitude_quat_bn = q0_target

    wheel_axes = np.vstack([w.axis_body for w in BASIC_REACTION_WHEEL_TRIAD.wheels])
    wheel_limits = np.array([w.max_torque_nm for w in BASIC_REACTION_WHEEL_TRIAD.wheels], dtype=float)
    att_lqr = SmallAngleLQRController.robust_profile(
        inertia_kg_m2=BASIC_SATELLITE.inertia_kg_m2,
        wheel_axes_body=wheel_axes,
        wheel_torque_limits_nm=wheel_limits,
        design_dt_s=dt_s,
    )
    ric_att_ctrl = RICFrameLQRController(
        lqr=att_lqr,
        desired_ric_euler_rad=np.zeros(3),
        desired_ric_rate_rad_s=np.zeros(3),
    )

    t = np.zeros(steps + 1)
    rel_pos_norm = np.zeros(steps + 1)
    rel_vel_norm = np.zeros(steps + 1)
    align_deg = np.zeros(steps + 1)
    thrust_norm = np.zeros(steps + 1)
    action = np.zeros(steps + 1)  # 0=slew only, 1=fire
    thrust_target_eci = np.zeros(3)
    q_target_bn = chaser.truth.attitude_quat_bn.copy()
    q_target_hist = np.zeros((steps + 1, 4))
    q_target_hist[0, :] = q_target_bn
    q_chaser_hist = np.zeros((steps + 1, 4))
    q_chaser_hist[0, :] = chaser.truth.attitude_quat_bn
    was_firing = False

    x_chief0 = np.hstack((chief.truth.position_eci_km, chief.truth.velocity_eci_km_s))
    x_chaser0 = np.hstack((chaser.truth.position_eci_km, chaser.truth.velocity_eci_km_s))
    x_rel0 = _eci_to_ric_rect(x_chief0, x_chaser0)
    rel_pos_norm[0] = np.linalg.norm(x_rel0[:3])
    rel_vel_norm[0] = np.linalg.norm(x_rel0[3:])

    for k in range(steps):
        # Guidance in relative frame (curvilinear RIC state expected by HCW controller).
        x_chief = np.hstack((chief.truth.position_eci_km, chief.truth.velocity_eci_km_s))
        x_chaser = np.hstack((chaser.truth.position_eci_km, chaser.truth.velocity_eci_km_s))
        x_rel_rect = _eci_to_ric_rect(x_chief, x_chaser)
        x_rel_curv = ric_rect_to_curv(x_rel_rect, r0_km=np.linalg.norm(chief.truth.position_eci_km))
        belief_orbit = StateBelief(
            state=np.hstack((x_rel_curv, x_chief)),
            covariance=np.eye(12),
            last_update_t_s=chief.truth.t_s,
        )
        c_orb = orbit_lqr.act(belief_orbit, t_s=chief.truth.t_s, budget_ms=1.0)
        a_cmd_eci = np.array(c_orb.thrust_eci_km_s2, dtype=float)
        if np.linalg.norm(a_cmd_eci) > 0.0:
            if np.linalg.norm(thrust_target_eci) == 0.0 or was_firing:
                thrust_target_eci = a_cmd_eci.copy()
                q_target_bn = maneuverer.required_attitude_for_delta_v(
                    truth=chaser.truth,
                    delta_v_eci_km_s=thrust_target_eci * dt_s,
                    thruster_direction_body=BASIC_CHEMICAL_BOTTOM_Z.mount.thrust_direction_body,
                )
        dv_cmd = thrust_target_eci * dt_s

        # Attitude target from desired thrust direction.
        c_bn_des = quaternion_to_dcm_bn(q_target_bn)
        c_ir_now = ric_dcm_ir_from_rv(chaser.truth.position_eci_km, chaser.truth.velocity_eci_km_s)
        c_br_des = c_bn_des @ c_ir_now
        e_ric = dcm_to_euler_321(c_br_des)
        ric_att_ctrl.set_desired_ric_state(
            yaw_r_rad=float(e_ric[0]),
            roll_i_rad=float(e_ric[1]),
            pitch_c_rad=float(e_ric[2]),
            w_ric_rad_s=np.zeros(3),
        )
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

        # Thrust only when aligned.
        dv_mag = float(np.linalg.norm(dv_cmd))
        if dv_mag > 0.0:
            c_bn = quaternion_to_dcm_bn(chaser.truth.attitude_quat_bn)
            thrust_axis_eci = c_bn.T @ _unit(BASIC_CHEMICAL_BOTTOM_Z.mount.thrust_direction_body)
            target_axis_eci = -_unit(dv_cmd)
            cosang = float(np.clip(np.dot(thrust_axis_eci, target_axis_eci), -1.0, 1.0))
            ang = float(np.arccos(cosang))
        else:
            ang = 0.0
        can_fire = bool(dv_mag > 0.0 and ang <= align_tol_rad)
        applied_thrust = thrust_target_eci if can_fire else np.zeros(3)
        was_firing = can_fire

        cmd_chaser = Command(thrust_eci_km_s2=applied_thrust, torque_body_nm=c_att.torque_body_nm, mode_flags={"mode": "rvz"})
        cmd_chief = Command.zero()
        app_chaser = chaser.actuator.apply(cmd_chaser, chaser.limits, dt_s)
        app_chief = chief.actuator.apply(cmd_chief, chief.limits, dt_s)

        chaser.truth = chaser.dynamics.step(chaser.truth, app_chaser, env={}, dt_s=dt_s)
        chief.truth = chief.dynamics.step(chief.truth, app_chief, env={}, dt_s=dt_s)

        t[k + 1] = chaser.truth.t_s
        x_chief_n = np.hstack((chief.truth.position_eci_km, chief.truth.velocity_eci_km_s))
        x_chaser_n = np.hstack((chaser.truth.position_eci_km, chaser.truth.velocity_eci_km_s))
        x_rel_n = _eci_to_ric_rect(x_chief_n, x_chaser_n)
        rel_pos_norm[k + 1] = np.linalg.norm(x_rel_n[:3])
        rel_vel_norm[k + 1] = np.linalg.norm(x_rel_n[3:])
        align_deg[k + 1] = np.rad2deg(ang)
        thrust_norm[k + 1] = np.linalg.norm(applied_thrust)
        action[k + 1] = 1.0 if can_fire else 0.0
        q_target_hist[k + 1, :] = q_target_bn
        q_chaser_hist[k + 1, :] = chaser.truth.attitude_quat_bn

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, rel_pos_norm)
    axes[0].set_ylabel("||r_rel|| km")
    axes[0].set_title("Rendezvous: Relative Position Norm")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, rel_vel_norm)
    axes[1].set_ylabel("||v_rel|| km/s")
    axes[1].set_title("Relative Velocity Norm")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, align_deg, label="align angle")
    axes[2].axhline(np.rad2deg(align_tol_rad), linestyle="--", label="tol")
    axes[2].set_ylabel("deg")
    axes[2].set_title("Thrust Alignment")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")
    axes[3].step(t, action, where="post")
    axes[3].set_yticks([0, 1], labels=["slew", "fire"])
    axes[3].set_ylabel("Action")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Fire/Slew Timeline")
    axes[3].grid(True, alpha=0.3)
    fig.tight_layout()

    if args.plot_mode in ("save", "both"):
        outdir = REPO_ROOT / "outputs" / "rendezvous_hcw_att_lqr_demo"
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "rendezvous_metrics.png", dpi=150)
    if args.plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    fig_q, axes_q = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    labels = ["q0", "q1", "q2", "q3"]
    for i in range(4):
        axes_q[i].plot(t, q_target_hist[:, i], label="target")
        axes_q[i].plot(t, q_chaser_hist[:, i], "--", label="chaser")
        axes_q[i].set_ylabel(labels[i])
        axes_q[i].grid(True, alpha=0.3)
        axes_q[i].legend(loc="best")
    axes_q[0].set_title("Thrust-Vector Target Quaternion (q_target_bn)")
    axes_q[-1].set_xlabel("Time (s)")
    fig_q.tight_layout()

    if args.plot_mode in ("save", "both"):
        fig_q.savefig(outdir / "thrust_vector_quaternion.png", dpi=150)
    if args.plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_q)

    print("Rendezvous demo complete.")
    print("Initial ||r_rel|| km:", float(rel_pos_norm[0]))
    print("Final ||r_rel|| km:", float(rel_pos_norm[-1]))
    print("Initial ||v_rel|| km/s:", float(rel_vel_norm[0]))
    print("Final ||v_rel|| km/s:", float(rel_vel_norm[-1]))
    print("Fire duty fraction:", float(np.mean(action)))
