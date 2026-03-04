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
from sim.control.attitude import SmallAngleLQRController
from sim.control.orbit import IntegratedManeuverCommand, OrbitalAttitudeManeuverCoordinator
from sim.core.models import Command, StateBelief
from sim.utils.quaternion import normalize_quaternion


def _quat_error_angle_deg(q_des_bn: np.ndarray, q_bn: np.ndarray) -> float:
    qd = normalize_quaternion(np.array(q_des_bn, dtype=float))
    q = normalize_quaternion(np.array(q_bn, dtype=float))
    qd_conj = np.array([qd[0], -qd[1], -qd[2], -qd[3]])
    qe = np.array(
        [
            qd_conj[0] * q[0] - qd_conj[1] * q[1] - qd_conj[2] * q[2] - qd_conj[3] * q[3],
            qd_conj[0] * q[1] + qd_conj[1] * q[0] + qd_conj[2] * q[3] - qd_conj[3] * q[2],
            qd_conj[0] * q[2] - qd_conj[1] * q[3] + qd_conj[2] * q[0] + qd_conj[3] * q[1],
            qd_conj[0] * q[3] + qd_conj[1] * q[2] - qd_conj[2] * q[1] + qd_conj[3] * q[0],
        ]
    )
    if qe[0] < 0.0:
        qe *= -1.0
    qe0 = float(np.clip(qe[0], -1.0, 1.0))
    return np.rad2deg(2.0 * np.arccos(qe0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated orbital-attitude closed-loop maneuver demo.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    args = parser.parse_args()

    dt_s = 2.0
    steps = 400
    sat = build_sim_object_from_presets(
        object_id="sat_integrated_demo",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        thruster=BASIC_CHEMICAL_BOTTOM_Z,
        rw_assembly=BASIC_REACTION_WHEEL_TRIAD,
        controller_budget_ms=2.0,
        enable_disturbances=False,
        enable_attitude_knowledge=True,
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.zeros(3),
    )

    wheel_axes = np.vstack([w.axis_body for w in BASIC_REACTION_WHEEL_TRIAD.wheels])
    wheel_limits = np.array([w.max_torque_nm for w in BASIC_REACTION_WHEEL_TRIAD.wheels], dtype=float)
    attitude_ctrl = SmallAngleLQRController(
        inertia_kg_m2=BASIC_SATELLITE.inertia_kg_m2,
        wheel_axes_body=wheel_axes,
        wheel_torque_limits_nm=wheel_limits,
        q_weights=np.array([20.0, 20.0, 20.0, 1.0, 1.0, 1.0]),
        r_weights=np.array([1.0, 1.0, 1.0]),
        design_dt_s=dt_s,
    )

    coordinator = OrbitalAttitudeManeuverCoordinator()
    available_dv_km_s = 0.05
    min_thrust_n = BASIC_CHEMICAL_BOTTOM_Z.min_impulse_bit_n_s / dt_s

    t_hist = []
    req_dv_hist = []
    app_dv_hist = []
    align_deg_hist = []
    action_hist = []
    mode_hist = []
    quat_err_deg_hist = []
    wheel_cmd_hist = []

    action_to_num = {"hold": 0.0, "slew": 1.0, "fire": 2.0}

    phase = 0
    phase1_count = 0
    target_quat = sat.truth.attitude_quat_bn.copy()
    dv_dir = np.array([0.1, 0.0, -1.0], dtype=float)
    dv_dir /= np.linalg.norm(dv_dir)

    for _ in range(steps):
        truth = sat.truth
        if phase == 0:
            dv_cmd = 0.004 * dv_dir  # intentionally slightly misaligned (> tolerance), so slew first
        elif phase == 1:
            dv_cmd = 1e-6 * dv_dir  # below min-thrust; slew but no fire
        else:
            dv_cmd = 0.001 * dv_dir  # same direction, above minimum; fire

        man_cmd = IntegratedManeuverCommand(
            delta_v_eci_km_s=dv_cmd,
            available_delta_v_km_s=available_dv_km_s,
            strategy="thrust_limited",
            max_thrust_n=BASIC_CHEMICAL_BOTTOM_Z.max_thrust_n,
            dt_s=dt_s,
            min_thrust_n=min_thrust_n,
            require_attitude_alignment=True,
            thruster_position_body_m=BASIC_CHEMICAL_BOTTOM_Z.mount.position_body_m,
            thruster_direction_body=BASIC_CHEMICAL_BOTTOM_Z.mount.thrust_direction_body,
            alignment_tolerance_rad=np.deg2rad(5.0),
        )
        truth_after_orbit, decision = coordinator.execute(truth, man_cmd)

        if decision.required_attitude_quat_bn is not None:
            target_quat = decision.required_attitude_quat_bn.copy()
            attitude_ctrl.set_target(target_quat, np.zeros(3))

        belief_vec = np.hstack(
            (
                truth_after_orbit.position_eci_km,
                truth_after_orbit.velocity_eci_km_s,
                truth_after_orbit.attitude_quat_bn,
                truth_after_orbit.angular_rate_body_rad_s,
            )
        )
        belief = StateBelief(state=belief_vec, covariance=np.eye(13), last_update_t_s=truth_after_orbit.t_s)
        att_cmd = attitude_ctrl.act(belief, t_s=truth_after_orbit.t_s, budget_ms=2.0)

        cmd = Command(
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=att_cmd.torque_body_nm,
            mode_flags={"mode": f"integrated_{decision.action}"},
        )
        applied = sat.actuator.apply(cmd, sat.limits, dt_s)
        sat.truth = sat.dynamics.step(truth_after_orbit, applied, env={}, dt_s=dt_s)
        available_dv_km_s = decision.remaining_delta_v_km_s

        t_hist.append(sat.truth.t_s)
        req_dv_hist.append(float(np.linalg.norm(dv_cmd)))
        app_dv_hist.append(float(decision.applied_delta_v_km_s))
        align_deg_hist.append(
            np.nan if decision.thrust_result.alignment_angle_rad is None else np.rad2deg(decision.thrust_result.alignment_angle_rad)
        )
        action_hist.append(action_to_num[decision.action])
        mode_hist.append(decision.action)
        quat_err_deg_hist.append(_quat_error_angle_deg(target_quat, sat.truth.attitude_quat_bn))
        wheel_cmd = np.array(att_cmd.mode_flags.get("wheel_torque_cmd_nm", [0.0, 0.0, 0.0]), dtype=float)
        wheel_cmd_hist.append(wheel_cmd)

        if phase == 0 and decision.alignment_ok:
            phase = 1
            phase1_count = 0
        elif phase == 1:
            phase1_count += 1
            if phase1_count >= 15:
                phase = 2
        elif phase == 2 and decision.action == "fire":
            # Let it run a little after first burn to visualize settling.
            if len(t_hist) > 250:
                break

    t = np.array(t_hist)
    req_dv = np.array(req_dv_hist)
    app_dv = np.array(app_dv_hist)
    align_deg = np.array(align_deg_hist)
    action = np.array(action_hist)
    quat_err_deg = np.array(quat_err_deg_hist)
    wheel_cmd = np.array(wheel_cmd_hist)

    fig1, axes1 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes1[0].plot(t, req_dv, label="Requested |dv|")
    axes1[0].plot(t, app_dv, label="Applied |dv|")
    axes1[0].set_ylabel("km/s")
    axes1[0].set_title("Delta-V Request vs Execution")
    axes1[0].grid(True, alpha=0.3)
    axes1[0].legend(loc="best")

    axes1[1].plot(t, align_deg, label="Alignment angle")
    axes1[1].plot(t, np.full_like(t, 5.0), "--", label="Tolerance")
    axes1[1].set_ylabel("deg")
    axes1[1].set_title("Alignment Angle")
    axes1[1].grid(True, alpha=0.3)
    axes1[1].legend(loc="best")

    axes1[2].step(t, action, where="post")
    axes1[2].set_yticks([0, 1, 2], labels=["hold", "slew", "fire"])
    axes1[2].set_ylabel("Mode")
    axes1[2].set_xlabel("Time (s)")
    axes1[2].set_title("Coordinator Action Timeline")
    axes1[2].grid(True, alpha=0.3)
    fig1.tight_layout()

    fig2, axes2 = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes2[0].plot(t, quat_err_deg)
    axes2[0].set_ylabel("deg")
    axes2[0].set_title("Attitude Tracking Error")
    axes2[0].grid(True, alpha=0.3)
    for i in range(3):
        axes2[i + 1].plot(t, wheel_cmd[:, i])
        axes2[i + 1].set_ylabel(f"Wheel {i+1} (Nm)")
        axes2[i + 1].grid(True, alpha=0.3)
    axes2[-1].set_xlabel("Time (s)")
    axes2[1].set_title("Reaction Wheel Commanded Torques")
    fig2.tight_layout()

    if args.plot_mode in ("save", "both"):
        outdir = REPO_ROOT / "outputs" / "integrated_maneuver_closed_loop"
        outdir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(outdir / "maneuver_timeline.png", dpi=150)
        fig2.savefig(outdir / "attitude_tracking.png", dpi=150)

    if args.plot_mode in ("interactive", "both"):
        plt.show()

    plt.close(fig1)
    plt.close(fig2)

    mode_counts = {m: mode_hist.count(m) for m in ("hold", "slew", "fire")}
    print("Mode counts:", mode_counts)
    print("Final available delta-V (km/s):", available_dv_km_s)
    print("Final attitude error (deg):", quat_err_deg[-1] if quat_err_deg.size > 0 else np.nan)


if __name__ == "__main__":
    main()
