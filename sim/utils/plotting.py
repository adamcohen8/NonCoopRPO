from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from sim.utils.frames import dcm_to_euler_321, ric_dcm_ir_from_rv
from sim.utils.quaternion import quaternion_to_dcm_bn


PlotMode = Literal["interactive", "save", "both"]


def plot_orbit_eci(truth_hist: np.ndarray, mode: PlotMode = "interactive", out_path: str | None = None) -> None:
    r = truth_hist[:, :3]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(r[:, 0], r[:, 1], r[:, 2], linewidth=1.5)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title("One-Orbit ECI Trajectory")
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def plot_attitude_tumble(
    t_s: np.ndarray, truth_hist: np.ndarray, mode: PlotMode = "interactive", out_path: str | None = None
) -> None:
    q = truth_hist[:, 6:10]
    w = truth_hist[:, 10:13]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t_s, q[:, 0], label="q0")
    axes[0].plot(t_s, q[:, 1], label="q1")
    axes[0].plot(t_s, q[:, 2], label="q2")
    axes[0].plot(t_s, q[:, 3], label="q3")
    axes[0].set_ylabel("Quaternion")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_s, w[:, 0], label="wx")
    axes[1].plot(t_s, w[:, 1], label="wy")
    axes[1].plot(t_s, w[:, 2], label="wz")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular rate (rad/s)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def plot_attitude_ric(
    t_s: np.ndarray, truth_hist: np.ndarray, mode: PlotMode = "interactive", out_path: str | None = None
) -> None:
    # Internal 3-2-1 extraction on RIC axes [R, I, C]:
    #   roll_321 -> about R, pitch_321 -> about I, yaw_321 -> about C.
    # User convention for this project:
    #   yaw -> about R, roll -> about I, pitch -> about C.
    euler_321_deg = np.zeros((truth_hist.shape[0], 3))
    for k in range(truth_hist.shape[0]):
        r = truth_hist[k, :3]
        v = truth_hist[k, 3:6]
        q_bn = truth_hist[k, 6:10]
        c_bn = quaternion_to_dcm_bn(q_bn)
        c_ir = ric_dcm_ir_from_rv(r, v)
        c_br = c_bn @ c_ir
        euler_321_deg[k, :] = np.rad2deg(dcm_to_euler_321(c_br))

    yaw_about_r_deg = euler_321_deg[:, 0]
    roll_about_i_deg = euler_321_deg[:, 1]
    pitch_about_c_deg = euler_321_deg[:, 2]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t_s, yaw_about_r_deg, linewidth=1.4)
    axes[0].set_ylabel("yaw about R (deg)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_s, roll_about_i_deg, linewidth=1.4)
    axes[1].set_ylabel("roll about I (deg)")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t_s, pitch_about_c_deg, linewidth=1.4)
    axes[2].set_ylabel("pitch about C (deg)")
    axes[2].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Body Attitude Relative to RIC Frame (Project Axis Convention)")

    fig.tight_layout()
    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)


def plot_angular_rates(
    t_s: np.ndarray, truth_hist: np.ndarray, mode: PlotMode = "interactive", out_path: str | None = None
) -> None:
    w = truth_hist[:, 10:13]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["wx (rad/s)", "wy (rad/s)", "wz (rad/s)"]
    for i, ax in enumerate(axes):
        ax.plot(t_s, w[:, i], linewidth=1.4)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Body Angular Rates Over Time")

    fig.tight_layout()
    if mode in ("save", "both"):
        if out_path is None:
            raise ValueError("out_path is required when mode is 'save' or 'both'")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)
