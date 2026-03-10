from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.cfs_sil.sim_loop_adapter import CfsSilLoopConfig, run_single_satellite_cfs_sil_loop


def run_demo(
    plot_mode: str = "interactive",
    dt_s: float = 0.2,
    duration_s: float = 180.0,
    bind_host: str = "127.0.0.1",
    bind_port: int = 50100,
    cfs_host: str = "127.0.0.1",
    cfs_port: int = 50101,
    atmosphere_model: str = "ussa1976",
) -> dict[str, str]:
    cfg = CfsSilLoopConfig(
        dt_s=dt_s,
        duration_s=duration_s,
        bind_host=bind_host,
        bind_port=bind_port,
        cfs_host=cfs_host,
        cfs_port=cfs_port,
        atmosphere_model=atmosphere_model,
        enable_disturbances=True,
    )
    out = run_single_satellite_cfs_sil_loop(cfg)
    t = out.time_s
    pos = out.position_eci_km
    vel = out.velocity_eci_km_s
    thrust = out.commanded_thrust_eci_km_s2
    torque = out.commanded_torque_body_nm
    mode = out.bridge_cmd_mode

    outdir = REPO_ROOT / "outputs" / "cfs_sil_single_sat"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    ax[0, 0].plot(t, np.linalg.norm(pos, axis=1))
    ax[0, 0].set_title("|r| (km)")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 1].plot(t, np.linalg.norm(vel, axis=1))
    ax[0, 1].set_title("|v| (km/s)")
    ax[0, 1].grid(True, alpha=0.3)

    ax[1, 0].plot(t, thrust[:, 0], label="tx")
    ax[1, 0].plot(t, thrust[:, 1], label="ty")
    ax[1, 0].plot(t, thrust[:, 2], label="tz")
    ax[1, 0].set_title("cFS Commanded Thrust (km/s^2)")
    ax[1, 0].legend(loc="best")
    ax[1, 0].grid(True, alpha=0.3)

    ax[1, 1].plot(t, torque[:, 0], label="mx")
    ax[1, 1].plot(t, torque[:, 1], label="my")
    ax[1, 1].plot(t, torque[:, 2], label="mz")
    ax[1, 1].set_title("cFS Commanded Torque (N m)")
    ax[1, 1].legend(loc="best")
    ax[1, 1].grid(True, alpha=0.3)

    ax[2, 0].step(t, mode, where="post")
    ax[2, 0].set_title("Bridge Command Mode")
    ax[2, 0].grid(True, alpha=0.3)
    ax[2, 0].set_xlabel("Time (s)")

    ax[2, 1].axis("off")
    ax[2, 1].text(
        0.0,
        0.95,
        "Run cFS app + UDP endpoint on configured ports.\n"
        f"Bind: {bind_host}:{bind_port}\n"
        f"cFS: {cfs_host}:{cfs_port}\n"
        f"Atmosphere: {atmosphere_model}",
        va="top",
    )
    fig.tight_layout()

    p = outdir / "cfs_sil_single_sat_profiles.png"
    if plot_mode in ("save", "both"):
        fig.savefig(p, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    return {
        "duration_s": f"{duration_s:.2f}",
        "steps": str(t.size),
        "plot": str(p) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-satellite cFS SIL bridge loop demo.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--bind-host", type=str, default="127.0.0.1")
    parser.add_argument("--bind-port", type=int, default=50100)
    parser.add_argument("--cfs-host", type=str, default="127.0.0.1")
    parser.add_argument("--cfs-port", type=int, default=50101)
    parser.add_argument("--atmosphere-model", choices=["exponential", "ussa1976", "nrlmsise00"], default="ussa1976")
    args = parser.parse_args()

    result = run_demo(
        plot_mode=args.plot_mode,
        dt_s=float(args.dt),
        duration_s=float(args.duration),
        bind_host=args.bind_host,
        bind_port=int(args.bind_port),
        cfs_host=args.cfs_host,
        cfs_port=int(args.cfs_port),
        atmosphere_model=args.atmosphere_model,
    )
    print("cFS SIL loop demo outputs:")
    for k, v in result.items():
        if v:
            print(f"  {k}: {v}")
