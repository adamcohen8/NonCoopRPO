import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Allow running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noncoop_rpo import MasterSimLog


def _interp_vec(t_src: np.ndarray, x_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    out = np.full((len(t_query), 3), np.nan, dtype=float)
    if len(t_src) < 2:
        return out
    for i in range(3):
        out[:, i] = np.interp(t_query, t_src, x_src[:, i], left=np.nan, right=np.nan)
    return out


def build_chaser_track(log: MasterSimLog, rocket_name: str, inserted_name: str) -> np.ndarray:
    t = log.t_s
    chaser = np.full((len(t), 3), np.nan, dtype=float)

    if rocket_name in log.rocket_t_s_by_name and rocket_name in log.rocket_x_eci_by_name:
        t_r = log.rocket_t_s_by_name[rocket_name]
        x_r = log.rocket_x_eci_by_name[rocket_name][:, :3]
        rocket_interp = _interp_vec(t_r, x_r, t)
        before_handoff = t <= t_r[-1]
        chaser[before_handoff, :] = rocket_interp[before_handoff, :]

    if inserted_name in log.x_eci_by_agent:
        x_i = log.x_eci_by_agent[inserted_name][:, :3]
        valid = np.isfinite(x_i[:, 0])
        chaser[valid, :] = x_i[valid, :]

    return chaser


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate target and rocket/inserted chaser as ECI dots.")
    parser.add_argument(
        "--log",
        type=str,
        default=str(REPO_ROOT / "outputs" / "master_sim_log.npz"),
        help="Path to master sim log (.npz).",
    )
    parser.add_argument("--target", type=str, default="target", help="Target agent name in log.")
    parser.add_argument("--inserted", type=str, default="inserted_sat", help="Inserted chaser agent name in log.")
    parser.add_argument("--rocket", type=str, default="booster_1", help="Rocket name in rocket telemetry.")
    parser.add_argument("--step", type=int, default=5, help="Frame stride over log samples.")
    parser.add_argument("--fps", type=int, default=30, help="Output MP4 frames per second.")
    parser.add_argument(
        "--mp4",
        type=str,
        default=str(REPO_ROOT / "outputs" / "master_sim_eci_animation.mp4"),
        help="Output MP4 path.",
    )
    parser.add_argument("--show", action="store_true", help="Show a window while also saving MP4.")
    parser.add_argument("--no-show", action="store_true", help="Build animation but do not open a window.")
    args = parser.parse_args()

    log = MasterSimLog.load_npz(args.log)

    if args.target not in log.x_eci_by_agent:
        raise ValueError(f"Target '{args.target}' not found in log x_eci_by_agent.")

    x_target = log.x_eci_by_agent[args.target][:, :3]
    chaser = build_chaser_track(log, rocket_name=args.rocket, inserted_name=args.inserted)
    handoff_time_s: Optional[float] = None
    if args.rocket in log.rocket_t_s_by_name:
        t_r = log.rocket_t_s_by_name[args.rocket]
        if len(t_r) > 0:
            handoff_time_s = float(t_r[-1])

    valid_target = np.isfinite(x_target[:, 0])
    valid_chaser = np.isfinite(chaser[:, 0])
    valid_any = valid_target | valid_chaser
    if not np.any(valid_any):
        raise RuntimeError("No valid target/chaser samples found for animation.")

    xyz_all = np.vstack((x_target[valid_target, :], chaser[valid_chaser, :]))
    mins = np.min(xyz_all, axis=0)
    maxs = np.max(xyz_all, axis=0)
    center = 0.5 * (mins + maxs)
    half_span = 0.5 * np.max(maxs - mins)
    half_span = max(half_span, 1.0)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X ECI (km)")
    ax.set_ylabel("Y ECI (km)")
    ax.set_zlabel("Z ECI (km)")
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.set_box_aspect((1.0, 1.0, 1.0))

    # Draw Earth as a translucent sphere in ECI.
    re_km = 6378.137
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    x_sphere = re_km * np.outer(np.cos(u), np.sin(v))
    y_sphere = re_km * np.outer(np.sin(u), np.sin(v))
    z_sphere = re_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, linewidth=0.0, color="tab:gray")

    target_dot = ax.scatter([], [], [], s=40, c="tab:blue", label="Target")
    chaser_dot = ax.scatter([], [], [], s=40, c="tab:orange", label="Rocket/Inserted Chaser")
    ax.legend(loc="upper right")

    indices = np.arange(0, len(log.t_s), max(args.step, 1), dtype=int)

    def _set_dot(dot, xyz: np.ndarray) -> None:
        if np.isfinite(xyz[0]):
            dot._offsets3d = ([xyz[0]], [xyz[1]], [xyz[2]])
        else:
            dot._offsets3d = ([], [], [])

    def update(frame_idx: int):
        k = indices[frame_idx]
        _set_dot(target_dot, x_target[k, :])
        _set_dot(chaser_dot, chaser[k, :])
        if handoff_time_s is None or log.t_s[k] <= handoff_time_s:
            chaser_dot.set_color("tab:orange")  # Rocket phase
        else:
            chaser_dot.set_color("tab:red")  # Inserted satellite phase
        ax.set_title(f"ECI Animation | t = {log.t_s[k]:.1f} s")
        return target_dot, chaser_dot

    if args.no_show:
        update(0)
        update(len(indices) - 1)
        plt.close(fig)
        print(f"Animation built with {len(indices)} frames (step={args.step}).")
        return

    fps = max(int(args.fps), 1)
    anim = FuncAnimation(fig, update, frames=len(indices), interval=int(1000 / fps), blit=False)

    out_path = Path(args.mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(out_path), writer="ffmpeg", fps=fps, bitrate=1800)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(
            "Failed to save MP4. Ensure ffmpeg is installed and available on PATH."
        ) from exc

    print(f"Saved MP4: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
