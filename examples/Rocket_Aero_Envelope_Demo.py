from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.rocket import RocketAeroConfig
from sim.rocket.aero import compute_aero_loads, compute_aero_state


def _slice_indices(values: np.ndarray, targets: list[float]) -> list[int]:
    idx: list[int] = []
    for x in targets:
        idx.append(int(np.argmin(np.abs(values - x))))
    return idx


def run_demo(plot_mode: str = "interactive") -> dict[str, str]:
    cfg = RocketAeroConfig(
        enabled=True,
        reference_area_m2=10.0,
        reference_length_m=35.0,
        cp_offset_body_m=np.array([-2.5, 0.0, 0.0]),
        cd_base=0.18,
        cd_alpha2=0.08,
        cd_supersonic=0.28,
        transonic_peak_cd=0.25,
        transonic_mach=1.0,
        transonic_width=0.20,
        cl_alpha_per_rad=0.12,
        cm_alpha_per_rad=-0.02,
    )

    mach_grid = np.linspace(0.05, 8.0, 240)
    alpha_deg_grid = np.linspace(-15.0, 15.0, 121)
    rho_kg_m3 = 1.225
    p_pa = 101325.0
    t_k = 288.15
    a_m_s = 340.0
    beta_rad = 0.0

    cd_map = np.zeros((alpha_deg_grid.size, mach_grid.size))
    cl_map = np.zeros((alpha_deg_grid.size, mach_grid.size))
    cm_map = np.zeros((alpha_deg_grid.size, mach_grid.size))

    for i, alpha_deg in enumerate(alpha_deg_grid):
        alpha = np.deg2rad(alpha_deg)
        for j, mach in enumerate(mach_grid):
            speed = float(mach * a_m_s)
            v_body = speed * np.array([np.cos(alpha) * np.cos(beta_rad), np.sin(beta_rad), np.sin(alpha)], dtype=float)
            state = compute_aero_state(
                rho_kg_m3=rho_kg_m3,
                pressure_pa=p_pa,
                temperature_k=t_k,
                sound_speed_m_s=a_m_s,
                v_rel_body_m_s=v_body,
                alpha_limit_deg=cfg.alpha_limit_deg,
                beta_limit_deg=cfg.beta_limit_deg,
            )
            loads = compute_aero_loads(v_rel_body_m_s=v_body, atmos=state, cfg=cfg)
            cd_map[i, j] = -loads.coeff_force_body[0]
            cl_map[i, j] = -loads.coeff_force_body[2]
            cm_map[i, j] = loads.coeff_moment_body[1]

    outdir = REPO_ROOT / "outputs" / "rocket_aero_envelope_demo"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    extent = [float(mach_grid[0]), float(mach_grid[-1]), float(alpha_deg_grid[0]), float(alpha_deg_grid[-1])]
    im0 = ax[0].imshow(cd_map, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    im1 = ax[1].imshow(cl_map, aspect="auto", origin="lower", extent=extent, cmap="coolwarm")
    im2 = ax[2].imshow(cm_map, aspect="auto", origin="lower", extent=extent, cmap="coolwarm")
    ax[0].set_title("Cd(Mach, alpha)")
    ax[1].set_title("Cl(Mach, alpha)")
    ax[2].set_title("Cm(Mach, alpha)")
    for a in ax:
        a.set_xlabel("Mach")
        a.grid(True, alpha=0.2)
    ax[0].set_ylabel("Alpha (deg)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    fig.suptitle("Rocket Aerodynamic Coefficient Envelope")
    fig.tight_layout()
    p1 = outdir / "rocket_aero_envelope_heatmaps.png"
    if plot_mode in ("save", "both"):
        fig.savefig(p1, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    alpha_targets = [-10.0, -5.0, 0.0, 5.0, 10.0]
    mach_targets = [0.5, 0.9, 1.0, 1.2, 2.0]
    alpha_idx = _slice_indices(alpha_deg_grid, alpha_targets)
    mach_idx = _slice_indices(mach_grid, mach_targets)

    fig2, ax2 = plt.subplots(3, 2, figsize=(13, 10))
    for idx in alpha_idx:
        label = f"alpha={alpha_deg_grid[idx]:.1f} deg"
        ax2[0, 0].plot(mach_grid, cd_map[idx, :], label=label)
        ax2[1, 0].plot(mach_grid, cl_map[idx, :], label=label)
        ax2[2, 0].plot(mach_grid, cm_map[idx, :], label=label)
    for idx in mach_idx:
        label = f"Mach={mach_grid[idx]:.2f}"
        ax2[0, 1].plot(alpha_deg_grid, cd_map[:, idx], label=label)
        ax2[1, 1].plot(alpha_deg_grid, cl_map[:, idx], label=label)
        ax2[2, 1].plot(alpha_deg_grid, cm_map[:, idx], label=label)

    ax2[0, 0].set_title("Cd vs Mach (alpha slices)")
    ax2[1, 0].set_title("Cl vs Mach (alpha slices)")
    ax2[2, 0].set_title("Cm vs Mach (alpha slices)")
    ax2[0, 1].set_title("Cd vs Alpha (Mach slices)")
    ax2[1, 1].set_title("Cl vs Alpha (Mach slices)")
    ax2[2, 1].set_title("Cm vs Alpha (Mach slices)")
    ax2[2, 0].set_xlabel("Mach")
    ax2[2, 1].set_xlabel("Alpha (deg)")
    for r in range(3):
        ax2[r, 0].set_ylabel(["Cd", "Cl", "Cm"][r])
        for c in range(2):
            ax2[r, c].grid(True, alpha=0.3)
            ax2[r, c].legend(loc="best", fontsize=8)
    fig2.suptitle("Rocket Aero Envelope Slices")
    fig2.tight_layout()
    p2 = outdir / "rocket_aero_envelope_slices.png"
    if plot_mode in ("save", "both"):
        fig2.savefig(p2, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig2)

    return {
        "heatmap_plot": str(p1) if plot_mode in ("save", "both") else "",
        "slice_plot": str(p2) if plot_mode in ("save", "both") else "",
        "mach_min": f"{mach_grid[0]:.2f}",
        "mach_max": f"{mach_grid[-1]:.2f}",
        "alpha_min_deg": f"{alpha_deg_grid[0]:.1f}",
        "alpha_max_deg": f"{alpha_deg_grid[-1]:.1f}",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rocket aero envelope demo (Mach/alpha sweeps for Cd, Cl, Cm).")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    args = parser.parse_args()
    result = run_demo(plot_mode=args.plot_mode)
    print("Rocket aero envelope demo outputs:")
    for k, v in result.items():
        if v:
            print(f"  {k}: {v}")
