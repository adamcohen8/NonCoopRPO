from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HCWPositionTransferSolution:
    transfer_time_s: float
    target_delta_v_ric_km_s: np.ndarray
    post_target_rel_state_ric: np.ndarray
    required_post_chaser_rel_velocity_ric_km_s: np.ndarray
    required_delta_v_ric_km_s: np.ndarray
    rendezvous_state_ric: np.ndarray

    @property
    def required_delta_v_mag_km_s(self) -> float:
        return float(np.linalg.norm(self.required_delta_v_ric_km_s))


@dataclass(frozen=True)
class HCWEvasionOptimizationResult:
    best_direction_ric: np.ndarray
    best_solution: HCWPositionTransferSolution
    candidate_direction_history_ric: np.ndarray
    candidate_cost_history_km_s: np.ndarray
    best_cost_history_km_s: np.ndarray
    sigma_history_rad: np.ndarray


def _as_unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(v, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        raise ValueError("Direction vector must be non-zero.")
    return vec / norm


def hcw_state_transition_matrix(mean_motion_rad_s: float, dt_s: float) -> np.ndarray:
    n = float(mean_motion_rad_s)
    dt = float(dt_s)
    if n <= 0.0:
        raise ValueError("mean_motion_rad_s must be positive.")
    if dt < 0.0:
        raise ValueError("dt_s must be non-negative.")

    nt = n * dt
    c = float(np.cos(nt))
    s = float(np.sin(nt))

    return np.array(
        [
            [4.0 - 3.0 * c, 0.0, 0.0, s / n, 2.0 * (1.0 - c) / n, 0.0],
            [6.0 * (s - nt), 1.0, 0.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n, 0.0],
            [0.0, 0.0, c, 0.0, 0.0, s / n],
            [3.0 * n * s, 0.0, 0.0, c, 2.0 * s, 0.0],
            [-6.0 * n * (1.0 - c), 0.0, 0.0, -2.0 * s, 4.0 * c - 3.0, 0.0],
            [0.0, 0.0, -n * s, 0.0, 0.0, c],
        ],
        dtype=float,
    )


def hcw_state_transition_blocks(
    mean_motion_rad_s: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi = hcw_state_transition_matrix(mean_motion_rad_s=mean_motion_rad_s, dt_s=dt_s)
    return phi[:3, :3], phi[:3, 3:], phi[3:, :3], phi[3:, 3:]


def hcw_phi_rv(mean_motion_rad_s: float, dt_s: float) -> np.ndarray:
    _, phi_rv, _, _ = hcw_state_transition_blocks(mean_motion_rad_s=mean_motion_rad_s, dt_s=dt_s)
    return phi_rv


def propagate_hcw_relative_state(x0_ric: np.ndarray, mean_motion_rad_s: float, dt_s: float) -> np.ndarray:
    x0 = np.asarray(x0_ric, dtype=float).reshape(6)
    phi = hcw_state_transition_matrix(mean_motion_rad_s=mean_motion_rad_s, dt_s=dt_s)
    return phi @ x0


def solve_hcw_position_rendezvous(
    initial_rel_state_ric: np.ndarray,
    target_delta_v_ric_km_s: np.ndarray,
    mean_motion_rad_s: float,
    transfer_time_s: float,
    singularity_tol: float = 1e-10,
) -> HCWPositionTransferSolution:
    x0 = np.asarray(initial_rel_state_ric, dtype=float).reshape(6)
    dv_target = np.asarray(target_delta_v_ric_km_s, dtype=float).reshape(3)
    if transfer_time_s <= 0.0:
        raise ValueError("transfer_time_s must be positive.")

    x_post_target = x0.copy()
    x_post_target[3:] = x_post_target[3:] - dv_target

    phi_rr, phi_rv, _, _ = hcw_state_transition_blocks(mean_motion_rad_s=mean_motion_rad_s, dt_s=transfer_time_s)
    det_phi_rv = float(np.linalg.det(phi_rv))
    if abs(det_phi_rv) <= singularity_tol:
        raise ValueError(
            "Transfer time is near an HCW singularity; choose a different rendezvous time so Phi_rv is invertible."
        )

    required_postburn_velocity = np.linalg.solve(phi_rv, -(phi_rr @ x_post_target[:3]))
    required_delta_v = required_postburn_velocity - x_post_target[3:]
    x_post_chaser = np.hstack((x_post_target[:3], required_postburn_velocity))
    x_rendezvous = propagate_hcw_relative_state(
        x0_ric=x_post_chaser,
        mean_motion_rad_s=mean_motion_rad_s,
        dt_s=transfer_time_s,
    )
    return HCWPositionTransferSolution(
        transfer_time_s=float(transfer_time_s),
        target_delta_v_ric_km_s=dv_target,
        post_target_rel_state_ric=x_post_target,
        required_post_chaser_rel_velocity_ric_km_s=required_postburn_velocity,
        required_delta_v_ric_km_s=required_delta_v,
        rendezvous_state_ric=x_rendezvous,
    )


def optimize_hcw_evasion_burn_direction(
    initial_rel_state_ric: np.ndarray,
    target_delta_v_mag_km_s: float,
    mean_motion_rad_s: float,
    transfer_time_s: float,
    iterations: int = 200,
    seed: int = 0,
    initial_sigma_rad: float = 0.8,
    sigma_decay: float = 0.985,
    min_sigma_rad: float = np.deg2rad(0.25),
) -> HCWEvasionOptimizationResult:
    if target_delta_v_mag_km_s < 0.0:
        raise ValueError("target_delta_v_mag_km_s must be non-negative.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if initial_sigma_rad <= 0.0:
        raise ValueError("initial_sigma_rad must be positive.")
    if not (0.0 < sigma_decay <= 1.0):
        raise ValueError("sigma_decay must be in (0, 1].")
    if min_sigma_rad <= 0.0:
        raise ValueError("min_sigma_rad must be positive.")

    rng = np.random.default_rng(seed)

    seed_dirs = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
        ],
        dtype=float,
    )

    def evaluate(direction_ric: np.ndarray) -> HCWPositionTransferSolution:
        dv_target = target_delta_v_mag_km_s * _as_unit(direction_ric)
        return solve_hcw_position_rendezvous(
            initial_rel_state_ric=initial_rel_state_ric,
            target_delta_v_ric_km_s=dv_target,
            mean_motion_rad_s=mean_motion_rad_s,
            transfer_time_s=transfer_time_s,
        )

    best_solution: HCWPositionTransferSolution | None = None
    best_direction = np.array([1.0, 0.0, 0.0], dtype=float)
    for direction in seed_dirs:
        sol = evaluate(direction)
        if best_solution is None or sol.required_delta_v_mag_km_s > best_solution.required_delta_v_mag_km_s:
            best_solution = sol
            best_direction = _as_unit(direction)

    assert best_solution is not None

    sigma = float(initial_sigma_rad)
    candidate_dirs = np.zeros((iterations, 3), dtype=float)
    candidate_costs = np.zeros(iterations, dtype=float)
    best_costs = np.zeros(iterations, dtype=float)
    sigma_hist = np.zeros(iterations, dtype=float)

    for k in range(iterations):
        proposal = best_direction + sigma * rng.normal(size=3)
        proposal = _as_unit(proposal)
        sol = evaluate(proposal)
        if sol.required_delta_v_mag_km_s > best_solution.required_delta_v_mag_km_s:
            best_solution = sol
            best_direction = proposal
        candidate_dirs[k, :] = proposal
        candidate_costs[k] = sol.required_delta_v_mag_km_s
        best_costs[k] = best_solution.required_delta_v_mag_km_s
        sigma_hist[k] = sigma
        sigma = max(min_sigma_rad, sigma * sigma_decay)

    return HCWEvasionOptimizationResult(
        best_direction_ric=best_direction,
        best_solution=best_solution,
        candidate_direction_history_ric=candidate_dirs,
        candidate_cost_history_km_s=candidate_costs,
        best_cost_history_km_s=best_costs,
        sigma_history_rad=sigma_hist,
    )
