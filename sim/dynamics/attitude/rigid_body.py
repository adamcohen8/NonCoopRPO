from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from sim.utils.quaternion import (
    normalize_quaternion,
    omega_matrix,
    quaternion_delta_from_body_rate,
    quaternion_multiply,
)

_MAX_ABS_RATE_RAD_S = 1e6
_MAX_ABS_TORQUE_NM = 1e12


@dataclass
class AttitudeGuardrailStats:
    non_finite_input_events: int = 0
    rate_clamp_events: int = 0
    torque_clamp_events: int = 0
    non_finite_coriolis_events: int = 0
    singular_inertia_events: int = 0
    non_finite_output_events: int = 0


_ATTITUDE_GUARDRAIL_STATS = AttitudeGuardrailStats()


def reset_attitude_guardrail_stats() -> None:
    _ATTITUDE_GUARDRAIL_STATS.non_finite_input_events = 0
    _ATTITUDE_GUARDRAIL_STATS.rate_clamp_events = 0
    _ATTITUDE_GUARDRAIL_STATS.torque_clamp_events = 0
    _ATTITUDE_GUARDRAIL_STATS.non_finite_coriolis_events = 0
    _ATTITUDE_GUARDRAIL_STATS.singular_inertia_events = 0
    _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events = 0


def get_attitude_guardrail_stats() -> dict[str, int]:
    return asdict(_ATTITUDE_GUARDRAIL_STATS)


def rigid_body_derivatives(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    torque_body_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = normalize_quaternion(quat_bn)
    w = np.asarray(omega_body_rad_s, dtype=float).reshape(3)
    I = np.asarray(inertia_kg_m2, dtype=float).reshape(3, 3)
    tau = np.asarray(torque_body_nm, dtype=float).reshape(3)

    # Clamp/sanitize extremes to keep attitude propagation numerically stable.
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(tau)) and np.all(np.isfinite(I))):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_input_events += 1
    w_nonfinite = ~np.isfinite(w)
    tau_nonfinite = ~np.isfinite(tau)
    w = np.nan_to_num(w, nan=0.0, posinf=_MAX_ABS_RATE_RAD_S, neginf=-_MAX_ABS_RATE_RAD_S)
    tau = np.nan_to_num(tau, nan=0.0, posinf=_MAX_ABS_TORQUE_NM, neginf=-_MAX_ABS_TORQUE_NM)
    _ATTITUDE_GUARDRAIL_STATS.rate_clamp_events += int(np.sum(w_nonfinite))
    _ATTITUDE_GUARDRAIL_STATS.torque_clamp_events += int(np.sum(tau_nonfinite))
    w_pre_clip = w.copy()
    tau_pre_clip = tau.copy()
    w = np.clip(w, -_MAX_ABS_RATE_RAD_S, _MAX_ABS_RATE_RAD_S)
    tau = np.clip(tau, -_MAX_ABS_TORQUE_NM, _MAX_ABS_TORQUE_NM)
    _ATTITUDE_GUARDRAIL_STATS.rate_clamp_events += int(np.sum(w != w_pre_clip))
    _ATTITUDE_GUARDRAIL_STATS.torque_clamp_events += int(np.sum(tau != tau_pre_clip))

    q_dot = 0.5 * omega_matrix(w) @ q
    Iw = I @ w
    coriolis = np.array(
        [
            w[1] * Iw[2] - w[2] * Iw[1],
            w[2] * Iw[0] - w[0] * Iw[2],
            w[0] * Iw[1] - w[1] * Iw[0],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(coriolis)):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_coriolis_events += 1
    rhs = tau - np.nan_to_num(coriolis, nan=0.0, posinf=_MAX_ABS_TORQUE_NM, neginf=-_MAX_ABS_TORQUE_NM)
    try:
        omega_dot = np.linalg.solve(I, rhs)
    except np.linalg.LinAlgError:
        _ATTITUDE_GUARDRAIL_STATS.singular_inertia_events += 1
        omega_dot = np.zeros(3, dtype=float)
    if not np.all(np.isfinite(omega_dot)):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events += 1
    omega_dot = np.nan_to_num(omega_dot, nan=0.0, posinf=_MAX_ABS_RATE_RAD_S, neginf=-_MAX_ABS_RATE_RAD_S)
    return q_dot, omega_dot


def propagate_attitude_euler(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    torque_body_nm: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_dot, omega_dot = rigid_body_derivatives(quat_bn, omega_body_rad_s, inertia_kg_m2, torque_body_nm)
    dt = float(max(dt_s, 0.0))
    q_next = normalize_quaternion(np.array(quat_bn, dtype=float).reshape(4) + dt * q_dot)
    omega_next = np.array(omega_body_rad_s, dtype=float).reshape(3) + dt * omega_dot
    if not (np.all(np.isfinite(q_next)) and np.all(np.isfinite(omega_next))):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events += 1
    omega_next = np.nan_to_num(omega_next, nan=0.0, posinf=_MAX_ABS_RATE_RAD_S, neginf=-_MAX_ABS_RATE_RAD_S)
    omega_next = np.clip(omega_next, -_MAX_ABS_RATE_RAD_S, _MAX_ABS_RATE_RAD_S)
    return q_next, omega_next


def propagate_attitude_exponential_map(
    quat_bn: np.ndarray,
    omega_body_rad_s: np.ndarray,
    inertia_kg_m2: np.ndarray,
    torque_body_nm: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Integrate angular-rate dynamics with first-order step.
    _, omega_dot = rigid_body_derivatives(quat_bn, omega_body_rad_s, inertia_kg_m2, torque_body_nm)
    dt = float(max(dt_s, 0.0))
    omega_now = np.asarray(omega_body_rad_s, dtype=float).reshape(3)
    omega_next = omega_now + dt * omega_dot
    if not np.all(np.isfinite(omega_next)):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events += 1
    omega_next = np.nan_to_num(omega_next, nan=0.0, posinf=_MAX_ABS_RATE_RAD_S, neginf=-_MAX_ABS_RATE_RAD_S)
    omega_next = np.clip(omega_next, -_MAX_ABS_RATE_RAD_S, _MAX_ABS_RATE_RAD_S)

    # Use midpoint body rate to build quaternion delta via exponential map.
    omega_mid = omega_now + 0.5 * dt * omega_dot
    if not np.all(np.isfinite(omega_mid)):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events += 1
    omega_mid = np.nan_to_num(omega_mid, nan=0.0, posinf=_MAX_ABS_RATE_RAD_S, neginf=-_MAX_ABS_RATE_RAD_S)
    omega_mid = np.clip(omega_mid, -_MAX_ABS_RATE_RAD_S, _MAX_ABS_RATE_RAD_S)
    dq = quaternion_delta_from_body_rate(omega_mid, dt)
    # q_dot uses Omega(w) @ q with the convention equivalent to q ⊗ [0, w],
    # so the finite update must right-multiply by dq.
    q_next = normalize_quaternion(quaternion_multiply(quat_bn, dq))
    if not np.all(np.isfinite(q_next)):
        _ATTITUDE_GUARDRAIL_STATS.non_finite_output_events += 1
    return q_next, omega_next
