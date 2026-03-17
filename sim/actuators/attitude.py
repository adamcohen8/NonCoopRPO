from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Actuator
from sim.core.models import Command


@dataclass(frozen=True)
class ReactionWheelLimits:
    max_torque_nm: np.ndarray
    max_momentum_nms: np.ndarray
    wheel_axes_body: np.ndarray | None = None  # shape (3,N) or (N,3)
    wheel_inertia_kg_m2: np.ndarray | float | None = None
    max_speed_rad_s: np.ndarray | float | None = None
    torque_time_constant_s: float = 0.0
    viscous_friction_nms: np.ndarray | float = 0.0
    coulomb_friction_nm: np.ndarray | float = 0.0


@dataclass(frozen=True)
class MagnetorquerLimits:
    max_dipole_a_m2: np.ndarray


@dataclass(frozen=True)
class ThrusterPulseLimits:
    max_torque_nm: np.ndarray
    pulse_quantum_s: float = 0.02


@dataclass
class AttitudeActuator(Actuator):
    reaction_wheels: ReactionWheelLimits | None = None
    magnetorquers: MagnetorquerLimits | None = None
    thruster_pulse: ThrusterPulseLimits | None = None
    # Body-equivalent wheel momentum vector (N*m*s) for telemetry/compatibility.
    wheel_momentum_nms: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Per-wheel internal states (allocated lazily from wheel config).
    wheel_momentum_wheels_nms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wheel_speed_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wheel_motor_torque_nm: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def apply(self, command: Command, limits: dict, dt_s: float) -> Command:
        torque = np.array(command.torque_body_nm, dtype=float)
        mode_flags = dict(command.mode_flags)

        if self.reaction_wheels is not None:
            torque, rw_diag = self._apply_reaction_wheels(
                torque_body_cmd_nm=torque,
                dt_s=dt_s,
                mode_flags=mode_flags,
            )
            mode_flags.update(rw_diag)

        if self.thruster_pulse is not None:
            tp = self.thruster_pulse
            torque = np.clip(torque, -tp.max_torque_nm, tp.max_torque_nm)
            if tp.pulse_quantum_s > 0.0:
                pulses = np.round(dt_s / tp.pulse_quantum_s)
                scale = 0.0 if pulses <= 0 else pulses * tp.pulse_quantum_s / dt_s
                torque *= scale

        # Magnetorquer coupling to geomagnetic field would require B-field in env.
        # Here we enforce achievable moment command proxy via clamp.
        if self.magnetorquers is not None:
            mt = self.magnetorquers
            torque = np.clip(torque, -np.abs(mt.max_dipole_a_m2), np.abs(mt.max_dipole_a_m2))

        return Command(
            thrust_eci_km_s2=np.array(command.thrust_eci_km_s2, dtype=float),
            torque_body_nm=torque,
            mode_flags=mode_flags,
        )

    @staticmethod
    def _as_vector(value: np.ndarray | float | None, n: int, default: float = 0.0) -> np.ndarray:
        if value is None:
            return np.full(n, float(default), dtype=float)
        arr = np.array(value, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.full(n, float(arr[0]), dtype=float)
        if arr.size != n:
            raise ValueError(f"Expected scalar or length-{n} vector, got length {arr.size}.")
        return arr

    @staticmethod
    def _resolve_wheel_axes(wheel_axes_body: np.ndarray | None, n: int) -> np.ndarray:
        if wheel_axes_body is None:
            if n == 3:
                return np.eye(3, dtype=float)
            raise ValueError("wheel_axes_body is required when number of wheels is not 3.")
        axes = np.array(wheel_axes_body, dtype=float)
        if axes.ndim != 2:
            raise ValueError("wheel_axes_body must be a 2D array.")
        if axes.shape[0] == 3:
            g = axes.copy()
        elif axes.shape[1] == 3:
            g = axes.T.copy()
        else:
            raise ValueError("wheel_axes_body must have shape (3,N) or (N,3).")
        if g.shape[1] != n:
            raise ValueError(f"wheel_axes_body must contain {n} wheel axes.")
        for k in range(n):
            norm = float(np.linalg.norm(g[:, k]))
            if norm <= 0.0:
                raise ValueError("wheel_axes_body contains a zero vector.")
            g[:, k] /= norm
        return g

    def _apply_reaction_wheels(
        self,
        torque_body_cmd_nm: np.ndarray,
        dt_s: float,
        mode_flags: dict,
    ) -> tuple[np.ndarray, dict[str, object]]:
        rw = self.reaction_wheels
        if rw is None:
            return torque_body_cmd_nm, {}

        n_wheels = int(np.array(rw.max_torque_nm, dtype=float).reshape(-1).size)
        if n_wheels <= 0:
            return np.zeros(3, dtype=float), {}

        g = self._resolve_wheel_axes(rw.wheel_axes_body, n=n_wheels)
        max_torque_nm = np.abs(self._as_vector(rw.max_torque_nm, n=n_wheels, default=0.0))
        max_momentum_nms = np.abs(self._as_vector(rw.max_momentum_nms, n=n_wheels, default=np.inf))
        max_speed_rad_s = np.abs(self._as_vector(rw.max_speed_rad_s, n=n_wheels, default=np.inf))
        torque_tau_s = float(max(rw.torque_time_constant_s, 0.0))
        viscous_nms = np.abs(self._as_vector(rw.viscous_friction_nms, n=n_wheels, default=0.0))
        coulomb_nm = np.abs(self._as_vector(rw.coulomb_friction_nm, n=n_wheels, default=0.0))

        # Derive wheel inertia from explicit input or from h_max / w_max when available.
        j_kg_m2 = self._as_vector(rw.wheel_inertia_kg_m2, n=n_wheels, default=np.nan)
        inferred_mask = ~np.isfinite(j_kg_m2)
        if np.any(inferred_mask):
            infer_ok = np.isfinite(max_momentum_nms) & np.isfinite(max_speed_rad_s) & (max_speed_rad_s > 0.0)
            j_kg_m2[inferred_mask & infer_ok] = max_momentum_nms[inferred_mask & infer_ok] / max_speed_rad_s[inferred_mask & infer_ok]
            j_kg_m2[~np.isfinite(j_kg_m2)] = 5e-4
        j_kg_m2 = np.clip(j_kg_m2, 1e-9, np.inf)

        if self.wheel_speed_rad_s.size != n_wheels:
            self.wheel_speed_rad_s = np.zeros(n_wheels, dtype=float)
        if self.wheel_motor_torque_nm.size != n_wheels:
            self.wheel_motor_torque_nm = np.zeros(n_wheels, dtype=float)
        if self.wheel_momentum_wheels_nms.size != n_wheels:
            self.wheel_momentum_wheels_nms = np.zeros(n_wheels, dtype=float)

        wheel_torque_mode_flag = mode_flags.get("wheel_torque_cmd_nm", None)
        if wheel_torque_mode_flag is not None:
            tau_cmd = np.array(wheel_torque_mode_flag, dtype=float).reshape(-1)
            if tau_cmd.size != n_wheels:
                tau_cmd = np.linalg.pinv(g) @ np.array(torque_body_cmd_nm, dtype=float).reshape(3)
        else:
            tau_cmd = np.linalg.pinv(g) @ np.array(torque_body_cmd_nm, dtype=float).reshape(3)
        tau_cmd = np.clip(tau_cmd, -max_torque_nm, max_torque_nm)

        # First-order wheel motor torque lag.
        if dt_s <= 0.0 or torque_tau_s <= 0.0:
            tau_motor = tau_cmd
        else:
            alpha = float(np.clip(dt_s / torque_tau_s, 0.0, 1.0))
            tau_motor = self.wheel_motor_torque_nm + alpha * (tau_cmd - self.wheel_motor_torque_nm)
        tau_motor = np.clip(tau_motor, -max_torque_nm, max_torque_nm)
        tau_motor_eff = tau_motor.copy()

        # Friction torque opposes wheel spin and reduces achievable wheel acceleration.
        omega = np.array(self.wheel_speed_rad_s, dtype=float)
        tau_fric = viscous_nms * omega + coulomb_nm * np.sign(omega)
        tau_net = tau_motor_eff - tau_fric

        # Prevent driving further into momentum saturation.
        h_now = j_kg_m2 * omega
        sat_hi = (h_now >= (max_momentum_nms - 1e-12)) & (tau_net > 0.0)
        sat_lo = (h_now <= (-max_momentum_nms + 1e-12)) & (tau_net < 0.0)
        sat_h = sat_hi | sat_lo
        tau_net[sat_h] = 0.0
        tau_motor_eff[sat_h] = tau_fric[sat_h]

        omega_prop = omega + dt_s * (tau_net / j_kg_m2)
        sat_speed_hi = (omega_prop >= max_speed_rad_s) & (tau_net > 0.0)
        sat_speed_lo = (omega_prop <= -max_speed_rad_s) & (tau_net < 0.0)
        sat_speed = sat_speed_hi | sat_speed_lo
        tau_net[sat_speed] = 0.0
        tau_motor_eff[sat_speed] = tau_fric[sat_speed]
        omega_next = omega + dt_s * (tau_net / j_kg_m2)
        omega_next = np.clip(omega_next, -max_speed_rad_s, max_speed_rad_s)
        h_next = np.clip(j_kg_m2 * omega_next, -max_momentum_nms, max_momentum_nms)
        omega_next = h_next / j_kg_m2

        self.wheel_motor_torque_nm = tau_motor_eff
        self.wheel_speed_rad_s = omega_next
        self.wheel_momentum_wheels_nms = h_next
        self.wheel_momentum_nms = g @ h_next

        torque_body_nm = g @ tau_motor_eff
        diag = {
            "rw_num_wheels": int(n_wheels),
            "rw_torque_cmd_nm": tau_cmd.tolist(),
            "rw_torque_applied_nm": tau_motor_eff.tolist(),
            "rw_speed_rad_s": omega_next.tolist(),
            "rw_momentum_wheels_nms": h_next.tolist(),
            "rw_momentum_body_nms": self.wheel_momentum_nms.tolist(),
        }
        return torque_body_nm, diag
