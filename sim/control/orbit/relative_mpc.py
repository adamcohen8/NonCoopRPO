from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.utils.frames import eci_relative_to_ric_rect, ric_curv_to_rect, ric_dcm_ir_from_rv, ric_rect_state_to_eci


@dataclass
class RelativeOrbitMPCController(Controller):
    """Nonlinear MPC for relative orbital maneuvering with 2-body prediction."""

    max_accel_km_s2: float
    horizon_steps: int = 8
    step_dt_s: float = 10.0
    mu_km3_s2: float = 398600.4418

    ric_curv_state_slice: tuple[int, int] = (0, 6)
    chief_eci_state_slice: tuple[int, int] = (6, 12)
    state_signs: np.ndarray = field(default_factory=lambda: np.ones(6))
    target_rel_ric_rect: np.ndarray = field(default_factory=lambda: np.zeros(6))

    q_weights: np.ndarray = field(default_factory=lambda: np.array([2.0e3, 2.0e3, 2.0e3, 3.0e7, 3.0e7, 3.0e7]))
    terminal_weights: np.ndarray = field(
        default_factory=lambda: np.array([8.0e3, 8.0e3, 8.0e3, 1.2e8, 1.2e8, 1.2e8])
    )
    r_weights: np.ndarray = field(default_factory=lambda: np.ones(3) * 4.0e12)
    rd_weights: np.ndarray = field(default_factory=lambda: np.ones(3) * 4.0e12)

    seed_kp_pos: float = 1.0e-6
    seed_kd_vel: float = 2.0e-2
    seed_decay: float = 0.85

    max_iterations: int = 3
    gradient_alpha: float = 1.0
    line_search_shrink: float = 0.5
    fd_epsilon: float = 1e-7
    gradient_method: str = "spsa"
    spsa_delta: float = 1e-6
    grad_tol: float = 1e-12
    line_search_min_alpha: float = 1e-3
    min_cost_improvement: float = 1e-6
    trust_region_step_km_s2: float = 1e-5

    _u_guess_eci: np.ndarray = field(init=False, repr=False)
    _u_prev_eci: np.ndarray = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_accel_km_s2 < 0.0:
            raise ValueError("max_accel_km_s2 must be non-negative.")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive.")
        if self.step_dt_s <= 0.0:
            raise ValueError("step_dt_s must be positive.")
        if self.mu_km3_s2 <= 0.0:
            raise ValueError("mu_km3_s2 must be positive.")
        if self.ric_curv_state_slice[1] - self.ric_curv_state_slice[0] != 6:
            raise ValueError("ric_curv_state_slice must select exactly 6 elements.")
        if self.chief_eci_state_slice[1] - self.chief_eci_state_slice[0] != 6:
            raise ValueError("chief_eci_state_slice must select exactly 6 elements.")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if self.seed_kp_pos < 0.0:
            raise ValueError("seed_kp_pos must be non-negative.")
        if self.seed_kd_vel < 0.0:
            raise ValueError("seed_kd_vel must be non-negative.")
        if not (0.0 <= self.seed_decay <= 1.0):
            raise ValueError("seed_decay must be in [0, 1].")
        if self.gradient_alpha <= 0.0:
            raise ValueError("gradient_alpha must be positive.")
        if not (0.0 < self.line_search_shrink < 1.0):
            raise ValueError("line_search_shrink must be in (0, 1).")
        if self.fd_epsilon <= 0.0:
            raise ValueError("fd_epsilon must be positive.")
        if self.gradient_method not in ("finite_difference", "spsa"):
            raise ValueError("gradient_method must be 'finite_difference' or 'spsa'.")
        if self.spsa_delta <= 0.0:
            raise ValueError("spsa_delta must be positive.")
        if self.grad_tol <= 0.0:
            raise ValueError("grad_tol must be positive.")
        if self.line_search_min_alpha <= 0.0:
            raise ValueError("line_search_min_alpha must be positive.")
        if self.min_cost_improvement <= 0.0:
            raise ValueError("min_cost_improvement must be positive.")
        if self.trust_region_step_km_s2 <= 0.0:
            raise ValueError("trust_region_step_km_s2 must be positive.")

        signs = self._vector(self.state_signs, n=6)
        signs[signs == 0.0] = 1.0
        self.state_signs = np.sign(signs)

        self.target_rel_ric_rect = self._vector(self.target_rel_ric_rect, n=6)
        self.q_weights = self._vector(self.q_weights, n=6)
        self.terminal_weights = self._vector(self.terminal_weights, n=6)
        self.r_weights = self._vector(self.r_weights, n=3)
        self.rd_weights = self._vector(self.rd_weights, n=3)

        if np.any(self.q_weights < 0.0):
            raise ValueError("q_weights must be non-negative.")
        if np.any(self.terminal_weights < 0.0):
            raise ValueError("terminal_weights must be non-negative.")
        if np.any(self.r_weights <= 0.0):
            raise ValueError("r_weights must be positive.")
        if np.any(self.rd_weights < 0.0):
            raise ValueError("rd_weights must be non-negative.")

        self._u_guess_eci = np.zeros((self.horizon_steps, 3), dtype=float)
        self._u_prev_eci = np.zeros(3, dtype=float)
        self._rng = np.random.default_rng(0)

    @staticmethod
    def _vector(v: np.ndarray, n: int) -> np.ndarray:
        arr = np.array(v, dtype=float).reshape(-1)
        if arr.size == 1:
            arr = np.full(n, float(arr[0]))
        if arr.size != n:
            raise ValueError(f"Expected scalar or length-{n} vector.")
        return arr

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        i0, i1 = self.ric_curv_state_slice
        j0, j1 = self.chief_eci_state_slice
        if belief.state.size < max(i1, j1):
            return Command.zero()

        x_rel_curv = np.array(belief.state[i0:i1], dtype=float)
        x_tgt = np.array(belief.state[j0:j1], dtype=float)
        r_tgt = x_tgt[:3]
        v_tgt = x_tgt[3:]
        r0 = float(np.linalg.norm(r_tgt))
        if r0 <= 0.0:
            return Command.zero()

        c_ir = ric_dcm_ir_from_rv(r_tgt, v_tgt)
        x_rel_rect = ric_curv_to_rect(x_rel_curv, r0_km=r0)
        x_chaser0 = ric_rect_state_to_eci(x_rel_rect, r_tgt, v_tgt)

        x_rel_err = self.state_signs * x_rel_rect - self.target_rel_ric_rect
        a_seed_ric = -(self.seed_kp_pos * x_rel_err[:3] + self.seed_kd_vel * x_rel_err[3:])
        a_seed_eci = c_ir @ self._project_accel(a_seed_ric)

        if self._u_guess_eci.shape != (self.horizon_steps, 3):
            self._u_guess_eci = np.zeros((self.horizon_steps, 3), dtype=float)
        if not np.any(np.abs(self._u_guess_eci) > 0.0):
            self._u_guess_eci = self._build_seed_sequence(a_seed_eci)
        else:
            # Blend current warm-start with fresh local seed for robustness.
            self._u_guess_eci[0] = 0.5 * self._u_guess_eci[0] + 0.5 * a_seed_eci

        deadline_s = float("inf")
        if budget_ms > 0.0:
            deadline_s = perf_counter() + 0.95 * (budget_ms / 1e3)

        solve_t0 = perf_counter()
        u_opt, info = self._solve_mpc(
            x_chaser0=x_chaser0,
            x_target0=x_tgt,
            u_init=np.array(self._u_guess_eci, dtype=float),
            deadline_s=deadline_s,
        )
        solve_ms = 1e3 * (perf_counter() - solve_t0)
        if u_opt.size == 0:
            return Command.zero()

        u0 = self._project_accel(u_opt[0])
        self._u_prev_eci = u0
        self._u_guess_eci = self._shift_sequence(u_opt)

        a_cmd_ric = c_ir.T @ u0
        return Command(
            thrust_eci_km_s2=u0,
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "relative_orbit_mpc",
                "ric_curv_state_slice": [i0, i1],
                "chief_eci_state_slice": [j0, j1],
                "state_signs": self.state_signs.tolist(),
                "horizon_steps": int(self.horizon_steps),
                "step_dt_s": float(self.step_dt_s),
                "gradient_method": self.gradient_method,
                "accel_ric_km_s2": a_cmd_ric.tolist(),
                "seed_accel_ric_km_s2": (c_ir.T @ a_seed_eci).tolist(),
                "solve_time_ms": float(solve_ms),
                **info,
            },
        )

    def _solve_mpc(
        self,
        *,
        x_chaser0: np.ndarray,
        x_target0: np.ndarray,
        u_init: np.ndarray,
        deadline_s: float,
    ) -> tuple[np.ndarray, dict[str, float | int | str | list[float]]]:
        u = self._project_sequence(u_init)
        eval_count = 0
        iters = 0
        grad_norm_hist: list[float] = []
        accepted_alpha_hist: list[float] = []
        cost_hist: list[float] = []
        first_u_hist: list[list[float]] = []
        reason = "max_iterations_reached"
        timed_out = False
        trust_step = float(self.trust_region_step_km_s2)

        j = self._cost(x_chaser0=x_chaser0, x_target0=x_target0, u_seq=u)
        eval_count += 1
        cost_hist.append(float(j))
        first_u_hist.append(np.array(u[0], dtype=float).tolist())

        for it in range(self.max_iterations):
            if perf_counter() >= deadline_s:
                reason = "deadline_before_iteration"
                timed_out = True
                break
            iters = it + 1

            grad = np.zeros_like(u)
            if self.gradient_method == "finite_difference":
                for k in range(self.horizon_steps):
                    for j_idx in range(3):
                        if perf_counter() >= deadline_s:
                            reason = "deadline_during_gradient"
                            timed_out = True
                            break
                        up = np.array(u, dtype=float)
                        up[k, j_idx] += self.fd_epsilon
                        up = self._project_sequence(up)
                        jp = self._cost(x_chaser0=x_chaser0, x_target0=x_target0, u_seq=up)
                        eval_count += 1
                        grad[k, j_idx] = (jp - j) / self.fd_epsilon
                    if perf_counter() >= deadline_s:
                        break
            else:
                if perf_counter() >= deadline_s:
                    reason = "deadline_during_gradient"
                    timed_out = True
                else:
                    delta = self._rng.choice(np.array([-1.0, 1.0], dtype=float), size=u.shape)
                    up = self._project_sequence(u + self.spsa_delta * delta)
                    um = self._project_sequence(u - self.spsa_delta * delta)
                    jp = self._cost(x_chaser0=x_chaser0, x_target0=x_target0, u_seq=up)
                    jm = self._cost(x_chaser0=x_chaser0, x_target0=x_target0, u_seq=um)
                    eval_count += 2
                    grad = ((jp - jm) / (2.0 * self.spsa_delta)) * delta
            if timed_out:
                break

            grad_norm = float(np.linalg.norm(grad))
            grad_norm_hist.append(grad_norm)
            if grad_norm < self.grad_tol:
                reason = "grad_tol_reached"
                break

            # Use a normalized descent direction so line search scales a bounded trust-region step.
            step_dir = -grad / max(grad_norm, 1e-16)
            alpha = self.gradient_alpha
            improved = False
            while alpha >= self.line_search_min_alpha:
                if perf_counter() >= deadline_s:
                    reason = "deadline_during_line_search"
                    timed_out = True
                    break
                delta = alpha * trust_step * step_dir
                cand = self._project_sequence(u + delta)
                jc = self._cost(x_chaser0=x_chaser0, x_target0=x_target0, u_seq=cand)
                eval_count += 1
                if jc < (j - self.min_cost_improvement):
                    u = cand
                    j = jc
                    cost_hist.append(float(j))
                    first_u_hist.append(np.array(u[0], dtype=float).tolist())
                    accepted_alpha_hist.append(float(alpha))
                    improved = True
                    break
                alpha *= self.line_search_shrink
            if not improved:
                if not timed_out:
                    trust_step *= 0.5
                    if trust_step < 1e-9:
                        reason = "no_improvement_from_line_search"
                        break
                    reason = "line_search_shrunk_trust_region"
                    continue
                break

        return u, {
            "iterations": int(iters),
            "cost": float(j),
            "cost_evals": int(eval_count),
            "termination_reason": reason,
            "deadline_hit": bool(timed_out),
            "final_trust_step_km_s2": float(trust_step),
            "grad_norm_history": grad_norm_hist,
            "accepted_alpha_history": accepted_alpha_hist,
            "cost_history": cost_hist,
            "first_u_eci_history": first_u_hist,
        }

    def _cost(self, *, x_chaser0: np.ndarray, x_target0: np.ndarray, u_seq: np.ndarray) -> float:
        x_c = np.array(x_chaser0, dtype=float).reshape(6)
        x_t = np.array(x_target0, dtype=float).reshape(6)
        u_prev = np.array(self._u_prev_eci, dtype=float).reshape(3)
        u_seq = self._project_sequence(u_seq)

        j = 0.0
        err = np.zeros(6, dtype=float)
        for k in range(self.horizon_steps):
            u = u_seq[k]
            x_t = propagate_two_body_rk4(
                x_eci=x_t,
                dt_s=self.step_dt_s,
                mu_km3_s2=self.mu_km3_s2,
                accel_cmd_eci_km_s2=np.zeros(3, dtype=float),
            )
            x_c = propagate_two_body_rk4(
                x_eci=x_c,
                dt_s=self.step_dt_s,
                mu_km3_s2=self.mu_km3_s2,
                accel_cmd_eci_km_s2=u,
            )
            x_rel_rect = self._relative_rect_ric(x_chaser=x_c, x_target=x_t)
            err = self.state_signs * x_rel_rect - self.target_rel_ric_rect
            du = u - u_prev
            j += float(np.sum(self.q_weights * err * err))
            j += float(np.sum(self.r_weights * u * u))
            j += float(np.sum(self.rd_weights * du * du))
            u_prev = u

        j += float(np.sum(self.terminal_weights * err * err))
        return j

    @staticmethod
    def _relative_rect_ric(*, x_chaser: np.ndarray, x_target: np.ndarray) -> np.ndarray:
        return eci_relative_to_ric_rect(x_dep_eci=x_chaser, x_chief_eci=x_target)

    def _project_accel(self, u_eci: np.ndarray) -> np.ndarray:
        u = np.array(u_eci, dtype=float).reshape(3)
        norm_u = float(np.linalg.norm(u))
        if self.max_accel_km_s2 > 0.0 and norm_u > self.max_accel_km_s2:
            u *= self.max_accel_km_s2 / norm_u
        elif self.max_accel_km_s2 == 0.0:
            u = np.zeros(3, dtype=float)
        return u

    def _project_sequence(self, u_seq: np.ndarray) -> np.ndarray:
        u = np.array(u_seq, dtype=float).reshape(self.horizon_steps, 3)
        return np.vstack([self._project_accel(u[k]) for k in range(self.horizon_steps)])

    def _shift_sequence(self, u_seq: np.ndarray) -> np.ndarray:
        u = self._project_sequence(u_seq)
        if self.horizon_steps <= 1:
            return u
        shifted = np.zeros_like(u)
        shifted[:-1] = u[1:]
        shifted[-1] = u[-1]
        return shifted

    def _build_seed_sequence(self, a0_eci: np.ndarray) -> np.ndarray:
        seq = np.zeros((self.horizon_steps, 3), dtype=float)
        decay = 1.0
        for k in range(self.horizon_steps):
            seq[k] = self._project_accel(decay * np.array(a0_eci, dtype=float))
            decay *= self.seed_decay
        return seq
