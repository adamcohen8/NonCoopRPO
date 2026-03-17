from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_curv_to_rect, ric_dcm_ir_from_rv


@dataclass
class HCWRelativeOrbitMPCController(Controller):
    """Relative-orbit MPC using HCW (CW) linear dynamics for prediction in rectangular RIC."""

    max_accel_km_s2: float
    horizon_time_s: float = 600.0
    default_model_dt_s: float = 10.0
    model_dt_s: float | None = None
    max_horizon_steps: int = 400
    mu_km3_s2: float = 398600.4418
    mean_motion_rad_s: float | None = None

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
    debug_store_iteration_history: bool = False

    _u_guess_ric: np.ndarray = field(init=False, repr=False)
    _u_prev_ric: np.ndarray = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _ad: np.ndarray = field(init=False, repr=False)
    _bd: np.ndarray = field(init=False, repr=False)
    _last_eval_t_s: float | None = field(default=None, init=False, repr=False)
    _last_model_dt_s: float = field(default=10.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_accel_km_s2 < 0.0:
            raise ValueError("max_accel_km_s2 must be non-negative.")
        if self.horizon_time_s <= 0.0:
            raise ValueError("horizon_time_s must be positive.")
        if self.default_model_dt_s <= 0.0:
            raise ValueError("default_model_dt_s must be positive.")
        if self.model_dt_s is not None and self.model_dt_s <= 0.0:
            raise ValueError("model_dt_s must be positive when provided.")
        if self.max_horizon_steps <= 0:
            raise ValueError("max_horizon_steps must be positive.")
        if self.mu_km3_s2 <= 0.0:
            raise ValueError("mu_km3_s2 must be positive.")
        if self.mean_motion_rad_s is not None and self.mean_motion_rad_s <= 0.0:
            raise ValueError("mean_motion_rad_s must be positive when provided.")
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

        self._u_guess_ric = np.zeros((1, 3), dtype=float)
        self._u_prev_ric = np.zeros(3, dtype=float)
        self._rng = np.random.default_rng(0)
        self._ad = np.eye(6, dtype=float)
        self._bd = np.vstack((np.zeros((3, 3), dtype=float), np.eye(3, dtype=float))) * self.default_model_dt_s
        self._last_model_dt_s = float(self.default_model_dt_s)

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

        n = float(self.mean_motion_rad_s) if self.mean_motion_rad_s is not None else float(np.sqrt(self.mu_km3_s2 / (r0**3)))
        if n <= 0.0 or (not np.isfinite(n)):
            return Command.zero()
        dt_model = self._resolve_model_dt(t_s)
        h_steps = int(np.clip(np.ceil(self.horizon_time_s / max(dt_model, 1e-9)), 1, self.max_horizon_steps))
        self._refresh_discrete_model(n, dt_model)

        c_ir = ric_dcm_ir_from_rv(r_tgt, v_tgt)
        x_rel_rect = ric_curv_to_rect(x_rel_curv, r0_km=r0)
        x_rel_err = self.state_signs * x_rel_rect - self.target_rel_ric_rect
        a_seed_ric = -(self.seed_kp_pos * x_rel_err[:3] + self.seed_kd_vel * x_rel_err[3:])
        a_seed_ric = self._project_accel(a_seed_ric)

        if self._u_guess_ric.shape != (h_steps, 3):
            self._u_guess_ric = np.zeros((h_steps, 3), dtype=float)
        if not np.any(np.abs(self._u_guess_ric) > 0.0):
            self._u_guess_ric = self._build_seed_sequence(a_seed_ric, h_steps=h_steps)
        else:
            self._u_guess_ric[0] = 0.5 * self._u_guess_ric[0] + 0.5 * a_seed_ric

        deadline_s = float("inf")
        if budget_ms > 0.0:
            deadline_s = perf_counter() + 0.95 * (budget_ms / 1e3)

        solve_t0 = perf_counter()
        u_opt, info = self._solve_mpc(
            x_rel0=np.array(x_rel_rect, dtype=float),
            u_init=np.array(self._u_guess_ric, dtype=float),
            h_steps=h_steps,
            deadline_s=deadline_s,
        )
        solve_ms = 1e3 * (perf_counter() - solve_t0)
        if u_opt.size == 0:
            return Command.zero()

        u0_ric = self._project_accel(u_opt[0])
        self._u_prev_ric = u0_ric
        self._u_guess_ric = self._shift_sequence(u_opt)
        u0_eci = c_ir @ u0_ric
        return Command(
            thrust_eci_km_s2=u0_eci,
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "relative_orbit_hcw_mpc",
                "ric_curv_state_slice": [i0, i1],
                "chief_eci_state_slice": [j0, j1],
                "state_signs": self.state_signs.tolist(),
                "horizon_time_s": float(self.horizon_time_s),
                "horizon_steps": int(h_steps),
                "step_dt_s": float(dt_model),
                "gradient_method": self.gradient_method,
                "mean_motion_rad_s": float(n),
                "accel_ric_km_s2": u0_ric.tolist(),
                "seed_accel_ric_km_s2": a_seed_ric.tolist(),
                "solve_time_ms": float(solve_ms),
                **info,
            },
        )

    def _resolve_model_dt(self, t_s: float) -> float:
        if self.model_dt_s is not None:
            dt_model = float(self.model_dt_s)
        elif self._last_eval_t_s is None:
            dt_model = float(self.default_model_dt_s)
        else:
            dt_model = float(t_s) - float(self._last_eval_t_s)
            if dt_model <= 1e-9 or not np.isfinite(dt_model):
                dt_model = float(self._last_model_dt_s)
        self._last_eval_t_s = float(t_s)
        self._last_model_dt_s = float(max(dt_model, 1e-9))
        return self._last_model_dt_s

    def _refresh_discrete_model(self, n: float, dt: float) -> None:
        nt = float(n * dt)
        c = float(np.cos(nt))
        s = float(np.sin(nt))
        n2 = float(n * n)

        # Exact CW STM (solved equations) with piecewise-constant acceleration over dt.
        self._ad = np.array(
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
        self._bd = np.array(
            [
                [(1.0 - c) / n2, 2.0 * (dt / n - s / n2), 0.0],
                [-2.0 * (dt / n - s / n2), 4.0 * (1.0 - c) / n2 - 1.5 * dt * dt, 0.0],
                [0.0, 0.0, (1.0 - c) / n2],
                [s / n, 2.0 * (1.0 - c) / n, 0.0],
                [-2.0 * (1.0 - c) / n, 4.0 * s / n - 3.0 * dt, 0.0],
                [0.0, 0.0, s / n],
            ],
            dtype=float,
        )

    def _solve_mpc(
        self,
        *,
        x_rel0: np.ndarray,
        u_init: np.ndarray,
        h_steps: int,
        deadline_s: float,
    ) -> tuple[np.ndarray, dict[str, float | int | str | list[float]]]:
        u = self._project_sequence(u_init, h_steps=h_steps)
        eval_count = 0
        iters = 0
        grad_norm_hist: list[float] = []
        accepted_alpha_hist: list[float] = []
        cost_hist: list[float] = []
        first_u_hist: list[list[float]] = []
        reason = "max_iterations_reached"
        timed_out = False
        trust_step = float(self.trust_region_step_km_s2)
        keep_hist = bool(self.debug_store_iteration_history)

        j = self._cost(x_rel0=x_rel0, u_seq=u, h_steps=h_steps)
        eval_count += 1
        if keep_hist:
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
                for k in range(h_steps):
                    for j_idx in range(3):
                        if perf_counter() >= deadline_s:
                            reason = "deadline_during_gradient"
                            timed_out = True
                            break
                        up = np.array(u, dtype=float)
                        up[k, j_idx] += self.fd_epsilon
                        up = self._project_sequence(up, h_steps=h_steps)
                        jp = self._cost(x_rel0=x_rel0, u_seq=up, h_steps=h_steps)
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
                    up = self._project_sequence(u + self.spsa_delta * delta, h_steps=h_steps)
                    um = self._project_sequence(u - self.spsa_delta * delta, h_steps=h_steps)
                    jp = self._cost(x_rel0=x_rel0, u_seq=up, h_steps=h_steps)
                    jm = self._cost(x_rel0=x_rel0, u_seq=um, h_steps=h_steps)
                    eval_count += 2
                    grad = ((jp - jm) / (2.0 * self.spsa_delta)) * delta
            if timed_out:
                break

            grad_norm = float(np.linalg.norm(grad))
            grad_norm_hist.append(grad_norm)
            if grad_norm < self.grad_tol:
                reason = "grad_tol_reached"
                break

            step_dir = -grad / max(grad_norm, 1e-16)
            alpha = self.gradient_alpha
            improved = False
            while alpha >= self.line_search_min_alpha:
                if perf_counter() >= deadline_s:
                    reason = "deadline_during_line_search"
                    timed_out = True
                    break
                delta = alpha * trust_step * step_dir
                cand = self._project_sequence(u + delta, h_steps=h_steps)
                jc = self._cost(x_rel0=x_rel0, u_seq=cand, h_steps=h_steps)
                eval_count += 1
                if jc < (j - self.min_cost_improvement):
                    u = cand
                    j = jc
                    if keep_hist:
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

        out: dict[str, float | int | str | list[float] | list[list[float]]] = {
            "iterations": int(iters),
            "cost": float(j),
            "cost_evals": int(eval_count),
            "termination_reason": reason,
            "deadline_hit": bool(timed_out),
            "final_trust_step_km_s2": float(trust_step),
        }
        if keep_hist:
            out["grad_norm_history"] = grad_norm_hist
            out["accepted_alpha_history"] = accepted_alpha_hist
            out["cost_history"] = cost_hist
            out["first_u_ric_history"] = first_u_hist
        return u, out

    def _cost(self, *, x_rel0: np.ndarray, u_seq: np.ndarray, h_steps: int) -> float:
        x = np.array(x_rel0, dtype=float).reshape(6)
        u_prev = np.array(self._u_prev_ric, dtype=float).reshape(3)
        u_seq = self._project_sequence(u_seq, h_steps=h_steps)

        j = 0.0
        err = np.zeros(6, dtype=float)
        for k in range(h_steps):
            u = u_seq[k]
            x = self._ad @ x + self._bd @ u
            err = self.state_signs * x - self.target_rel_ric_rect
            du = u - u_prev
            j += float(np.sum(self.q_weights * err * err))
            j += float(np.sum(self.r_weights * u * u))
            j += float(np.sum(self.rd_weights * du * du))
            u_prev = u
        j += float(np.sum(self.terminal_weights * err * err))
        return j

    def _project_accel(self, u_ric: np.ndarray) -> np.ndarray:
        u = np.array(u_ric, dtype=float).reshape(3)
        norm_u = float(np.linalg.norm(u))
        if self.max_accel_km_s2 > 0.0 and norm_u > self.max_accel_km_s2:
            u *= self.max_accel_km_s2 / norm_u
        elif self.max_accel_km_s2 == 0.0:
            u = np.zeros(3, dtype=float)
        return u

    def _project_sequence(self, u_seq: np.ndarray, *, h_steps: int) -> np.ndarray:
        u = np.array(u_seq, dtype=float).reshape(h_steps, 3)
        if self.max_accel_km_s2 == 0.0:
            return np.zeros_like(u)
        if self.max_accel_km_s2 < 0.0:
            return u
        norms = np.linalg.norm(u, axis=1, keepdims=True)
        scale = np.ones_like(norms)
        mask = norms > self.max_accel_km_s2
        scale[mask] = self.max_accel_km_s2 / np.maximum(norms[mask], 1e-16)
        return u * scale

    def _shift_sequence(self, u_seq: np.ndarray) -> np.ndarray:
        h_steps = int(np.array(u_seq, dtype=float).reshape(-1, 3).shape[0])
        u = self._project_sequence(u_seq, h_steps=h_steps)
        if h_steps <= 1:
            return u
        shifted = np.zeros_like(u)
        shifted[:-1] = u[1:]
        shifted[-1] = u[-1]
        return shifted

    def _build_seed_sequence(self, a0_ric: np.ndarray, *, h_steps: int) -> np.ndarray:
        seq = np.zeros((h_steps, 3), dtype=float)
        decay = 1.0
        for k in range(h_steps):
            seq[k] = self._project_accel(decay * np.array(a0_ric, dtype=float))
            decay *= self.seed_decay
        return seq
