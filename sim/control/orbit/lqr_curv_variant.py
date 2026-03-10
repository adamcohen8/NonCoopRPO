from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.control.orbit.lqr import HCWLQRController
from sim.core.interfaces import Controller
from sim.core.models import Command, StateBelief
from sim.utils.frames import ric_curv_to_rect, ric_dcm_ir_from_rv


@dataclass
class HCWCurvInputRectOutputController(Controller):
    """Pipeline-explicit HCW LQR variant.

    Flow:
    1) Curvilinear RIC state input
    2) Convert to rectangular RIC state
    3) Apply rectangular HCW LQR
    4) Convert rectangular RIC acceleration command to ECI for thrusting
    """

    base_lqr: HCWLQRController

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        i0, i1 = self.base_lqr.ric_curv_state_slice
        j0, j1 = self.base_lqr.chief_eci_state_slice
        if belief.state.size < max(i1, j1):
            return Command.zero()

        x_curv = np.array(belief.state[i0:i1], dtype=float)
        chief_eci = np.array(belief.state[j0:j1], dtype=float)
        r_chief = chief_eci[:3]
        v_chief = chief_eci[3:]
        r0 = float(np.linalg.norm(r_chief))
        if r0 <= 0.0:
            return Command.zero()

        # Stage 1: curvilinear -> rectangular RIC for HCW model.
        x_rect = ric_curv_to_rect(x_curv, r0_km=r0)
        x_rect_for_lqr = self.base_lqr.state_signs * x_rect

        # Stage 2: rectangular HCW LQR control (u_rect).
        u_rect = -self.base_lqr._k_gain @ x_rect_for_lqr
        u_norm = float(np.linalg.norm(u_rect))
        if u_norm > self.base_lqr.max_accel_km_s2 > 0.0:
            u_rect *= self.base_lqr.max_accel_km_s2 / u_norm

        # Stage 3: rectangular RIC -> ECI for thrust command.
        c_ir = ric_dcm_ir_from_rv(r_chief, v_chief)
        u_eci = c_ir @ u_rect

        return Command(
            thrust_eci_km_s2=u_eci,
            torque_body_nm=np.zeros(3),
            mode_flags={
                "mode": "hcw_lqr_curv_variant",
                "ric_curv_state_slice": [i0, i1],
                "chief_eci_state_slice": [j0, j1],
                "x_curv": x_curv.tolist(),
                "x_rect": x_rect.tolist(),
                "accel_rect_ric_km_s2": u_rect.tolist(),
                "accel_eci_km_s2": u_eci.tolist(),
            },
        )
