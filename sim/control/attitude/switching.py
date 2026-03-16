from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np

from sim.core.models import Command, StateBelief


def _construct_component(spec: Any) -> Any:
    if spec is None:
        return None
    if isinstance(spec, dict):
        module = spec.get("module")
        class_name = spec.get("class_name")
        params = dict(spec.get("params", {}) or {})
        if not module or not class_name:
            raise ValueError("Controller spec dict must include 'module' and 'class_name'.")
        mod = importlib.import_module(str(module))
        cls = getattr(mod, str(class_name))
        return cls(**params)
    return spec


@dataclass
class DetumbleThenSlewController:
    nominal: Any
    detumble: Any
    initial_mode: str = "nominal"  # nominal | detumble
    _mode: str = "nominal"

    def __post_init__(self) -> None:
        self.nominal = _construct_component(self.nominal)
        self.detumble = _construct_component(self.detumble)
        if self.nominal is None or not hasattr(self.nominal, "act"):
            raise ValueError("nominal controller must be provided and implement act().")
        if self.detumble is None or not hasattr(self.detumble, "act"):
            raise ValueError("detumble controller must be provided and implement act().")
        m = str(self.initial_mode).strip().lower()
        self._mode = "detumble" if m == "detumble" else "nominal"

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        m = str(mode).strip().lower()
        if m not in ("nominal", "detumble"):
            raise ValueError("mode must be 'nominal' or 'detumble'.")
        self._mode = m

    @staticmethod
    def _set_target_on_controller(ctrl: Any, q_des: np.ndarray, w_des: np.ndarray | None) -> None:
        if not hasattr(ctrl, "set_target"):
            return
        # Some controllers accept (q_des), others accept (q_des, w_des).
        if w_des is not None:
            try:
                ctrl.set_target(q_des, w_des)
                return
            except TypeError:
                pass
        ctrl.set_target(q_des)

    def set_target(self, desired_attitude_quat_bn: np.ndarray, desired_rate_body_rad_s: np.ndarray | None = None) -> None:
        self._set_target_on_controller(self.nominal, desired_attitude_quat_bn, desired_rate_body_rad_s)
        self._set_target_on_controller(self.detumble, desired_attitude_quat_bn, desired_rate_body_rad_s)

    def set_desired_ric_state(
        self,
        yaw_r_rad: float,
        roll_i_rad: float,
        pitch_c_rad: float,
        w_ric_rad_s: np.ndarray | None = None,
    ) -> None:
        if hasattr(self.nominal, "set_desired_ric_state"):
            self.nominal.set_desired_ric_state(yaw_r_rad, roll_i_rad, pitch_c_rad, w_ric_rad_s)
        if hasattr(self.detumble, "set_desired_ric_state"):
            self.detumble.set_desired_ric_state(yaw_r_rad, roll_i_rad, pitch_c_rad, w_ric_rad_s)

    def act(self, belief: StateBelief, t_s: float, budget_ms: float) -> Command:
        ctrl = self.detumble if self._mode == "detumble" else self.nominal
        cmd = ctrl.act(belief, t_s, budget_ms)
        flags = dict(cmd.mode_flags or {})
        flags["attitude_switch_mode"] = self._mode
        return Command(
            thrust_eci_km_s2=np.array(cmd.thrust_eci_km_s2, dtype=float),
            torque_body_nm=np.array(cmd.torque_body_nm, dtype=float),
            mode_flags=flags,
        )
