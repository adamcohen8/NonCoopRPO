from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dynamics import two_body_deriv
from .frames import eci2hcw_curv
from .integrators import rk4_step
from .knowledge import KnowledgeModel
from .satellite import Satellite


@dataclass(frozen=True)
class Scenario:
    dt_s: float
    steps: int


@dataclass
class SimLogger:
    t_s: np.ndarray
    target_x_eci: np.ndarray
    chaser_x_eci: np.ndarray
    rel_ric: np.ndarray
    obs_chaser_ric_curv: np.ndarray
    obs_target_ric_curv: np.ndarray
    detected_by_chaser: np.ndarray
    detected_by_target: np.ndarray
    u_ric: np.ndarray
    u_target_ric: np.ndarray
    u_mag: np.ndarray
    u_target_mag: np.ndarray
    dv_cum_km_s: np.ndarray
    dv_target_cum_km_s: np.ndarray
    target_dv_remaining_km_s: np.ndarray
    chaser_dv_remaining_km_s: np.ndarray

    @classmethod
    def allocate(cls, steps: int) -> "SimLogger":
        n = steps + 1
        return cls(
            t_s=np.zeros(n),
            target_x_eci=np.zeros((n, 6)),
            chaser_x_eci=np.zeros((n, 6)),
            rel_ric=np.zeros((n, 6)),
            obs_chaser_ric_curv=np.zeros((n, 6)),
            obs_target_ric_curv=np.zeros((n, 6)),
            detected_by_chaser=np.zeros(n, dtype=bool),
            detected_by_target=np.zeros(n, dtype=bool),
            u_ric=np.zeros((n, 3)),
            u_target_ric=np.zeros((n, 3)),
            u_mag=np.zeros(n),
            u_target_mag=np.zeros(n),
            dv_cum_km_s=np.zeros(n),
            dv_target_cum_km_s=np.zeros(n),
            target_dv_remaining_km_s=np.zeros(n),
            chaser_dv_remaining_km_s=np.zeros(n),
        )


class Simulator:
    def __init__(self, scenario: Scenario, knowledge_model: Optional[KnowledgeModel] = None):
        self.scenario = scenario
        self.knowledge_model = knowledge_model or KnowledgeModel()

    def run(self, target: Satellite, chaser: Satellite) -> SimLogger:
        log = SimLogger.allocate(self.scenario.steps)
        log.target_x_eci[0, :] = target.state.x_eci
        log.chaser_x_eci[0, :] = chaser.state.x_eci
        log.rel_ric[0, :] = eci2hcw_curv(target.state.x_eci, chaser.state.x_eci)
        obs_chaser, det_chaser = self.knowledge_model.observe(chaser.state.x_eci, target.state.x_eci)
        obs_target, det_target = self.knowledge_model.observe(target.state.x_eci, chaser.state.x_eci)
        log.obs_chaser_ric_curv[0, :] = obs_chaser if obs_chaser is not None else np.full(6, np.nan)
        log.obs_target_ric_curv[0, :] = obs_target if obs_target is not None else np.full(6, np.nan)
        log.detected_by_chaser[0] = det_chaser
        log.detected_by_target[0] = det_target
        log.target_dv_remaining_km_s[0] = target.state.dv_remaining_km_s
        log.chaser_dv_remaining_km_s[0] = chaser.state.dv_remaining_km_s

        for k in range(self.scenario.steps):
            t = k * self.scenario.dt_s
            obs_chaser, det_chaser = self.knowledge_model.observe(chaser.state.x_eci, target.state.x_eci)
            obs_target, det_target = self.knowledge_model.observe(target.state.x_eci, chaser.state.x_eci)

            if det_chaser:
                u_ric = chaser.command_accel_ric(t, obs_chaser, self.scenario.dt_s)
            else:
                u_ric = np.zeros(3)

            if det_target:
                u_target_ric = target.command_accel_ric(t, obs_target, self.scenario.dt_s)
            else:
                u_target_ric = np.zeros(3)

            from .frames import eci_to_rsw_dcm

            rsw_target = eci_to_rsw_dcm(target.state.x_eci[:3], target.state.x_eci[3:])
            rsw_chaser = eci_to_rsw_dcm(chaser.state.x_eci[:3], chaser.state.x_eci[3:])

            a_chaser_eci = rsw_target @ u_ric
            a_target_eci = rsw_chaser @ u_target_ric

            target.state.x_eci = rk4_step(
                two_body_deriv,
                target.state.x_eci,
                self.scenario.dt_s,
                target.params.mu,
                a_target_eci,
            )
            chaser.state.x_eci = rk4_step(
                two_body_deriv,
                chaser.state.x_eci,
                self.scenario.dt_s,
                chaser.params.mu,
                a_chaser_eci,
            )

            target.state.t = t + self.scenario.dt_s
            chaser.state.t = t + self.scenario.dt_s
            log.t_s[k + 1] = t + self.scenario.dt_s
            log.target_x_eci[k + 1, :] = target.state.x_eci
            log.chaser_x_eci[k + 1, :] = chaser.state.x_eci
            log.rel_ric[k + 1, :] = eci2hcw_curv(target.state.x_eci, chaser.state.x_eci)
            log.obs_chaser_ric_curv[k + 1, :] = obs_chaser if obs_chaser is not None else np.full(6, np.nan)
            log.obs_target_ric_curv[k + 1, :] = obs_target if obs_target is not None else np.full(6, np.nan)
            log.detected_by_chaser[k + 1] = det_chaser
            log.detected_by_target[k + 1] = det_target
            log.u_ric[k + 1, :] = u_ric
            log.u_target_ric[k + 1, :] = u_target_ric
            log.u_mag[k + 1] = np.linalg.norm(u_ric)
            log.u_target_mag[k + 1] = np.linalg.norm(u_target_ric)
            log.dv_cum_km_s[k + 1] = log.dv_cum_km_s[k] + log.u_mag[k + 1] * self.scenario.dt_s
            log.dv_target_cum_km_s[k + 1] = (
                log.dv_target_cum_km_s[k] + log.u_target_mag[k + 1] * self.scenario.dt_s
            )
            log.target_dv_remaining_km_s[k + 1] = target.state.dv_remaining_km_s
            log.chaser_dv_remaining_km_s[k + 1] = chaser.state.dv_remaining_km_s

        return log
