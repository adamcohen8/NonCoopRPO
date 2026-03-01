from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import DynamicsModel
from sim.core.models import Command, StateTruth
from sim.dynamics.attitude.disturbances import DisturbanceTorqueModel
from sim.dynamics.attitude.rigid_body import propagate_attitude_euler
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator


@dataclass(frozen=True)
class OrbitalAttitudeDynamics(DynamicsModel):
    mu_km3_s2: float
    inertia_kg_m2: np.ndarray
    disturbance_model: DisturbanceTorqueModel | None = None
    area_m2: float = 1.0
    cd: float = 2.2
    cr: float = 1.2
    orbit_propagator: OrbitPropagator = field(default_factory=lambda: OrbitPropagator(integrator="rk4"))

    def step(self, state: StateTruth, command: Command, env: dict, dt_s: float) -> StateTruth:
        x_orbit = np.hstack((state.position_eci_km, state.velocity_eci_km_s))
        orbit_ctx = OrbitContext(
            mu_km3_s2=self.mu_km3_s2,
            mass_kg=state.mass_kg,
            area_m2=self.area_m2,
            cd=self.cd,
            cr=self.cr,
        )
        x_orbit_next = self.orbit_propagator.propagate(
            x_eci=x_orbit,
            dt_s=dt_s,
            t_s=state.t_s,
            command_accel_eci_km_s2=command.thrust_eci_km_s2,
            env=env,
            ctx=orbit_ctx,
        )

        disturbance_torque = (
            np.zeros(3) if self.disturbance_model is None else self.disturbance_model.total_torque_body_nm(state, env)
        )
        total_torque = command.torque_body_nm + disturbance_torque

        q_next, w_next = propagate_attitude_euler(
            quat_bn=state.attitude_quat_bn,
            omega_body_rad_s=state.angular_rate_body_rad_s,
            inertia_kg_m2=self.inertia_kg_m2,
            torque_body_nm=total_torque,
            dt_s=dt_s,
        )
        delta_mass_kg = float(command.mode_flags.get("delta_mass_kg", 0.0))
        mass_next = max(0.0, state.mass_kg - delta_mass_kg)

        return StateTruth(
            position_eci_km=x_orbit_next[:3],
            velocity_eci_km_s=x_orbit_next[3:],
            attitude_quat_bn=q_next,
            angular_rate_body_rad_s=w_next,
            mass_kg=mass_next,
            t_s=state.t_s + dt_s,
        )
