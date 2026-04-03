from __future__ import annotations

from unittest.mock import patch

import numpy as np

from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.estimation.attitude_ekf import AttitudeEKFEstimator
from sim.estimation.orbit_ekf import OrbitEKFEstimator
from sim.estimation.orbit_ukf import OrbitUKFEstimator
from sim.utils.quaternion import normalize_quaternion


def test_orbit_ekf_ignores_partial_measurements_instead_of_zero_padding() -> None:
    estimator = OrbitEKFEstimator(
        mu_km3_s2=398600.4418,
        dt_s=1.0,
        process_noise_diag=np.ones(6) * 1e-10,
        meas_noise_diag=np.ones(6) * 1e-6,
    )
    belief = StateBelief(
        state=np.array([7000.0, 0.0, 0.0, 0.0, 7.546049108166282, 0.0], dtype=float),
        covariance=np.eye(6) * 1e-3,
        last_update_t_s=0.0,
    )

    predicted = estimator.update(belief, None, 1.0)
    partial = estimator.update(belief, Measurement(vector=np.array([12.0, -0.1]), t_s=1.0), 1.0)

    assert np.allclose(partial.state, predicted.state)
    assert np.allclose(partial.covariance, predicted.covariance)
    assert estimator.last_update_diagnostics is not None
    assert estimator.last_update_diagnostics.measurement_available is True
    assert estimator.last_update_diagnostics.update_applied is False


def test_orbit_ekf_update_avoids_np_inv_and_preserves_symmetric_covariance() -> None:
    estimator = OrbitEKFEstimator(
        mu_km3_s2=398600.4418,
        dt_s=1.0,
        process_noise_diag=np.ones(6) * 1e-10,
        meas_noise_diag=np.ones(6) * 1e-6,
    )
    belief = StateBelief(
        state=np.array([7000.0, 1.0, -0.5, -0.001, 7.5460, 0.002], dtype=float),
        covariance=np.array(
            [
                [1e-3, 2e-5, 0.0, 0.0, 0.0, 0.0],
                [2e-5, 1.1e-3, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 9e-4, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1e-5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.2e-5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 8e-6],
            ],
            dtype=float,
        ),
        last_update_t_s=0.0,
    )
    measurement = Measurement(vector=belief.state + np.array([1e-3, -2e-3, 1e-3, 1e-6, -1e-6, 2e-6]), t_s=1.0)

    with patch("numpy.linalg.inv", side_effect=AssertionError("np.linalg.inv should not be used")):
        updated = estimator.update(belief, measurement, 1.0)

    assert np.allclose(updated.covariance, updated.covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(updated.covariance) >= -1e-12)


def test_attitude_ekf_update_avoids_np_inv_and_preserves_symmetric_covariance() -> None:
    estimator = AttitudeEKFEstimator(
        dt_s=0.1,
        inertia_kg_m2=np.diag([10.0, 12.0, 8.0]),
        process_noise_diag=np.ones(7) * 1e-8,
        meas_noise_diag=np.ones(7) * 1e-6,
    )
    belief = StateBelief(
        state=np.hstack((normalize_quaternion(np.array([1.0, 0.01, -0.02, 0.0])), np.array([0.01, -0.02, 0.03]))),
        covariance=np.eye(7) * 1e-3,
        last_update_t_s=0.0,
    )
    measurement = Measurement(
        vector=np.hstack((normalize_quaternion(np.array([1.0, 0.012, -0.018, 0.002])), np.array([0.011, -0.019, 0.029]))),
        t_s=0.1,
    )

    with patch("numpy.linalg.inv", side_effect=AssertionError("np.linalg.inv should not be used")):
        updated = estimator.update(belief, measurement, 0.1)

    assert np.allclose(updated.covariance, updated.covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(updated.covariance) >= -1e-12)


def test_orbit_ukf_update_avoids_np_inv_and_preserves_symmetric_covariance() -> None:
    estimator = OrbitUKFEstimator(
        propagator=OrbitPropagator(),
        context=OrbitContext(mu_km3_s2=398600.4418, mass_kg=100.0),
        dt_s=1.0,
        process_noise_diag=np.ones(6) * 1e-10,
        meas_noise_diag=np.ones(6) * 1e-6,
    )
    belief = StateBelief(
        state=np.array([7000.0, 0.5, -0.2, -0.001, 7.5458, 0.001], dtype=float),
        covariance=np.eye(6) * 1e-3,
        last_update_t_s=0.0,
    )
    measurement = Measurement(vector=belief.state + np.array([2e-3, -1e-3, 1e-3, 2e-6, -1e-6, 1e-6]), t_s=1.0)

    with patch("numpy.linalg.inv", side_effect=AssertionError("np.linalg.inv should not be used")):
        updated = estimator.update(belief, measurement, 1.0)

    assert np.allclose(updated.covariance, updated.covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(updated.covariance) >= -1e-12)
