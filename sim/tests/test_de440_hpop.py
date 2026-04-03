import numpy as np

from sim.dynamics.orbit.de440_hpop import _cheb3d, _eval_body, _find_coeff_row, mjd_tt_to_mjd_tdb


def test_cheb3d_constant_vector():
    out = _cheb3d(1.5, 3, 1.0, 2.0, np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))
    assert np.allclose(out, np.array([1.0, 2.0, 3.0]))


def test_find_coeff_row_selects_covering_interval():
    pc = np.array([[0.0, 32.0, 1.0], [32.0, 64.0, 2.0]], dtype=float)
    row = _find_coeff_row(pc, 40.0)
    assert float(row[2]) == 2.0


def test_eval_body_constant_segment_returns_scaled_constant():
    row = np.zeros(1000, dtype=float)
    row[0] = 0.0
    row[1] = 32.0
    row[440] = 1.0
    row[453] = 2.0
    row[466] = 3.0
    out = _eval_body(row, 2.0, "moon")
    assert np.allclose(out, np.array([1000.0, 2000.0, 3000.0]))


def test_mjd_tt_to_tdb_is_close():
    out = mjd_tt_to_mjd_tdb(51544.5)
    assert abs(out - 51544.5) < 1e-4
