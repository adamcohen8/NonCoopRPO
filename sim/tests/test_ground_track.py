import unittest

import numpy as np

from sim.utils.ground_track import ground_track_from_eci_history, split_ground_track_dateline


class GroundTrackTests(unittest.TestCase):
    def test_ground_track_shapes_and_ranges(self):
        t = np.array([0.0, 10.0, 20.0], dtype=float)
        r = np.array(
            [
                [7000.0, 0.0, 0.0],
                [0.0, 7000.0, 0.0],
                [-7000.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        lat, lon, alt = ground_track_from_eci_history(r_eci_hist_km=r, t_s=t, jd_utc_start=2451545.0)
        self.assertEqual(lat.shape, (3,))
        self.assertEqual(lon.shape, (3,))
        self.assertEqual(alt.shape, (3,))
        self.assertTrue(np.all(lat <= 90.0) and np.all(lat >= -90.0))
        self.assertTrue(np.all(lon <= 180.0) and np.all(lon >= -180.0))

    def test_split_dateline_inserts_nans(self):
        lon = np.array([170.0, 179.0, -179.0, -170.0], dtype=float)
        lat = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        lo, la = split_ground_track_dateline(lon_deg=lon, lat_deg=lat, jump_threshold_deg=180.0)
        self.assertGreater(lo.size, lon.size)
        self.assertTrue(np.isnan(lo).any())
        self.assertTrue(np.isnan(la).any())


if __name__ == "__main__":
    unittest.main()
