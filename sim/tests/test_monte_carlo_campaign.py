from __future__ import annotations

import unittest

from sim.scenarios.monte_carlo import compute_threshold_probabilities


class TestMonteCarloCampaign(unittest.TestCase):
    def test_threshold_probability_and_confidence_interval(self):
        rows = compute_threshold_probabilities(
            min_separations=[0.05, 0.08, 0.12, 0.20],
            thresholds_km=[0.1],
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.threshold_km, 0.1)
        self.assertEqual(row.success_count, 2)
        self.assertEqual(row.failure_count, 2)
        self.assertAlmostEqual(row.probability, 0.5, places=12)
        self.assertLess(row.confidence_interval_95["low"], 0.5)
        self.assertGreater(row.confidence_interval_95["high"], 0.5)
        self.assertAlmostEqual(row.confidence_interval_95["low"], 0.15003898915214947, places=12)
        self.assertAlmostEqual(row.confidence_interval_95["high"], 0.8499610108478506, places=12)


if __name__ == "__main__":
    unittest.main()
