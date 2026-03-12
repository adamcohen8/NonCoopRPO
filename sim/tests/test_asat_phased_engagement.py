import unittest

import numpy as np

from sim.scenarios import ASATPhasedScenarioConfig, AgentStrategyConfig, run_asat_phased_engagement


class TestASATPhasedEngagement(unittest.TestCase):
    def test_chaser_deploys_and_tracks(self):
        cfg = ASATPhasedScenarioConfig(
            dt_s=2.0,
            duration_s=120.0,
            deploy_time_s=20.0,
            chaser_deploy_dv_body_m_s=np.array([1.0, 0.0, 0.0]),
            seed=5,
        )
        out = run_asat_phased_engagement(
            cfg,
            target_strategy=AgentStrategyConfig(mode="coast", max_accel_km_s2=0.0, target_id=cfg.rocket_id),
            chaser_strategy=AgentStrategyConfig(mode="knowledge_pursuit", max_accel_km_s2=1e-6, target_id=cfg.target_id),
        )
        self.assertTrue(out["chaser_deployed"])
        self.assertGreaterEqual(out["chaser_deploy_index"], 0)
        chaser = out["truth_by_object"][cfg.chaser_id]
        self.assertTrue(np.any(np.isfinite(chaser[:, 0])))
        self.assertTrue(np.isfinite(out["min_chaser_target_range_km"]))


if __name__ == "__main__":
    unittest.main()
