import unittest

from sim.config import (
    build_default_ops_orbit_propagator,
    default_env_for_profile,
    default_disturbance_config_for_profile,
    get_simulation_profile,
    profile_choices,
    resolve_dt_s,
)


class SimulationProfilesTests(unittest.TestCase):
    def test_profile_choices(self):
        self.assertEqual(set(profile_choices()), {"fast", "ops", "high_fidelity"})

    def test_resolve_dt_override(self):
        self.assertAlmostEqual(resolve_dt_s("ops"), 1.0)
        self.assertAlmostEqual(resolve_dt_s("ops", 3.5), 3.5)

    def test_ops_orbit_plugins_enabled(self):
        prop = build_default_ops_orbit_propagator("ops")
        plugin_names = {p.__name__ for p in prop.plugins}
        self.assertIn("j2_plugin", plugin_names)
        self.assertIn("j3_plugin", plugin_names)
        self.assertIn("j4_plugin", plugin_names)
        self.assertIn("drag_plugin", plugin_names)
        self.assertIn("srp_plugin", plugin_names)
        self.assertIn("third_body_sun_plugin", plugin_names)
        self.assertIn("third_body_moon_plugin", plugin_names)

    def test_fast_disturbance_defaults_off(self):
        cfg = default_disturbance_config_for_profile("fast")
        self.assertFalse(cfg.use_gravity_gradient)
        self.assertFalse(cfg.use_magnetic)
        self.assertFalse(cfg.use_drag)
        self.assertFalse(cfg.use_srp)

    def test_high_fidelity_integrator(self):
        p = get_simulation_profile("high_fidelity")
        self.assertEqual(p.orbit_integrator, "rkf78")
        self.assertEqual(p.kernel_integrator, "rkf78")

    def test_default_env_does_not_pin_static_sun_moon_positions(self):
        env = default_env_for_profile("ops")
        self.assertNotIn("sun_pos_eci_km", env)
        self.assertNotIn("moon_pos_eci_km", env)
        self.assertIn("jd_utc_start", env)


if __name__ == "__main__":
    unittest.main()
