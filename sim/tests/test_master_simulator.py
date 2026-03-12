from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from sim.master_simulator import run_master_simulation


class TestMasterSimulator(unittest.TestCase):
    def test_master_runner_executes_rocket_ascent_from_yaml(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")

        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "out"
            cfg_path = Path(td) / "cfg.yaml"
            cfg_path.write_text(
                """
scenario_name: "master_smoke"
rocket:
  enabled: true
  specs:
    payload_mass_kg: 50.0
    thrust_axis_body: [1.0, 0.0, 0.0]
  initial_state:
    launch_lat_deg: 28.5
    launch_lon_deg: -80.6
    launch_alt_km: 0.0
    launch_azimuth_deg: 90.0
target:
  enabled: false
chaser:
  enabled: false
simulator:
  scenario_type: "rocket_ascent"
  duration_s: 8.0
  dt_s: 1.0
  dynamics:
    rocket:
      atmosphere_model: "ussa1976"
  termination:
    earth_impact_enabled: true
    earth_radius_km: 6378.137
outputs:
  output_dir: "{outdir}"
  mode: "interactive"
  stats:
    save_json: true
monte_carlo:
  enabled: false
                """.strip().format(outdir=str(outdir).replace("\\", "\\\\")),
                encoding="utf-8",
            )
            res = run_master_simulation(cfg_path)
            self.assertEqual(res["scenario_name"], "master_smoke")
            self.assertIn("run", res)
            self.assertIn("objects", res["run"])
            self.assertTrue((outdir / "master_run_summary.json").exists())

    def test_master_runner_fails_fast_on_invalid_plugin_when_strict(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML not installed in this environment.")

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "bad_cfg.yaml"
            cfg_path.write_text(
                """
scenario_name: "bad_plugins"
rocket:
  enabled: true
  guidance:
    module: "sim.control.orbit.zero_controller"
    class_name: "ZeroController"
    params: {}
chaser:
  enabled: false
target:
  enabled: false
simulator:
  duration_s: 10.0
  dt_s: 1.0
  plugin_validation:
    strict: true
outputs:
  output_dir: "outputs/bad_plugins"
  mode: "save"
monte_carlo:
  enabled: false
                """.strip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _ = run_master_simulation(cfg_path)


if __name__ == "__main__":
    unittest.main()
