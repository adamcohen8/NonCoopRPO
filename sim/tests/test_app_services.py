from __future__ import annotations

from pathlib import Path
import tempfile

from sim.app.services import dump_config_text, get_default_config_path, get_gui_capabilities, list_available_configs, load_config, summarize_config, validate_config
from sim.app.services import run_config_via_api


def test_default_config_available() -> None:
    assert get_default_config_path() in list_available_configs()


def test_load_validate_and_summarize_default_config() -> None:
    cfg_dict = load_config(get_default_config_path())
    cfg = validate_config(cfg_dict)
    summary = summarize_config(cfg)
    assert summary.scenario_name == cfg.scenario_name
    assert summary.output_dir == cfg.outputs.output_dir


def test_dump_config_text_contains_scenario_name() -> None:
    cfg_dict = load_config(get_default_config_path())
    text = dump_config_text(cfg_dict)
    assert "scenario_name" in text


def test_gui_capabilities_reflect_backend_catalog() -> None:
    caps = get_gui_capabilities()

    assert "BASIC_SATELLITE" in caps.satellite_presets
    assert "BASIC_TWO_STAGE_STACK" in caps.rocket_preset_stacks
    assert caps.analysis_study_types == [("monte_carlo", "Monte Carlo"), ("sensitivity", "Sensitivity")]
    assert caps.sensitivity_methods == [("one_at_a_time", "One-at-a-Time"), ("lhs", "Latin Hypercube")]
    assert caps.monte_carlo_modes == ["choice", "uniform", "normal"]
    assert caps.monte_carlo_lhs_modes == ["uniform", "normal"]
    assert caps.chaser_init_modes == ["rocket_deployment", "relative_ric_rect", "relative_ric_curv"]
    assert caps.analysis_ui_profiles["monte_carlo"].inputs_title == "Monte Carlo Variations"
    assert caps.analysis_ui_profiles["sensitivity_lhs"].mode_label == "Distribution"
    assert "One-at-a-time sensitivity" in caps.analysis_ui_profiles["sensitivity_one_at_a_time"].help_text
    assert "rocket_fuel_remaining" in caps.figure_ids
    assert "ground_track_multi" in caps.animation_types
    assert "attitude_ric_thruster" in caps.animation_types
    assert "battlespace_dashboard" in caps.animation_types
    assert "target_reference_ric_curv_3d" in caps.animation_types
    assert "target_reference_ric_curv_2d" in caps.animation_types
    assert "target_reference_ric_curv_2d_ri" in caps.animation_types
    assert "target_reference_ric_curv_2d_ic" in caps.animation_types
    assert "target_reference_ric_curv_2d_rc" in caps.animation_types
    assert any(label == "Open Loop Pitch Program" for label, _ in caps.base_guidance_options["rocket"])
    assert any(label == "Relative Orbit MPC" for label, _ in caps.orbit_control_options["chaser"])
    assert any(label == "HCW LQR (No Radial Burn)" for label, _ in caps.orbit_control_options["chaser"])
    assert any(label == "HCW Manual Gain (No Radial Burn)" for label, _ in caps.orbit_control_options["chaser"])
    assert any(label == "HCW Relative MPC (In/Cross Track Only)" for label, _ in caps.orbit_control_options["chaser"])
    assert caps.monte_carlo_parameter_categories["Environment"][0][1] == "simulator.environment.atmosphere_env.solar_flux_f107"
    assert any(path == "target.initial_state.coes.true_anomaly_deg" for _, path in caps.monte_carlo_parameter_categories["Target Orbit"])
    schema = caps.parameter_form_schemas["RelativeOrbitMPCController"]
    assert any(field["key"] == "gradient_method" and field["kind"] == "choice" for field in schema)
    assert caps.parameter_form_schemas["MissionExecutiveStrategy"][1]["kind"] == "yaml"
    assert any(
        field["key"] == "r_weights" and field.get("length") == 2
        for field in caps.parameter_form_schemas["HCWNoRadialLQRController"]
    )
    assert any(
        field["key"] == "k_gain" and field.get("length") == 12
        for field in caps.parameter_form_schemas["HCWNoRadialManualController"]
    )
    assert any(
        field["key"] == "rd_weights" and field.get("length") == 2
        for field in caps.parameter_form_schemas["HCWInTrackCrossTrackMPCController"]
    )


def test_run_config_via_api_executes_single_run() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "gui_service_run.yaml"
        cfg_path.write_text(
            dump_config_text(
                {
                    "scenario_name": "gui_service_run",
                    "rocket": {"enabled": False},
                    "chaser": {"enabled": False},
                    "target": {
                        "enabled": True,
                        "specs": {"mass_kg": 100.0},
                        "initial_state": {
                            "position_eci_km": [7000.0, 0.0, 0.0],
                            "velocity_eci_km_s": [0.0, 7.5, 0.0],
                        },
                    },
                    "simulator": {
                        "duration_s": 2.0,
                        "dt_s": 1.0,
                        "termination": {"earth_impact_enabled": False},
                        "dynamics": {"attitude": {"enabled": False}},
                    },
                    "outputs": {
                        "output_dir": str(Path(tmpdir) / "outputs"),
                        "mode": "save",
                        "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
                        "plots": {"enabled": False, "figure_ids": []},
                        "animations": {"enabled": False, "types": []},
                    },
                    "monte_carlo": {"enabled": False, "iterations": 1},
                    "analysis": {"enabled": False},
                }
            ),
            encoding="utf-8",
        )

        result = run_config_via_api(cfg_path)

        assert result.returncode == 0
        assert result.scenario_name == "gui_service_run"
        assert "Scenario: gui_service_run" in result.stdout
