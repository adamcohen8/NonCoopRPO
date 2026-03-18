from __future__ import annotations

from sim.app.io import DEFAULT_CONFIG_PATH, dump_yaml_text, ensure_sections, list_config_files, load_config_dict, validate_config_dict


def test_list_config_files_includes_template() -> None:
    configs = list_config_files()
    assert DEFAULT_CONFIG_PATH in configs


def test_load_and_validate_template_config() -> None:
    cfg_dict = ensure_sections(load_config_dict(DEFAULT_CONFIG_PATH))
    cfg = validate_config_dict(cfg_dict)
    assert cfg.scenario_name == "everything_on_template"
    assert cfg.outputs.output_dir


def test_dump_yaml_text_round_trip_smoke(tmp_path) -> None:
    cfg_dict = {
        "scenario_name": "gui_test",
        "rocket": {"enabled": False},
        "chaser": {"enabled": False},
        "target": {"enabled": True},
        "simulator": {"scenario_type": "auto", "duration_s": 10.0, "dt_s": 1.0},
        "outputs": {"output_dir": "outputs/gui_test", "mode": "save"},
        "monte_carlo": {"enabled": False, "iterations": 1},
    }
    rendered = dump_yaml_text(ensure_sections(cfg_dict))
    path = tmp_path / "gui_test.yaml"
    path.write_text(rendered, encoding="utf-8")
    reparsed = validate_config_dict(ensure_sections(load_config_dict(path)))
    assert reparsed.scenario_name == "gui_test"
