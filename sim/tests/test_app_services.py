from __future__ import annotations

from sim.app.services import dump_config_text, get_default_config_path, list_available_configs, load_config, summarize_config, validate_config


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
