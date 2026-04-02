from __future__ import annotations

from sim.app.pointer_utils import (
    default_params_for_pointer,
    format_vector_text,
    format_yaml_text,
    normalize_form_value,
    parse_vector_text,
    parse_yaml_text,
    pointer_display_name,
    pointer_form_schema,
)


def test_pointer_helpers_cover_display_schema_and_defaults() -> None:
    pointer = {
        "module": "sim.mission.modules",
        "class_name": "MissionExecutiveStrategy",
    }
    schemas = {"MissionExecutiveStrategy": [{"key": "initial_mode", "kind": "text"}]}

    assert pointer_display_name(pointer) == "sim.mission.modules.MissionExecutiveStrategy"
    assert pointer_form_schema(pointer, schemas) == [{"key": "initial_mode", "kind": "text"}]

    defaults = default_params_for_pointer(pointer)

    assert defaults["initial_mode"] is None
    assert defaults["modes"] == []
    assert defaults["transitions"] == []


def test_value_normalization_and_vector_round_trip() -> None:
    field = {"key": "gain"}

    assert normalize_form_value(field, {"gain": 4.0}, {"gain": 1.0}) == 4.0
    assert normalize_form_value(field, {}, {"gain": 1.0}) == 1.0
    assert format_vector_text([1, 2, 3]) == "1, 2, 3"
    assert format_vector_text(None, length=3) == "0.0, 0.0, 0.0"
    assert parse_vector_text("1, 2, 3", length=3) == [1.0, 2.0, 3.0]


def test_yaml_text_helpers_round_trip() -> None:
    payload = {"alpha": 1, "beta": [2, 3]}

    text = format_yaml_text(payload)

    assert "alpha: 1" in text
    assert parse_yaml_text(text) == payload
    assert parse_yaml_text("") == []
