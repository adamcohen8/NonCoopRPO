from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any


@dataclass(frozen=True)
class PluginContract:
    methods_all: tuple[str, ...] = ()
    methods_any: tuple[str, ...] = ()
    allow_function: bool = False


_CONTRACTS = {
    "guidance": PluginContract(methods_all=("command",), allow_function=False),
    "guidance_generic": PluginContract(methods_all=(), methods_any=("command", "act"), allow_function=False),
    "orbit_control": PluginContract(methods_all=("act",), allow_function=False),
    "attitude_control": PluginContract(methods_all=("act",), allow_function=False),
    "bridge": PluginContract(methods_all=(), methods_any=("step", "start", "send_command", "receive_command"), allow_function=True),
    "mission_objective": PluginContract(methods_all=(), methods_any=("evaluate", "update", "check", "act"), allow_function=True),
}


def _validate_pointer(pointer: Any, contract: PluginContract, path: str) -> list[str]:
    errs: list[str] = []
    if pointer is None:
        return errs
    if not getattr(pointer, "module", None):
        errs.append(f"{path}: missing 'module'.")
        return errs

    try:
        mod = importlib.import_module(pointer.module)
    except Exception as ex:
        errs.append(f"{path}: failed to import module '{pointer.module}': {ex}")
        return errs

    class_name = getattr(pointer, "class_name", None)
    function = getattr(pointer, "function", None)
    params = dict(getattr(pointer, "params", {}) or {})

    if class_name:
        if not hasattr(mod, class_name):
            errs.append(f"{path}: class '{class_name}' not found in module '{pointer.module}'.")
            return errs
        cls = getattr(mod, class_name)
        try:
            obj = cls(**params)
        except Exception as ex:
            errs.append(f"{path}: failed to construct class '{class_name}' with params {params}: {ex}")
            return errs
        for m in contract.methods_all:
            if not hasattr(obj, m) or not callable(getattr(obj, m)):
                errs.append(f"{path}: class '{class_name}' missing required callable method '{m}'.")
        if contract.methods_any:
            if not any(hasattr(obj, m) and callable(getattr(obj, m)) for m in contract.methods_any):
                errs.append(f"{path}: class '{class_name}' must implement one of {list(contract.methods_any)}.")
        return errs

    if function:
        if not contract.allow_function:
            errs.append(f"{path}: function pointers are not allowed for this plugin type.")
            return errs
        if not hasattr(mod, function):
            errs.append(f"{path}: function '{function}' not found in module '{pointer.module}'.")
            return errs
        fn = getattr(mod, function)
        if not callable(fn):
            errs.append(f"{path}: '{function}' in module '{pointer.module}' is not callable.")
        return errs

    errs.append(f"{path}: must define either 'class_name' or 'function'.")
    return errs


def validate_scenario_plugins(cfg: Any) -> list[str]:
    errs: list[str] = []
    # Rocket
    if getattr(cfg.rocket, "enabled", False):
        errs.extend(_validate_pointer(getattr(cfg.rocket, "guidance", None), _CONTRACTS["guidance"], "rocket.guidance"))
        errs.extend(
            _validate_pointer(getattr(cfg.rocket, "orbit_control", None), _CONTRACTS["orbit_control"], "rocket.orbit_control")
        )
        errs.extend(
            _validate_pointer(
                getattr(cfg.rocket, "attitude_control", None), _CONTRACTS["attitude_control"], "rocket.attitude_control"
            )
        )
        rb = getattr(cfg.rocket, "bridge", None)
        if rb is not None and getattr(rb, "enabled", False):
            errs.extend(_validate_pointer(rb, _CONTRACTS["bridge"], "rocket.bridge"))
        for i, p in enumerate(getattr(cfg.rocket, "mission_objectives", []) or []):
            errs.extend(_validate_pointer(p, _CONTRACTS["mission_objective"], f"rocket.mission_objectives[{i}]"))

    # Chaser
    if getattr(cfg.chaser, "enabled", False):
        errs.extend(_validate_pointer(getattr(cfg.chaser, "guidance", None), _CONTRACTS["guidance_generic"], "chaser.guidance"))
        errs.extend(
            _validate_pointer(getattr(cfg.chaser, "orbit_control", None), _CONTRACTS["orbit_control"], "chaser.orbit_control")
        )
        errs.extend(
            _validate_pointer(
                getattr(cfg.chaser, "attitude_control", None), _CONTRACTS["attitude_control"], "chaser.attitude_control"
            )
        )
        cb = getattr(cfg.chaser, "bridge", None)
        if cb is not None and getattr(cb, "enabled", False):
            errs.extend(_validate_pointer(cb, _CONTRACTS["bridge"], "chaser.bridge"))
        for i, p in enumerate(getattr(cfg.chaser, "mission_objectives", []) or []):
            errs.extend(_validate_pointer(p, _CONTRACTS["mission_objective"], f"chaser.mission_objectives[{i}]"))

    # Target
    if getattr(cfg.target, "enabled", False):
        errs.extend(_validate_pointer(getattr(cfg.target, "guidance", None), _CONTRACTS["guidance_generic"], "target.guidance"))
        errs.extend(
            _validate_pointer(getattr(cfg.target, "orbit_control", None), _CONTRACTS["orbit_control"], "target.orbit_control")
        )
        errs.extend(
            _validate_pointer(
                getattr(cfg.target, "attitude_control", None), _CONTRACTS["attitude_control"], "target.attitude_control"
            )
        )
        tb = getattr(cfg.target, "bridge", None)
        if tb is not None and getattr(tb, "enabled", False):
            errs.extend(_validate_pointer(tb, _CONTRACTS["bridge"], "target.bridge"))
        for i, p in enumerate(getattr(cfg.target, "mission_objectives", []) or []):
            errs.extend(_validate_pointer(p, _CONTRACTS["mission_objective"], f"target.mission_objectives[{i}]"))
    return errs
