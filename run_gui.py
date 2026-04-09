from __future__ import annotations

from importlib import import_module


def main() -> int:
    try:
        gui_main = import_module("sim.gui.main")
    except ModuleNotFoundError as exc:
        if str(getattr(exc, "name", "") or "").split(".", 1)[0] == "PySide6":
            raise SystemExit(
                "PySide6 is not installed. Install GUI dependencies with "
                "`python -m pip install -r requirements-gui.txt`."
            ) from exc
        raise
    return int(gui_main.main())


if __name__ == "__main__":
    raise SystemExit(main())
