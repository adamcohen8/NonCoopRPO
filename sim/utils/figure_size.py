from __future__ import annotations

MAX_FIGURE_HEIGHT_IN = 7.0


def cap_figsize(width: float, height: float) -> tuple[float, float]:
    return float(width), float(min(height, MAX_FIGURE_HEIGHT_IN))
