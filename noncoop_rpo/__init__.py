"""Legacy compatibility package.

The project now uses the modular `sim/` framework. This module re-exports
that surface to keep old import paths from failing immediately.
"""

import sim as _sim
from sim import *  # noqa: F401,F403

__all__ = _sim.__all__
