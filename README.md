# NonCooperativeRPO

Physics-based simulation toolkit for non-cooperative rendezvous/proximity operations (RPO) and ASAT-style orbital engagement scenarios.

## What It Does

- Launch-to-insertion simulation in ECI with gravity and atmospheric drag.
- Timing modes for launch window selection (`GO_NOW`, `WHEN_FEASIBLE`, `OPTIMAL`).
- Rocket guidance with:
  - feedback control,
  - predictive shooting control,
  - optional attitude-constrained thrust pointing (inertia + torque limits).
- Target/chaser propagation and relative motion analysis in curvilinear RIC.
- Satellite acceleration limits, delta-v accounting, and optional attitude-constrained actuation.

## Repository Layout

- `noncoop_rpo/`: core simulation library
- `examples/`: runnable scripts and visualizations

## Requirements

This project is pure Python and uses scientific Python packages. Install at minimum:

- `numpy`
- `matplotlib`

## Quick Start

From the repo root:

```bash
python examples/Launch_To_Insertion_RPO_Example.py
```

Predictive-shooting launch guidance variant:

```bash
python examples/Launch_To_Insertion_RPO_PredictiveShooting_Example.py
```

## Current Launch Termination Conditions

The launch simulation ends on exactly one of:

- `out_of_fuel`
- `insertion_orbit_achieved`
- `max_time_reached`

## Enabling Attitude Constraints

Attitude limits are **off by default** for both rockets and satellites.

### Rocket

```python
from noncoop_rpo import Rocket
import numpy as np

rocket = Rocket(
    # existing fields...
    attitude_control_enabled=True,
    inertia_body_kg_m2=np.diag([9000.0, 7000.0, 6500.0]),
    max_torque_nm=np.array([1500.0, 1200.0, 1200.0]),
)
```

### Satellite

```python
from noncoop_rpo import SatParams
import numpy as np

sat = SatParams(
    # existing fields...
    attitude_control_enabled=True,
    inertia_body_kg_m2=np.diag([120.0, 100.0, 80.0]),
    max_torque_nm=np.array([0.5, 0.5, 0.5]),
)
```

## Notes

- Launch guidance currently supports both classic feedback and predictive shooting (`Rocket.guidance_mode`).
- Example scripts include trajectory, relative-motion, altitude, and thrust-vs-time plots.
