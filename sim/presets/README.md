# Presets

This folder contains reusable baseline hardware presets for the new simulation framework.

## Files

- `rockets.py`
  - `BASIC_SSTO_ROCKET`
  - `BASIC_1ST_STAGE`
  - `BASIC_2ND_STAGE`
  - `BASIC_TWO_STAGE_STACK`
- `satellites.py`
  - `BASIC_SATELLITE`
- `thrusters.py`
  - `BASIC_CHEMICAL_BOTTOM_Z`
  - bottom `-Z` panel mount, centerline-aligned with CG (`x=y=0`)
- `attitude_control.py`
  - `BASIC_REACTION_WHEEL_TRIAD`
  - one wheel on each principal axis (`+X`, `+Y`, `+Z`)
- `simulation.py`
  - `build_sim_object_from_presets(...)`: one-call builder that maps satellite + thruster + attitude presets into a ready `SimObject`
  - `build_rocket_vehicle_from_presets(...)`: unified rocket vehicle object from SSTO or staged presets

## Quick Usage

```python
from sim.presets import build_sim_object_from_presets

sat = build_sim_object_from_presets(
    object_id="sat_01",
    dt_s=2.0,
    orbit_radius_km=6778.0,
)
```

Enable attitude knowledge estimation in the same builder:

```python
sat = build_sim_object_from_presets(
    object_id="sat_01",
    dt_s=2.0,
    enable_attitude_knowledge=True,
)
```
