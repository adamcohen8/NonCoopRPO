# NonCooperativeRPO

Modular orbital engagement simulation framework for SIL/HIL-style closed-loop experimentation.

## Current Status

The roadmap architecture is implemented across simulation kernel, dynamics, sensing/estimation, controls, metrics, and example harnesses.

## Implemented Capabilities

### Simulation Kernel
- Deterministic step order in `sim/core/kernel.py`:
  1. truth propagation
  2. sensor measurement
  3. estimator update
  4. controller execution with runtime budget check
  5. actuator application
  6. logging
- Arbitrary number of simulated objects.
- Runtime budget enforcement with skip-to-zero-command behavior.
- Per-object runtime and skip logging.

### Orbital Dynamics
- Two-body ECI propagation.
- Optional perturbation plugins:
  - J2
  - drag
  - solar radiation pressure
  - third-body Moon/Sun
- Fixed-step and adaptive propagation options through `OrbitPropagator`.

### Attitude Dynamics
- Quaternion + body-rate rigid body dynamics.
- Optional disturbance torques:
  - gravity-gradient
  - magnetic dipole
  - drag torque
  - SRP torque

### Actuators
- Orbital actuator:
  - acceleration saturation
  - throttle slew-rate limiting
  - minimum impulse-bit behavior
  - optional lag
  - propellant mass depletion
- Attitude actuator:
  - reaction wheel torque/momentum limits
  - magnetorquer clamping proxy
  - thruster pulse quantization

### Sensing and Estimation
- Sensors:
  - own-state
  - noisy own-state
  - joint state
  - relative measurement
  - access gating
- Estimators:
  - Orbit EKF
  - Orbit UKF
  - Attitude EKF
  - Joint state estimators
  - AoI tracking wrapper

### Control
- Orbital baseline controllers:
  - stationkeeping
  - safety barrier
  - risk-threshold switching
- Orbital maneuvering:
  - impulsive desired-velocity command
  - impulsive delta-V vector command
  - thrust-limited delta-V command
  - minimum-thrust enforcement (below minimum => no fire)
  - optional attitude-alignment gating with tolerance
  - required attitude target (`quat_bn`) solver from thrust vector
- Integrated orbital + attitude maneuver coordinator:
  - evaluates burn feasibility in current pose
  - if aligned and feasible => fire
  - if misaligned => slew target attitude, no fire
  - if below min thrust => slew target attitude, no fire
  - exposes controller-ready target attitude for downstream attitude control
- Orbital LQR:
  - `HCWLQRController` expects **curvilinear RIC relative state**
  - internally converts curvilinear RIC -> rectangular RIC for HCW/LQR
  - computes control in RIC and outputs thrust command in **ECI**
- Attitude controllers:
  - zero torque
  - snap
  - snap-and-hold (RIC mode flag path)
  - quaternion PD
  - generalized small-angle LQR with configurable inertia and wheel mounting geometry

### Scoring and Harness
- Engagement metrics and score summary utilities.
- Monte Carlo harness with seed control and JSON summaries.

## Frames and Conventions

- Primary truth orbit state: ECI.
- Attitude quaternion convention: `quat_bn` (body relative to inertial/ECI).
- Free-tumble RIC plotting utilities are available in examples.
- Orbital HCW LQR input/output handling:
  - input state to controller: curvilinear RIC
  - control law state: rectangular RIC (internal)
  - final command to actuator: ECI acceleration vector

## Presets

`presets/` provides reusable parameter sets for rapid simulation setup:
- rockets (SSTO, stage presets)
- satellites
- thrusters (including mounted chemical thruster)
- attitude control hardware (reaction wheel presets)

Quickstart:

```python
from presets import build_sim_object_from_presets

sat = build_sim_object_from_presets(object_id="sat_01", dt_s=2.0, orbit_radius_km=6778.0)
```

## Repository Layout

- `sim/core/`
- `sim/dynamics/orbit/`
- `sim/dynamics/attitude/`
- `sim/actuators/`
- `sim/sensors/`
- `sim/estimation/`
- `sim/control/orbit/`
- `sim/control/attitude/`
- `sim/scenarios/`
- `sim/metrics/`
- `sim/utils/`
- `sim/tests/`
- `examples/`
- `presets/`
- `archive/`

## Example Scripts

From repo root:

```bash
.venv/bin/python examples/Free_Tumble_One_Orbit.py
.venv/bin/python examples/Free_Tumble_One_Orbit_RIC.py
.venv/bin/python examples/Satellite_One_Orbit_StateKnowledge.py
.venv/bin/python examples/Satellite_One_Orbit_AttitudeKnowledge.py
.venv/bin/python examples/Impulsive_Maneuver_Demo.py
.venv/bin/python examples/Impulsive_DeltaV_Vector_Demo.py
.venv/bin/python examples/Impulsive_DeltaV_ThrustLimited_Demo.py
.venv/bin/python examples/Orbit_HCW_LQR_Demo.py
.venv/bin/python examples/Full_Framework_Demo.py
.venv/bin/python examples/MonteCarlo_Framework_Run.py
.venv/bin/python examples/Preset_Quickstart.py
```

## Notes

- Use `.venv/bin/python` in this environment to avoid local NumPy/Matplotlib ABI mismatch issues.
- Plotting default is interactive IDE display; file export is opt-in where supported by script flags.
- `noncoop_rpo` re-exports the `sim` framework surface for compatibility.
