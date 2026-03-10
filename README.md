# NonCooperativeRPO

Modular orbital engagement simulation framework for SIL/HIL-style closed-loop experimentation.

## Current Status

Core architecture is implemented across simulation kernel, orbit/attitude dynamics, sensing/estimation, controls, optimization, ML training harnesses, and integration stubs.

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
  - J3
  - J4
  - drag
  - solar radiation pressure
  - third-body Moon/Sun
- Configurable atmospheric density backends for drag:
  - exponential
  - USSA1976
  - NRLMSISE-00 (via optional backend/callable hook)
- Earth rotation is included in atmosphere-relative drag velocity.
- Optional generic spherical harmonics plugin for sectoral/tesseral terms:
  - user specifies arbitrary `(n, m, c_nm, s_nm)` terms.
- Fixed-step and adaptive propagation options through `OrbitPropagator`.

### Attitude Dynamics
- Quaternion + body-rate rigid body dynamics.
- Optional disturbance torques:
  - gravity-gradient
  - magnetic dipole
  - drag torque
  - SRP torque
- Optional rectangular-prism coupling mode (non-default):
  - user sets `Lx, Ly, Lz`
  - attitude-dependent projected area for drag and SRP in orbit propagation
  - face-based drag/SRP disturbance torque from individual face forces
  - requires coupled orbit+attitude disturbance simulation to be enabled

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
  - predictive burn scheduler (future-state planning + burn gate)
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
  - curvilinear-input/rectangular-LQR variant available
- Attitude controllers:
  - zero torque
  - snap
  - snap-and-hold (RIC mode flag path)
  - quaternion PD
  - reaction-wheel PD/PID (ECI and RIC-frame wrappers)
  - generalized small-angle LQR with configurable inertia and wheel mounting geometry

### Scoring and Harness
- Engagement metrics and score summary utilities.
- Monte Carlo harness with seed control and JSON summaries.
- Controller gain optimization framework with pluggable interface:
  - PSO backend implemented
  - preset and custom test-case support
  - equal-weight aggregate cost across cases

## Frames and Conventions

- Primary truth orbit state: ECI.
- Attitude quaternion convention: `quat_bn` (body relative to inertial/ECI).
- Free-tumble RIC plotting utilities are available in examples.
- Orbital HCW LQR input/output handling:
  - input state to controller: curvilinear RIC
  - control law state: rectangular RIC (internal)
  - final command to actuator: ECI acceleration vector

## Rocket Ascent Engine (Dedicated)

`sim/rocket/` provides a dedicated launch-to-insertion simulation path:
- multi-stage mass/thrust propagation with stage separation
- launch initialization from site/azimuth
- coupled orbit + attitude propagation
- guidance-law interface for ascent logic
- insertion criteria (target altitude/eccentricity + hold time)

Demo:
- `examples/Rocket_Launch_To_Orbit_Demo.py`

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
- `sim/optimization/`
- `sim/rocket/`
- `sim/utils/`
- `sim/tests/`
- `examples/`
- `presets/`
- `integrations/`
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
.venv/bin/python examples/Orbit_HCW_LQR_CurvVariant_Demo.py
.venv/bin/python examples/Orbit_OneOrbit_PerturbationError_Demo.py
.venv/bin/python examples/Full_Framework_Demo.py
.venv/bin/python examples/MonteCarlo_Framework_Run.py
.venv/bin/python examples/MonteCarlo_Rendezvous_PredictiveEKF.py
.venv/bin/python examples/Rendezvous_HCW_AttitudeLQR_Demo.py
.venv/bin/python examples/Rendezvous_HCW_AttitudeLQR_PredictiveEKF_Demo.py
.venv/bin/python examples/Rendezvous_HCW_AttitudePD_PredictiveEKF_Demo.py
.venv/bin/python examples/Attitude_PD_ReactionWheel_Demo.py
.venv/bin/python examples/Attitude_PD_RIC_ReactionWheel_Demo.py
.venv/bin/python examples/Attitude_PID_RIC_ReactionWheel_Demo.py
.venv/bin/python examples/Optimize_Attitude_Controller_Gains.py
.venv/bin/python examples/Object_Knowledge_EKF_Demo.py
.venv/bin/python examples/Train_NN_Rendezvous_PPO.py
.venv/bin/python examples/Train_NN_AttitudeRIC_PPO.py
.venv/bin/python examples/Demo_NN_Rendezvous_BestEpoch.py
.venv/bin/python examples/Demo_NN_AttitudeRIC_BestEpoch.py
.venv/bin/python examples/Rocket_Launch_To_Orbit_Demo.py
.venv/bin/python examples/CFS_SIL_SingleSat_Loop_Demo.py
.venv/bin/python examples/Preset_Quickstart.py
```

## cFS SIL Starter

`integrations/cfs_sil/` includes:
- UDP ICD (`icd.yaml`)
- Python bridge endpoint (`python_bridge.py`)
- cFS app stub (`cfs_app_stub/`)
- simulator loop adapter + demo integration

## Notes

- Use `.venv/bin/python` in this environment to avoid local NumPy/Matplotlib ABI mismatch issues.
- Plotting default is interactive IDE display; file export is opt-in where supported by script flags.
- `noncoop_rpo` re-exports the `sim` framework surface for compatibility.
