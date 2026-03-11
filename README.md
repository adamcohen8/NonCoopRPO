# NonCooperativeRPO

Modular orbital engagement simulation framework for closed-loop SIL/HIL-oriented development.

## What This Project Is

This repository provides a simulation stack for:

- orbital dynamics (from two-body through higher-fidelity perturbations),
- attitude dynamics and control,
- sensing and estimation,
- actuator constraints and command application,
- multi-object interaction and rendezvous-style scenarios,
- validation and controller experimentation (including RL and gain tuning).

The core intent is to support realistic control algorithm development and mission concept prototyping.

## Current Architecture

Each simulation step follows a deterministic order in [sim/core/kernel.py](/Users/adamcohen/Downloads/NonCooperativeRPO/sim/core/kernel.py):

1. Truth propagation (dynamics)
2. Sensor measurement generation
3. Estimator update
4. Controller execution (with runtime budget/deadline logic)
5. Actuator application (limits/saturation/lag)
6. Logging/metrics capture

## Repository Layout

- `sim/core/` kernel, models, scheduling
- `sim/config/` shared simulation fidelity profiles (`fast`, `ops`, `high_fidelity`)
- `sim/dynamics/orbit/` orbital forces, integrators, atmosphere, spherical harmonics
- `sim/dynamics/attitude/` rigid body dynamics and disturbance torques
- `sim/actuators/` orbital and attitude actuator models
- `sim/sensors/` own-state and relative sensing models
- `sim/estimation/` EKF/UKF/joint estimators
- `sim/control/orbit/` orbital control, maneuver logic, HCW/LQR variants
- `sim/control/attitude/` PD/PID/LQR, RIC wrappers, pose command generation
- `sim/knowledge/` object knowledge tracking and update logic
- `sim/metrics/` scoring and engagement metrics
- `sim/optimization/` gain tuning and PSO framework
- `sim/rocket/` dedicated ascent engine
- `sim/tests/` unit/regression tests
- `examples/` runnable scripts and demos
- `presets/` reusable parameter presets
- `integrations/` external integration stubs (including cFS SIL bridge)
- `validation/` validation tooling and external reference-model workflows
- `archive/` legacy code retained out of active path

## Implemented Capability Snapshot

### Simulation Kernel

- Multi-object deterministic simulation loop
- Controller compute budget tracking with overrun skip behavior
- Runtime and skip logging per object

### Orbital Dynamics

- Two-body ECI propagation
- J2, J3, J4 perturbations
- Generic spherical harmonics perturbation pipeline
- Drag, SRP, third-body (Sun/Moon) plugins
- Earth rotation in drag relative velocity calculation
- Fixed-step RK4 and adaptive integration support
- Atmosphere models:
  - exponential
  - USSA 1976
  - NRLMSISE-00 (optional backend dependency)

### Attitude Dynamics

- Quaternion + body rate rigid-body propagation
- Disturbance torques:
  - gravity-gradient
  - magnetic dipole proxy
  - drag torque
  - SRP torque
- Reaction wheel-compatible torque application path
- Optional rectangular-prism coupling for drag/SRP projected area and face-based torques

### Actuators and Maneuvering

- Orbital actuator limits, lag, throttle dynamics, impulse-bit, mass depletion
- Reaction wheel limits and momentum saturation
- Impulsive and thrust-limited delta-V logic
- Attitude-gated thrusting with angular tolerance
- Integrated orbital-attitude maneuver coordination flow

### Estimation and Knowledge

- Orbit EKF/UKF
- Attitude EKF and joint state estimators
- Object knowledge tracking with update cadence, access conditions, and noise models

### Control

- Attitude:
  - quaternion PD
  - reaction wheel PD/PID
  - small-angle LQR
  - RIC-frame wrappers
  - snap/snap-and-hold modes
- Orbit:
  - HCW LQR variants
  - stationkeeping and safety-oriented controllers
  - predictive burn scheduling

### Mission Pose Command Generation

Quaternion-only mission pointing commands are available in [pose_commands.py](/Users/adamcohen/Downloads/NonCooperativeRPO/sim/control/attitude/pose_commands.py):

- `PoseCommandGenerator.sun_track(...)`
- `PoseCommandGenerator.spotlight_latlon(...)`
- `PoseCommandGenerator.spotlight_ric_direction(...)`

These commands return desired `q_bn` targets and do not directly actuate the vehicle.

## Validation Status

You have an active HPOP cross-validation workflow:

- Comparison script: [hpop_compare.py](/Users/adamcohen/Downloads/NonCooperativeRPO/validation/hpop_compare.py)
- Supports model-by-model comparisons (`two_body`, `drag`, `srp`, `j2`, `j3`, `j4`, `j2j3`, `sh8x8`)
- Uses fixed validation grid defaults:
  - `dt = 1 s`
  - `duration = 150 min`
- Produces:
  - component-wise state-difference plots (`simulator - HPOP`)
  - 3D ECI orbit overlays
  - RMS/max summary metrics

For spherical harmonics parity, the validator can use HPOP’s `GGM03C.txt` coefficients directly.

## Quick Start

### 1) Create and activate venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 2) Run a representative demo

```bash
python examples/Full_Framework_Demo.py
```

### 2b) Use fidelity profiles

Most modern demos now support a shared profile selector:

```bash
python examples/Full_Framework_Demo.py --profile ops
python examples/Free_Tumble_One_Orbit.py --profile fast
python examples/Orbit_SphericalHarmonics_8x8_Demo.py --profile high_fidelity
```

Profiles:

- `fast`: quickest turnaround (larger steps, minimal default modeling)
- `ops`: mission-engineering default balance
- `high_fidelity`: tighter integration settings for validation-style runs

### 2c) Initialize with Julian Date (epoch-aware dynamics)

You can now initialize simulation time with a Julian date using `SimConfig.initial_jd_utc` or by setting `env["jd_utc_start"]`.
When provided, the kernel populates `env["jd_utc"]` each step and enables simple analytic Sun/Moon ephemerides (`ephemeris_mode="analytic_simple"` by default), improving time dependence for:

- SRP sun direction
- Sun/Moon third-body vectors
- NRLMSISE-00 epoch handling

Ephemeris modes:

- `analytic_enhanced` (default): improved low-cost Sun/Moon analytic model
- `analytic_simple`: previous lightweight model
- `external`: use `env["ephemeris_callable"](jd_utc, env)` returning `sun_pos_eci_km` and `moon_pos_eci_km`
- `spice`: use `spiceypy` + kernels (`env["spice_kernels"]`), or a custom hook `env["spice_ephemeris_callable"]`

### 3) Run a validation comparison against HPOP output

```bash
python validation/hpop_compare.py --model two_body --dt 1 --duration-min 150 --plot-mode interactive
```

## Example Scripts

Common entry points in `examples/`:

- `Free_Tumble_One_Orbit.py`
- `Free_Tumble_One_Orbit_RIC.py`
- `Satellite_One_Orbit_StateKnowledge.py`
- `Satellite_One_Orbit_AttitudeKnowledge.py`
- `Orbit_OneOrbit_PerturbationError_Demo.py`
- `Orbit_SphericalHarmonics_8x8_Demo.py`
- `Orbit_SRP_Eclipse_Demo.py`
- `Rendezvous_HCW_AttitudeLQR_Demo.py`
- `Rendezvous_HCW_AttitudeLQR_PredictiveEKF_Demo.py`
- `Rendezvous_HCW_AttitudePD_PredictiveEKF_Demo.py`
- `Attitude_PD_ReactionWheel_Demo.py`
- `Attitude_PD_RIC_ReactionWheel_Demo.py`
- `Attitude_PID_RIC_ReactionWheel_Demo.py`
- `Attitude_Rectangle_Animation_ECI_Demo.py`
- `Attitude_Rectangle_Animation_ECI_PD_Demo.py`
- `Attitude_Rectangle_Animation_ECI_PD_Stabilize_Demo.py`
- `Two_Satellite_TargetFrame_DriftTumble_Demo.py`
- `Pose_Command_Quaternion_Demo.py`
- `Rocket_Launch_To_Orbit_Demo.py`
- `CFS_SIL_SingleSat_Loop_Demo.py`
- `Train_NN_Rendezvous_PPO.py`
- `Train_NN_AttitudeRIC_PPO.py`

## Presets

The `presets/` package provides reusable templates for:

- rockets
- satellites
- thrusters
- reaction wheel assemblies

Example:

```python
from presets import build_sim_object_from_presets

sat = build_sim_object_from_presets(
    object_id="sat_01",
    dt_s=2.0,
    orbit_radius_km=6778.0,
)
```

## cFS SIL Integration

`integrations/cfs_sil/` contains a lightweight bridge and demo flow:

- ICD definition
- Python bridge endpoint
- cFS app stub
- simulator loop adapter

Use this as the base for extending to real cFS message interfaces.

## Notes and Current Limitations

- Plotting defaults are interactive for current scripts, with save modes where available.
- SRP eclipse/shadow gating is modeled with configurable shadow mode (`conical`, `cylindrical`, or `none`).
- Some validation differences are expected from integrator and model-convention differences.
- In this environment, use the project venv to avoid NumPy/Matplotlib ABI conflicts.

## Roadmap Context

The project is already beyond the initial kernel-only stage in [simulation_framework_roadmap.txt](/Users/adamcohen/Downloads/NonCooperativeRPO/simulation_framework_roadmap.txt), with most phases implemented and active validation now in progress.
