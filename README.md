# NonCooperativeRPO

Modular orbital engagement simulation framework for SIL/HIL-style closed-loop experimentation.

## Roadmap Implementation Status

The framework now implements the full roadmap architecture across phases 1-7.

### Phase 1: Simulation Kernel
- Deterministic execution order in `sim/core/kernel.py`:
  1. Truth dynamics propagation
  2. Sensors/knowledge measurement
  3. Estimator belief update
  4. Controller command generation with deadline checks
  5. Actuator application (saturation/lag/quantization)
  6. Scoring/logging
- Multi-object world model with arbitrary object count.
- Real-time budget enforcement with skip behavior and zero-command fallback.
- Runtime and overrun logging per object.

### Phase 2: Orbital Dynamics
- Tier 1:
  - Two-body Earth gravity
  - ECI propagation
  - Fixed-step RK4
- Tier 2:
  - J2 perturbation plugin
  - Adaptive integration via Dormand-Prince 4/5 (`integrator="adaptive"`)
- Tier 3:
  - Drag plugin
  - Solar radiation pressure plugin
  - Third-body plugins (Moon, Sun)
  - ECI/ECEF frame transforms
- Plugin acceleration composition via `OrbitPropagator`.

### Phase 3: Attitude Dynamics
- Quaternion + angular-rate rigid body equations.
- Disturbance torques:
  - gravity-gradient,
  - magnetic dipole,
  - drag torque,
  - SRP torque.

### Phase 4: Actuator Modeling
- Orbital actuator:
  - continuous thrust,
  - lag,
  - throttle-rate limit,
  - minimum impulse bit,
  - propellant mass depletion model.
- Attitude actuator:
  - reaction wheel torque and momentum saturation,
  - magnetorquer command constraints,
  - thruster pulse quantization.

### Phase 5: Knowledge Module
- Sensor models:
  - own-state measurement,
  - relative angle/range/range-rate.
- Effects:
  - noise,
  - bias,
  - latency,
  - dropouts.
- Access model:
  - update cadence,
  - max range,
  - FOV,
  - optional line-of-sight visibility check.
- Estimation:
  - EKF and UKF implementations,
  - age-of-information tracking wrapper.

### Phase 6: Control Modules
- Orbit baseline strategies:
  - stationkeeping,
  - safety barrier,
  - risk-threshold switching.
- Attitude baseline strategies:
  - quaternion PD,
  - small-angle LQR.
- Advanced strategy scaffolds:
  - robust MPC wrapper,
  - stochastic policy wrapper.

### Phase 7: Scoring & Experiment Harness
- Engagement metrics:
  - minimum separation,
  - time inside keep-out region,
  - fuel usage,
  - compute overruns,
  - runtime jitter.
- Monte Carlo harness:
  - seed control,
  - uncertainty injection,
  - JSON summary output,
  - automatic histogram plot.

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
- `sim/examples/` (module placeholder; runnable scripts live in `examples/`)

## Run Examples

From repo root:

### One Orbit Free Tumble
Interactive plotting default:

```bash
.venv/bin/python examples/Free_Tumble_One_Orbit.py
```

Save plots instead:

```bash
.venv/bin/python examples/Free_Tumble_One_Orbit.py --plot-mode save
```

### Full Framework Demo

```bash
.venv/bin/python examples/Full_Framework_Demo.py
```

Outputs under `outputs/full_stack_demo/`.

### Monte Carlo Harness

```bash
.venv/bin/python examples/MonteCarlo_Framework_Run.py
```

Outputs under `outputs/full_stack_demo_mc/`.

## Presets Workflow

Use `presets/` when you want to fill simulation parameters from reusable hardware defaults instead of entering values one-by-one.

```python
from presets import build_sim_object_from_presets

sat = build_sim_object_from_presets(
    object_id="sat_01",
    dt_s=2.0,
    orbit_radius_km=6778.0,
)
```

Quick runnable example:

```bash
.venv/bin/python examples/Preset_Quickstart.py
```

## Notes

- Use `.venv/bin/python` in this environment to avoid local system NumPy/Matplotlib ABI mismatches.
- Plotting default is `interactive` (IDE display). File export is opt-in via `--plot-mode save` or `--plot-mode both`.
- `noncoop_rpo` now re-exports the new `sim` framework surface for compatibility.
