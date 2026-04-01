# Orbital Engagement Lab

Integrated spacecraft simulation and analysis framework for rendezvous, proximity operations, control development, and mission concept prototyping.

## Overview

This repository is best thought of as an engineering sandbox for spacecraft GNC and mission simulation. It combines:

- orbital dynamics, from two-body through higher-fidelity perturbations,
- attitude dynamics and control,
- sensing, estimation, and object knowledge,
- actuator limits and command application,
- multi-object scenarios including rendezvous-style engagements,
- campaign analysis such as Monte Carlo and sensitivity studies, and
- validation, RL experimentation, and GUI-driven workflows.

The goal is not just to propagate trajectories. The goal is to support closed-loop simulation where dynamics, estimation, control, mission behavior, and outputs all interact in one place.

## Best Fit

This repository is strongest for:

- spacecraft rendezvous and proximity operations research,
- integrated orbit-attitude control development,
- controller comparison and mission-logic prototyping,
- Monte Carlo and sensitivity campaign analysis,
- simulation-backed validation against higher-fidelity references, and
- RL or autonomy experiments that need access to a live simulation stack.

It is less optimized for:

- minimal, highly polished SDK-style workflows,
- flight-qualified or high-assurance operational software, and
- narrowly scoped single-purpose orbit tools.

## Primary Workflows

There are five main ways to use the project:

1. CLI single-run simulation via [`run_simulation.py`](run_simulation.py)
2. Native desktop GUI via [`run_gui.py`](run_gui.py)
3. Programmatic API via [`sim/api.py`](sim/api.py)
4. Analysis campaigns using the `analysis` config section for Monte Carlo or sensitivity/LHS studies
5. Validation workflows via [`validation/automated_validation_harness.py`](validation/automated_validation_harness.py) and [`validation/hpop_compare.py`](validation/hpop_compare.py)

## Quick Start

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 2) Run a representative CLI scenario

```bash
python run_simulation.py --config configs/automation_smoke.yaml
```

### 3) Open the desktop GUI

```bash
python run_gui.py
```

The GUI lets you:

- load and edit YAML-backed scenario configs,
- run the existing CLI entrypoint from a desktop workflow,
- configure Analysis studies, including sensitivity and LHS,
- inspect output artifacts and summaries.

### 4) Use the public API

```python
from sim import SimulationConfig, SimulationSession

cfg = SimulationConfig.from_yaml("configs/automation_smoke.yaml")
session = SimulationSession.from_config(cfg)
result = session.run()

print(result.summary["scenario_name"])
```

### 5) Run campaign analysis

Monte Carlo and sensitivity studies now live under the top-level `analysis` section in scenario configs.

Sensitivity supports:

- one-at-a-time parameter studies,
- tracked metrics,
- optional baseline comparison, and
- Latin hypercube sampling (LHS).

### 6) Run validation tooling

Smoke validation suite:

```bash
python validation/automated_validation_harness.py --spec configs/validation_harness_smoke.yaml
```

HPOP comparison:

```bash
python validation/hpop_compare.py --model two_body --dt 1 --duration-min 150 --plot-mode interactive
```

## Architecture At A Glance

Each simulation step follows a deterministic order in [`sim/core/kernel.py`](sim/core/kernel.py):

1. truth propagation,
2. sensor measurement generation,
3. estimator update,
4. controller execution,
5. actuator application, and
6. logging and metrics capture.

The broader project flow is:

- scenario config defines vehicles, dynamics, outputs, and analysis settings,
- the simulation engine executes a single run or campaign,
- summaries, plots, animations, and analysis artifacts are written to the configured output directory,
- validation and RL wrappers reuse the same underlying simulation stack rather than duplicating a separate physics engine.

## Repository Layout

- `sim/core/` kernel, models, scheduling
- `sim/config/` config schema, fidelity profiles, plugin validation
- `sim/api.py` public programmatic simulation API
- `sim/master_simulator.py` orchestration and campaign execution
- `sim/master_outputs.py` plotting and animation helpers
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
- `sim_gui/` native desktop GUI
- `machine_learning/` RL environments and training helpers
- `validation/` validation tooling and external reference-model workflows
- `examples/` runnable scripts and demos
- `presets/` reusable parameter presets
- `integrations/` external integration stubs, including a cFS SIL bridge
- `sim/tests/` unit and regression tests

## Capability Snapshot

### Simulation and Dynamics

- multi-object deterministic simulation loop,
- two-body through higher-fidelity orbital propagation,
- quaternion rigid-body attitude propagation,
- drag, SRP, third-body, and spherical harmonics support,
- actuator limits, lag, saturation, and mass depletion,
- optional rectangular-prism coupling for drag/SRP projected area and torque effects.

### Estimation, Control, and Mission Logic

- orbit EKF/UKF and attitude/joint estimation paths,
- quaternion PD/PID, LQR, RIC wrappers, and snap-style attitude modes,
- HCW LQR, stationkeeping, and predictive burn orbital control,
- mission execution and objective composition for rocket and satellite scenarios,
- object knowledge tracking with cadence, access conditions, and noise models.

### Analysis and Validation

- single-run summaries, plots, and animations,
- Monte Carlo campaigns with serial or parallel execution,
- sensitivity studies including one-at-a-time and LHS methods,
- HPOP cross-validation and config-driven validation suites,
- JSON and Markdown reporting for automated validation runs.

### Interfaces

- CLI entrypoint for scriptable runs,
- desktop GUI for scenario editing and analysis workflows,
- public Python API for programmatic runs and live stepping,
- RL wrappers for Gymnasium-style and vectorized environments.

## Validation Status

You have an active HPOP cross-validation workflow:

- Comparison script: [`hpop_compare.py`](validation/hpop_compare.py)
- Supports model-by-model comparisons (`two_body`, `drag`, `srp`, `j2`, `j3`, `j4`, `j2j3`, `sh8x8`)
- Uses fixed validation grid defaults:
  - `dt = 1 s`
  - `duration = 150 min`
- Produces:
  - component-wise state-difference plots (`simulator - HPOP`)
  - 3D ECI orbit overlays
  - RMS/max summary metrics

For spherical harmonics parity, the validator can use HPOP’s `GGM03C.txt` coefficients directly.

## Automated Validation Harness

You can run a config-driven validation suite that combines:

- plugin/config validation,
- end-to-end simulation runs, and
- HPOP cross-validation checks

with tolerance gates and consolidated JSON/Markdown reporting.

Smoke suite:

```bash
python validation/automated_validation_harness.py --spec configs/validation_harness_smoke.yaml
```

Default suite:

```bash
python validation/automated_validation_harness.py --spec configs/validation_harness_default.yaml
```

The default suite includes:

- plugin validation,
- a single-run integrated rendezvous benchmark,
- Monte Carlo rendezvous validation,
- high-fidelity orbit-stack Monte Carlo validation, and
- multiple HPOP parity cases (`two_body`, `j2`, `j3`, `sh8x8`)

If `--spec` is omitted, the harness runs a built-in smoke suite.

Artifacts are written under the suite `output_dir`, including:

- `validation_harness_report.json`
- `validation_harness_report.md`

## Additional Notes

### Headless automation and CI

For non-interactive environments, use a save-mode config such as:

```bash
python run_simulation.py --config configs/automation_smoke.yaml
```

`run_master_simulation` also auto-switches `outputs.mode: interactive` to `save` when `SIM_AUTOMATION=1` or `CI=1`.

### Fidelity profiles

Many demos support a shared profile selector:

```bash
python examples/Full_Framework_Demo.py --profile ops
python examples/Free_Tumble_One_Orbit.py --profile fast
python examples/Orbit_SphericalHarmonics_8x8_Demo.py --profile high_fidelity
```

Profiles:

- `fast`: quickest turnaround
- `ops`: mission-engineering default balance
- `high_fidelity`: tighter integration settings for validation-style runs

### Epoch-aware simulation time

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

Planetary third-body perturbations:

- Use `third_body_planets_plugin` in your propagator.
- Select bodies via `env["third_body_planets"]` (e.g., `["venus","mars","jupiter"]` or `"all"`).
- For non-Sun/Moon planets, resolution is supported through:
  - `ephemeris_mode="spice"` (with kernels), or
  - explicit `env["<planet>_pos_eci_km"]`, or
  - custom `env["ephemeris_body_callable"]` / `env["spice_body_ephemeris_callable"]`.

## Example Scripts

Common entry points in `examples/`:

- `Free_Tumble_One_Orbit.py`
- `Free_Tumble_One_Orbit_RIC.py`
- `Satellite_One_Orbit_StateKnowledge.py`
- `Satellite_One_Orbit_AttitudeKnowledge.py`
- `Orbit_OneOrbit_PerturbationError_Demo.py`
- `Orbit_SphericalHarmonics_8x8_Demo.py`
- `Orbit_SRP_Eclipse_Demo.py`
- `Orbit_GroundTrack_Demo.py`
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
- `Train_Generic_RL_Env.py`
- `Train_MultiAgent_SelfPlay_Demo.py`

## Generic RL Environment

There is now an MVP Gymnasium-style RL wrapper in [`machine_learning/gym_env.py`](machine_learning/gym_env.py).

It supports:

- `reset()` / `step()` with Gymnasium return signatures
- configurable observation selection via path strings
- configurable action channels via named action fields
- per-episode parameter variation using the existing Monte Carlo variation schema
- hybrid control via action adapters, including a built-in thrust-vector adapter that can feed mission execution / attitude control instead of commanding raw torques directly
- vectorized rollouts with synchronous or subprocess-based execution
- a multi-agent wrapper for simultaneous `chaser` / `target` control in one shared environment

The intended pattern is:

```python
from machine_learning import (
    ActionField,
    GymEnvConfig,
    GymSimulationEnv,
    ObservationField,
    ThrustVectorToPointingAdapter,
)

scenario = {
    "rocket": {"enabled": False},
    "target": {"enabled": True},
    "chaser": {
        "enabled": True,
        "mission_execution": {
            "module": "sim.mission.modules",
            "class_name": "ControllerPointingExecution",
            "params": {"alignment_tolerance_deg": 10.0},
        },
    },
    "simulator": {"duration_s": 300.0, "dt_s": 1.0},
}

env = GymSimulationEnv(
    GymEnvConfig(
        scenario=scenario,
        controlled_agent_id="chaser",
        observation_fields=(
            ObservationField("truth.chaser.position_eci_km"),
            ObservationField("truth.target.position_eci_km"),
            ObservationField("truth.chaser.attitude_quat_bn"),
        ),
        action_fields=(
            ActionField("thrust_direction_eci[0]", -1.0, 1.0),
            ActionField("thrust_direction_eci[1]", -1.0, 1.0),
            ActionField("thrust_direction_eci[2]", -1.0, 1.0),
            ActionField("throttle", 0.0, 1.0),
        ),
        action_adapter=ThrustVectorToPointingAdapter(),
    )
)
```

The wrapper currently targets satellite scenarios and reuses the simulator’s controller / mission composition pipeline rather than the output-heavy full `run_master_simulation()` path.

For rollout throughput, build a vector env:

```python
from machine_learning import VectorEnvConfig, make_vector_env

vec_env = make_vector_env(
    VectorEnvConfig(
        env_cfg=GymEnvConfig(
            scenario=scenario,
            controlled_agent_id="chaser",
            observation_fields=(
                ObservationField("truth.chaser.position_eci_km"),
                ObservationField("truth.target.position_eci_km"),
            ),
            action_fields=(
                ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),
                ActionField("thrust_eci_km_s2[1]", -1e-5, 1e-5),
                ActionField("thrust_eci_km_s2[2]", -1e-5, 1e-5),
            ),
        ),
        num_envs=8,
        parallel=True,
        auto_reset=True,
    )
)
```

`parallel=False` runs a synchronous batched wrapper in-process. `parallel=True` uses spawned subprocess workers so multiple episodes can advance concurrently.

For custom PPO/A2C loops, collect batched rollouts directly:

```python
from machine_learning import collect_vector_rollout
import numpy as np

batch = collect_vector_rollout(
    vec_env,
    policy_fn=lambda obs: np.zeros((obs.shape[0], 3), dtype=np.float32),
    horizon=128,
)
```

This returns stacked `obs`, `actions`, `rewards`, `terminated`, `truncated`, `next_obs`, and `infos`.

For Stable-Baselines3, use the optional adapter:

```python
from machine_learning import VectorEnvConfig, make_sb3_vec_env

sb3_env = make_sb3_vec_env(
    VectorEnvConfig(
        env_cfg=GymEnvConfig(
            scenario=scenario,
            controlled_agent_id="chaser",
            observation_fields=(ObservationField("truth.chaser.position_eci_km"),),
            action_fields=(
                ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),
                ActionField("thrust_eci_km_s2[1]", -1e-5, 1e-5),
                ActionField("thrust_eci_km_s2[2]", -1e-5, 1e-5),
            ),
        ),
        num_envs=8,
        parallel=True,
    )
)
```

`make_sb3_vec_env()` requires `stable_baselines3` to be installed.

For adversarial or self-play setups, use the multi-agent env:

```python
from machine_learning import MultiAgentEnvConfig, MultiAgentSimulationEnv

ma_env = MultiAgentSimulationEnv(
    MultiAgentEnvConfig(
        scenario=scenario,
        controlled_agent_ids=("chaser", "target"),
        observation_fields_by_agent={
            "chaser": (
                ObservationField("truth.chaser.position_eci_km"),
                ObservationField("truth.target.position_eci_km"),
            ),
            "target": (
                ObservationField("truth.target.position_eci_km"),
                ObservationField("truth.chaser.position_eci_km"),
            ),
        },
        action_fields_by_agent={
            "chaser": (
                ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),
                ActionField("thrust_eci_km_s2[1]", -1e-5, 1e-5),
                ActionField("thrust_eci_km_s2[2]", -1e-5, 1e-5),
            ),
            "target": (
                ActionField("thrust_eci_km_s2[0]", -1e-5, 1e-5),
                ActionField("thrust_eci_km_s2[1]", -1e-5, 1e-5),
                ActionField("thrust_eci_km_s2[2]", -1e-5, 1e-5),
            ),
        },
    )
)
```

`reset()` returns per-agent observation and info dicts. `step()` takes a dict of per-agent actions and returns per-agent observations, rewards, terminations, truncations, and infos.

For training infrastructure, the repository now also includes:

- `collect_multi_agent_rollout(...)` for batched self-play/adversarial trajectory collection
- `run_self_play_training(...)` with opponent-pool support and alternating vs simultaneous update schedules
- `Train_MultiAgent_SelfPlay_Demo.py` as a runnable example built on that trainer

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
- Master simulation summaries now include `attitude_guardrail_stats` so numerical clamp/sanitize events are visible in run outputs.

## Roadmap Context

The project is already beyond the initial kernel-only stage in [`simulation_framework_roadmap.txt`](simulation_framework_roadmap.txt), with most phases implemented and active validation now in progress.
