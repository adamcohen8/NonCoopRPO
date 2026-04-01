# cFS SIL Starter Kit

This folder contains a first-pass wiring pattern for running NASA cFS against this simulator in SIL mode.

## Contents

- `icd.yaml`: packet/interface control document for simulator <-> cFS exchange.
- `python_bridge.py`: UDP bridge endpoint + packet encoder/decoder.
- `cfs_app_stub/`: minimal cFS app scaffold that subscribes to simulated sensor packets and publishes actuator command packets.

## Data Flow

1. Simulator generates truth and synthetic sensors.
2. `python_bridge.py` sends `SIM_SENSOR_STATE (0x1900)` to cFS.
3. cFS bridge/GNC app receives sensor packet, computes command.
4. cFS sends `ACTUATOR_CMD (0x1901)` back.
5. Simulator applies command on next step (sample-and-hold).

## Quick Bring-up

### 1) Run mock cFS endpoint (no external cFS required)

```bash
python integrations/cfs_sil/mock_cfs_endpoint.py --mode hold
```

or with a simple prograde thrust command:

```bash
python integrations/cfs_sil/mock_cfs_endpoint.py --mode prograde --accel-mag-km-s2 2e-6
```

### 1b) Single-terminal launcher (starts both mock endpoint and sim loop)

```bash
python integrations/cfs_sil/run_mock_sil_stack.py --mock-mode hold
```

### 2) (Optional) Run bridge demo endpoint

```bash
python integrations/cfs_sil/python_bridge.py --demo
```

### 3) Run simulator loop with truth->bridge->command feedback

```bash
python examples/CFS_SIL_SingleSat_Loop_Demo.py --plot-mode interactive
```

This loop:
- publishes truth-derived sensor packets each sim step,
- polls cFS actuator command packets each step,
- applies commands to the simulator dynamics.

### 4) Integrate cFS stub

Copy `cfs_app_stub/fsw/src/*` into your cFS app tree and wire it through your cFS build system.

### 5) Match packet IDs and ports

Use message IDs and field layout from `icd.yaml`. Keep simulator as time master.

## Notes

- This starter uses plain UDP for fast iteration.
- For qualification-grade SIL, add:
  - packet sequence checks + replay logs,
  - modeled latency/jitter/drop,
  - strict task-rate scheduling validation,
  - requirements-driven pass/fail reports.
