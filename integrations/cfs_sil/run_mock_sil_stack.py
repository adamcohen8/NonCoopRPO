from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_cmd(py: str, script: Path, args: list[str]) -> list[str]:
    return [py, str(script)] + args


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mock cFS endpoint + simulator SIL loop from one terminal.")
    parser.add_argument("--mock-mode", choices=["hold", "prograde", "retrograde", "att_damp"], default="hold")
    parser.add_argument("--accel-mag-km-s2", type=float, default=2e-6)
    parser.add_argument("--bind-host", type=str, default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=50100)
    parser.add_argument("--mock-port", type=int, default=50101)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--atmosphere-model", choices=["exponential", "ussa1976", "nrlmsise00"], default="ussa1976")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    args = parser.parse_args()

    root = _repo_root()
    mock_script = root / "integrations" / "cfs_sil" / "mock_cfs_endpoint.py"
    sim_script = root / "examples" / "CFS_SIL_SingleSat_Loop_Demo.py"

    py = sys.executable
    mock_cmd = _build_cmd(
        py,
        mock_script,
        [
            "--bind-host",
            args.bind_host,
            "--bind-port",
            str(args.mock_port),
            "--sim-host",
            args.bind_host,
            "--sim-port",
            str(args.sim_port),
            "--mode",
            args.mock_mode,
            "--accel-mag-km-s2",
            str(args.accel_mag_km_s2),
        ],
    )
    sim_cmd = _build_cmd(
        py,
        sim_script,
        [
            "--bind-host",
            args.bind_host,
            "--bind-port",
            str(args.sim_port),
            "--cfs-host",
            args.bind_host,
            "--cfs-port",
            str(args.mock_port),
            "--duration",
            str(args.duration),
            "--dt",
            str(args.dt),
            "--atmosphere-model",
            args.atmosphere_model,
            "--plot-mode",
            args.plot_mode,
        ],
    )

    print("Starting mock cFS endpoint...")
    mock_proc = subprocess.Popen(mock_cmd)
    time.sleep(0.4)

    print("Starting simulator loop...")
    sim_rc = 1
    try:
        sim_rc = subprocess.call(sim_cmd)
    finally:
        if mock_proc.poll() is None:
            print("Stopping mock cFS endpoint...")
            mock_proc.terminate()
            try:
                mock_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                mock_proc.send_signal(signal.SIGKILL)
                mock_proc.wait(timeout=2.0)

    return int(sim_rc)


if __name__ == "__main__":
    raise SystemExit(main())
