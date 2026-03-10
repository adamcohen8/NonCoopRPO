from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.cfs_sil.python_bridge import (
    CfsActuatorCommand,
    decode_sensor_packet,
    encode_command_packet,
)


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _build_command(sensor_vel_eci: np.ndarray, mode: str, accel_mag_km_s2: float, valid_timeout_ms: int) -> CfsActuatorCommand:
    thrust = np.zeros(3)
    torque = np.zeros(3)
    wheel = np.zeros(3)

    if mode == "hold":
        pass
    elif mode == "prograde":
        thrust = accel_mag_km_s2 * _unit(sensor_vel_eci)
    elif mode == "retrograde":
        thrust = -accel_mag_km_s2 * _unit(sensor_vel_eci)
    elif mode == "att_damp":
        # Simple body-rate damping torque placeholder.
        # (No direct omega here unless caller injects it; keep zero for now.)
        torque = np.zeros(3)
    else:
        pass

    return CfsActuatorCommand(
        mode=1,
        thrust_eci_km_s2=thrust,
        torque_body_nm=torque,
        wheel_torque_nm=wheel,
        valid_timeout_ms=int(valid_timeout_ms),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock cFS UDP endpoint for SIL loop testing.")
    parser.add_argument("--bind-host", type=str, default="127.0.0.1", help="Mock endpoint bind host.")
    parser.add_argument("--bind-port", type=int, default=50101, help="Mock endpoint bind port (receives sensors).")
    parser.add_argument("--sim-host", type=str, default="127.0.0.1", help="Simulator host to send commands to.")
    parser.add_argument("--sim-port", type=int, default=50100, help="Simulator port to send commands to.")
    parser.add_argument("--mode", choices=["hold", "prograde", "retrograde", "att_damp"], default="hold")
    parser.add_argument("--accel-mag-km-s2", type=float, default=2e-6, help="Thrust accel magnitude for pro/retrograde.")
    parser.add_argument("--valid-timeout-ms", type=int, default=500)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_host, int(args.bind_port)))
    sock.settimeout(1.0)

    sim_addr = (args.sim_host, int(args.sim_port))
    tx_seq = 0
    rx_count = 0
    t_last_print = time.time()

    print(
        f"Mock cFS endpoint listening on {args.bind_host}:{args.bind_port}, "
        f"sending commands to {args.sim_host}:{args.sim_port}, mode={args.mode}"
    )
    while True:
        try:
            raw, _addr = sock.recvfrom(4096)
        except socket.timeout:
            continue

        try:
            rx_seq, sim_time_ns, sensor = decode_sensor_packet(raw)
        except ValueError:
            continue

        cmd = _build_command(
            sensor_vel_eci=np.array(sensor.vel_eci_km_s, dtype=float),
            mode=args.mode,
            accel_mag_km_s2=float(args.accel_mag_km_s2),
            valid_timeout_ms=int(args.valid_timeout_ms),
        )
        pkt = encode_command_packet(cmd, seq=tx_seq, sim_time_ns=sim_time_ns)
        sock.sendto(pkt, sim_addr)
        tx_seq += 1
        rx_count += 1

        now = time.time()
        if now - t_last_print >= 1.0:
            print(
                f"rx={rx_count:6d} rx_seq={rx_seq:6d} tx_seq={tx_seq:6d} "
                f"sim_t={sim_time_ns/1e9:8.2f}s thrust=({cmd.thrust_eci_km_s2[0]:+.2e},"
                f"{cmd.thrust_eci_km_s2[1]:+.2e},{cmd.thrust_eci_km_s2[2]:+.2e})"
            )
            t_last_print = now


if __name__ == "__main__":
    main()
