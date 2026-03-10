from __future__ import annotations

import argparse
import socket
import struct
import time
from dataclasses import dataclass

import numpy as np

MAGIC = b"RPO1"
VERSION = 1

MSG_SIM_SENSOR_STATE = 0x1900
MSG_CFS_ACTUATOR_CMD = 0x1901

HEADER_FMT = "<4sHHIQ"  # magic, version, msg_id, seq, sim_time_ns
HEADER_SIZE = struct.calcsize(HEADER_FMT)

SENSOR_FMT = "<B 3d 3d 4d 3d d 3d d"
CMD_FMT = "<B 3d 3d 3d I"
SENSOR_SIZE = struct.calcsize(SENSOR_FMT)
CMD_SIZE = struct.calcsize(CMD_FMT)


@dataclass
class SimSensorState:
    valid_flags: int
    pos_eci_km: np.ndarray
    vel_eci_km_s: np.ndarray
    quat_bn: np.ndarray
    omega_body_rad_s: np.ndarray
    mass_kg: float
    sun_dir_eci: np.ndarray
    density_kg_m3: float


@dataclass
class CfsActuatorCommand:
    mode: int
    thrust_eci_km_s2: np.ndarray
    torque_body_nm: np.ndarray
    wheel_torque_nm: np.ndarray
    valid_timeout_ms: int


def _pack_header(msg_id: int, seq: int, sim_time_ns: int) -> bytes:
    return struct.pack(HEADER_FMT, MAGIC, VERSION, int(msg_id), int(seq), int(sim_time_ns))


def _unpack_header(raw: bytes) -> tuple[int, int, int]:
    if len(raw) < HEADER_SIZE:
        raise ValueError("Packet too short for header.")
    magic, version, msg_id, seq, sim_time_ns = struct.unpack(HEADER_FMT, raw[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError("Bad magic.")
    if version != VERSION:
        raise ValueError(f"Unsupported protocol version {version}.")
    return int(msg_id), int(seq), int(sim_time_ns)


def encode_sensor_packet(state: SimSensorState, seq: int, sim_time_ns: int) -> bytes:
    payload = struct.pack(
        SENSOR_FMT,
        int(state.valid_flags),
        *np.array(state.pos_eci_km, dtype=float).reshape(3),
        *np.array(state.vel_eci_km_s, dtype=float).reshape(3),
        *np.array(state.quat_bn, dtype=float).reshape(4),
        *np.array(state.omega_body_rad_s, dtype=float).reshape(3),
        float(state.mass_kg),
        *np.array(state.sun_dir_eci, dtype=float).reshape(3),
        float(state.density_kg_m3),
    )
    return _pack_header(MSG_SIM_SENSOR_STATE, seq, sim_time_ns) + payload


def decode_sensor_packet(raw: bytes) -> tuple[int, int, SimSensorState]:
    msg_id, seq, sim_time_ns = _unpack_header(raw)
    if msg_id != MSG_SIM_SENSOR_STATE:
        raise ValueError(f"Unexpected msg_id 0x{msg_id:04X}.")
    payload = raw[HEADER_SIZE:]
    if len(payload) != SENSOR_SIZE:
        raise ValueError(f"Invalid sensor payload size {len(payload)} != {SENSOR_SIZE}.")
    values = struct.unpack(SENSOR_FMT, payload)
    st = SimSensorState(
        valid_flags=int(values[0]),
        pos_eci_km=np.array(values[1:4], dtype=float),
        vel_eci_km_s=np.array(values[4:7], dtype=float),
        quat_bn=np.array(values[7:11], dtype=float),
        omega_body_rad_s=np.array(values[11:14], dtype=float),
        mass_kg=float(values[14]),
        sun_dir_eci=np.array(values[15:18], dtype=float),
        density_kg_m3=float(values[18]),
    )
    return seq, sim_time_ns, st


def encode_command_packet(cmd: CfsActuatorCommand, seq: int, sim_time_ns: int) -> bytes:
    payload = struct.pack(
        CMD_FMT,
        int(cmd.mode),
        *np.array(cmd.thrust_eci_km_s2, dtype=float).reshape(3),
        *np.array(cmd.torque_body_nm, dtype=float).reshape(3),
        *np.array(cmd.wheel_torque_nm, dtype=float).reshape(3),
        int(cmd.valid_timeout_ms),
    )
    return _pack_header(MSG_CFS_ACTUATOR_CMD, seq, sim_time_ns) + payload


def decode_command_packet(raw: bytes) -> tuple[int, int, CfsActuatorCommand]:
    msg_id, seq, sim_time_ns = _unpack_header(raw)
    if msg_id != MSG_CFS_ACTUATOR_CMD:
        raise ValueError(f"Unexpected msg_id 0x{msg_id:04X}.")
    payload = raw[HEADER_SIZE:]
    if len(payload) != CMD_SIZE:
        raise ValueError(f"Invalid command payload size {len(payload)} != {CMD_SIZE}.")
    values = struct.unpack(CMD_FMT, payload)
    cmd = CfsActuatorCommand(
        mode=int(values[0]),
        thrust_eci_km_s2=np.array(values[1:4], dtype=float),
        torque_body_nm=np.array(values[4:7], dtype=float),
        wheel_torque_nm=np.array(values[7:10], dtype=float),
        valid_timeout_ms=int(values[10]),
    )
    return seq, sim_time_ns, cmd


class CfsSilUdpBridge:
    """
    UDP bridge endpoint for simulator <-> cFS SIL coupling.

    - Sends SIM_SENSOR_STATE packets to cFS.
    - Receives CFS_ACTUATOR_CMD packets from cFS.
    """

    def __init__(self, local_bind: tuple[str, int], cfs_addr: tuple[str, int]) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(local_bind)
        self.sock.setblocking(False)
        self.cfs_addr = cfs_addr
        self.tx_seq = 0
        self.last_cmd_seq = -1
        self.last_cmd_sim_time_ns = 0
        self.last_cmd = CfsActuatorCommand(
            mode=0,
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=np.zeros(3),
            wheel_torque_nm=np.zeros(3),
            valid_timeout_ms=200,
        )
        self.last_cmd_rx_wall_time_ns = time.time_ns()

    def send_sensor(self, state: SimSensorState, sim_time_ns: int) -> None:
        pkt = encode_sensor_packet(state, seq=self.tx_seq, sim_time_ns=sim_time_ns)
        self.sock.sendto(pkt, self.cfs_addr)
        self.tx_seq += 1

    def poll_command(self) -> CfsActuatorCommand:
        while True:
            try:
                raw, _addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                seq, sim_time_ns, cmd = decode_command_packet(raw)
            except ValueError:
                continue
            self.last_cmd = cmd
            self.last_cmd_seq = seq
            self.last_cmd_sim_time_ns = sim_time_ns
            self.last_cmd_rx_wall_time_ns = time.time_ns()
        # command timeout -> hold
        age_ms = (time.time_ns() - self.last_cmd_rx_wall_time_ns) / 1e6
        if age_ms > max(1, self.last_cmd.valid_timeout_ms):
            return CfsActuatorCommand(
                mode=0,
                thrust_eci_km_s2=np.zeros(3),
                torque_body_nm=np.zeros(3),
                wheel_torque_nm=np.zeros(3),
                valid_timeout_ms=self.last_cmd.valid_timeout_ms,
            )
        return self.last_cmd


def _demo_sensor_state(t_s: float) -> SimSensorState:
    return SimSensorState(
        valid_flags=0x0F,
        pos_eci_km=np.array([6778.0 * np.cos(t_s / 5400.0), 6778.0 * np.sin(t_s / 5400.0), 0.0]),
        vel_eci_km_s=np.array([-7.6 * np.sin(t_s / 5400.0), 7.6 * np.cos(t_s / 5400.0), 0.0]),
        quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body_rad_s=np.zeros(3),
        mass_kg=300.0,
        sun_dir_eci=np.array([1.0, 0.0, 0.0]),
        density_kg_m3=1e-12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="UDP bridge endpoint for cFS SIL integration.")
    parser.add_argument("--bind-host", type=str, default="127.0.0.1")
    parser.add_argument("--bind-port", type=int, default=50100)
    parser.add_argument("--cfs-host", type=str, default="127.0.0.1")
    parser.add_argument("--cfs-port", type=int, default=50101)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--demo", action="store_true", help="Run standalone packet loop with synthetic sensor state.")
    args = parser.parse_args()

    bridge = CfsSilUdpBridge(
        local_bind=(args.bind_host, int(args.bind_port)),
        cfs_addr=(args.cfs_host, int(args.cfs_port)),
    )
    period_s = 1.0 / max(float(args.rate_hz), 1e-3)
    t0 = time.time()
    k = 0
    if args.demo:
        print(
            f"cFS SIL bridge demo running: bind={args.bind_host}:{args.bind_port} -> "
            f"cfs={args.cfs_host}:{args.cfs_port}, rate={args.rate_hz:.2f} Hz"
        )
    while args.demo:
        t_s = k * period_s
        sim_time_ns = int(t_s * 1e9)
        state = _demo_sensor_state(t_s)
        bridge.send_sensor(state, sim_time_ns=sim_time_ns)
        cmd = bridge.poll_command()
        if k % int(max(1, round(args.rate_hz))) == 0:
            print(
                f"t={t_s:8.2f}s seq={bridge.tx_seq:6d} cmd_mode={cmd.mode} "
                f"thrust=({cmd.thrust_eci_km_s2[0]:+.3e},{cmd.thrust_eci_km_s2[1]:+.3e},{cmd.thrust_eci_km_s2[2]:+.3e})"
            )
        k += 1
        next_t = t0 + k * period_s
        sleep_s = next_t - time.time()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
