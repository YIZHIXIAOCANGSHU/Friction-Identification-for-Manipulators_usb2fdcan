import argparse
import errno
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


CAN_MTU = 16
CANFD_MTU = 72
CAN_RAW_FD_FRAMES = getattr(socket, "CAN_RAW_FD_FRAMES", 5)
SOL_CAN_RAW = getattr(socket, "SOL_CAN_RAW", socket.SOL_CAN_BASE + socket.CAN_RAW)
CANFD_BRS = getattr(socket, "CANFD_BRS", 0x01)

DEFAULT_CAN_ID = 0x01
DEFAULT_MST_ID = 0x11
DEFAULT_MOTOR_TYPE = "DM8009"
DEFAULT_INTERFACE = "can0"
CLEAR_ERROR_CMD = 0xFB
ENABLE_CMD = 0xFC
DISABLE_CMD = 0xFD
ZERO_CMD = 0xFE
DEFAULT_COMMAND_INTERVAL_MS = 0.0
DEFAULT_LISTEN_DURATION = 0.0
DEFAULT_PRINT_INTERVAL = 0.1
DEFAULT_SEND_RATE_LOG_INTERVAL = 0.1
DEFAULT_BACKPRESSURE_SLEEP = 0.0005
MAX_BACKPRESSURE_SLEEP = 0.01
DEFAULT_CONTROL_COMMAND_REPEAT = 5
DEFAULT_CONTROL_COMMAND_INTERVAL = 0.002
DEFAULT_PARAM_WRITE_SETTLE = 0.002
SERIALIZED_FEEDBACK_HEAD = 0xA5
SERIALIZED_FEEDBACK_FORMAT = "<BBBffff"
SERIALIZED_FEEDBACK_STRUCT = struct.Struct(SERIALIZED_FEEDBACK_FORMAT)
VALID_FEEDBACK_STATE_CODES = frozenset({0x0, 0x1, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE})


class DM_Motor_Type(IntEnum):
    DM3507 = 0
    DM4310 = 1
    DM4310_48V = 2
    DM4340 = 3
    DM4340_48V = 4
    DM6006 = 5
    DM6248 = 6
    DM8006 = 7
    DM8009 = 8
    DM10010L = 9
    DM10010 = 10
    DMH3510 = 11
    DMH6215 = 12
    DMS3519 = 13
    DMG6220 = 14


class Control_Mode(IntEnum):
    MIT_MODE = 0x000
    POS_VEL_MODE = 0x100
    VEL_MODE = 0x200
    POS_FORCE_MODE = 0x300


class Control_Mode_Code(IntEnum):
    MIT = 1
    POS_VEL = 2
    VEL = 3
    POS_FORCE = 4


LIMIT_PARAM = [
    [12.566, 50, 5],
    [12.5, 30, 10],
    [12.5, 50, 10],
    [12.5, 10, 28],
    [12.5, 20, 28],
    [12.5, 45, 12],
    [12.566, 20, 120],
    [12.5, 45, 20],
    [12.5, 45, 54],
    [12.5, 25, 200],
    [12.5, 20, 200],
    [12.5, 280, 1],
    [12.5, 45, 10],
    [12.5, 2000, 2],
    [12.5, 45, 10],
]


@dataclass(frozen=True)
class MotorLimits:
    pmax: float
    vmax: float
    tmax: float


MOTOR_LIMITS = {
    motor_type: MotorLimits(*LIMIT_PARAM[motor_type.value])
    for motor_type in DM_Motor_Type
}


STATE_CODE_LABELS = {
    0x0: "disabled",
    0x1: "enabled",
    0x8: "overvoltage",
    0x9: "undervoltage",
    0xA: "overcurrent",
    0xB: "mos_overtemp",
    0xC: "rotor_overtemp",
    0xD: "comm_lost",
    0xE: "overload",
}


RUNNING = threading.Event()
RUNNING.set()


def signal_handler(signum, frame):
    _ = frame
    RUNNING.clear()
    sys.stderr.write(f"\nInterrupt signal ({signum}) received.\n")
    sys.stderr.flush()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@dataclass
class MotorConfig:
    motor_type: DM_Motor_Type
    can_id: int
    mst_id: int
    mode: Control_Mode


@dataclass
class MotorFeedback:
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    delta_time: float = 0.0
    last_time: float = 0.0
    controller_id: int = 0
    state_code: int = 0
    mos_temp: float = 0.0
    rotor_temp: float = 0.0

    def update(
        self,
        position: float,
        velocity: float,
        torque: float,
        *,
        controller_id: int = 0,
        state_code: int = 0,
        mos_temp: float = 0.0,
        rotor_temp: float = 0.0,
    ) -> None:
        now = time.monotonic()
        self.delta_time = 0.0 if self.last_time == 0.0 else now - self.last_time
        self.last_time = now
        self.position = float(position)
        self.velocity = float(velocity)
        self.torque = float(torque)
        self.controller_id = int(controller_id)
        self.state_code = int(state_code)
        self.mos_temp = float(mos_temp)
        self.rotor_temp = float(rotor_temp)


@dataclass(frozen=True)
class DecodedFeedbackFrame:
    motor_id: int
    can_id: int
    mst_id: int
    state: int
    controller_id: int
    position: float
    velocity: float
    torque: float
    mos_temperature: float
    rotor_temperature: float


@dataclass(frozen=True)
class _DamiaoMotorMapping:
    motor_id: int
    can_id: int
    mst_id: int
    motor_type: DM_Motor_Type


def int_auto(value: str) -> int:
    return int(value, 0)


def float_to_uint(value: float, xmin: float, xmax: float, bits: int) -> int:
    if xmax <= xmin:
        raise ValueError("xmax must be larger than xmin")
    clamped = min(max(float(value), xmin), xmax)
    scale = (1 << bits) - 1
    return int((clamped - xmin) / (xmax - xmin) * scale)


def uint_to_float(value: int, xmin: float, xmax: float, bits: int) -> float:
    scale = (1 << bits) - 1
    return ((float(value) / scale) * (xmax - xmin)) + xmin


def mode_to_code(mode: Control_Mode) -> Control_Mode_Code:
    mapping = {
        Control_Mode.MIT_MODE: Control_Mode_Code.MIT,
        Control_Mode.POS_VEL_MODE: Control_Mode_Code.POS_VEL,
        Control_Mode.VEL_MODE: Control_Mode_Code.VEL,
        Control_Mode.POS_FORCE_MODE: Control_Mode_Code.POS_FORCE,
    }
    return mapping[mode]


def parse_mode(value: str) -> Control_Mode:
    mapping = {
        "mit": Control_Mode.MIT_MODE,
        "pos-vel": Control_Mode.POS_VEL_MODE,
        "vel": Control_Mode.VEL_MODE,
        "pos-force": Control_Mode.POS_FORCE_MODE,
    }
    return mapping[value]


def parse_motor_type(value: str) -> DM_Motor_Type:
    return DM_Motor_Type[value]


def get_motor_limits(motor_type: DM_Motor_Type | str) -> MotorLimits:
    if isinstance(motor_type, str):
        motor_type = DM_Motor_Type[str(motor_type)]
    return MOTOR_LIMITS[DM_Motor_Type(motor_type)]


def pack_can_frame(can_id: int, payload: bytes) -> bytes:
    if len(payload) > 8:
        raise ValueError("Classic CAN payload must be 8 bytes or fewer")
    return struct.pack("=IB3x8s", int(can_id), len(payload), payload.ljust(8, b"\x00"))


def pack_canfd_frame(can_id: int, payload: bytes, flags: int = 0) -> bytes:
    if len(payload) > 64:
        raise ValueError("CAN FD payload must be 64 bytes or fewer")
    return struct.pack("=IBB2x64s", int(can_id), len(payload), int(flags), payload.ljust(64, b"\x00"))


def unpack_can_packet(packet: bytes) -> tuple[int, bytes]:
    if len(packet) == CAN_MTU:
        can_id, can_dlc, data = struct.unpack("=IB3x8s", packet)
        return can_id & socket.CAN_SFF_MASK, data[:can_dlc]
    if len(packet) == CANFD_MTU:
        can_id, length, _, data = struct.unpack("=IBB2x64s", packet)
        return can_id & socket.CAN_SFF_MASK, data[:length]
    raise ValueError(f"Unsupported CAN packet size: {len(packet)}")


def build_control_cmd_frame(can_id: int, cmd: int) -> tuple[int, bytes]:
    return int(can_id), bytes([0xFF] * 7 + [int(cmd)])


def build_param_read_frame(can_id: int, rid: int) -> tuple[int, bytes]:
    return 0x7FF, bytes([can_id & 0xFF, (can_id >> 8) & 0xFF, 0x33, rid, 0x00, 0x00, 0x00, 0x00])


def build_param_write_frame(can_id: int, rid: int, data: bytes) -> tuple[int, bytes]:
    if len(data) != 4:
        raise ValueError("Motor parameter writes require exactly 4 data bytes")
    return 0x7FF, bytes([can_id & 0xFF, (can_id >> 8) & 0xFF, 0x55, rid, *data])


def build_vel_frame(can_id: int, velocity: float) -> tuple[int, bytes]:
    return int(can_id) + Control_Mode.VEL_MODE, struct.pack("<f", float(velocity))


def build_pos_vel_frame(can_id: int, position: float, velocity: float) -> tuple[int, bytes]:
    return int(can_id) + Control_Mode.POS_VEL_MODE, struct.pack("<ff", float(position), float(velocity))


def build_mit_frame(
    can_id: int,
    motor_type: DM_Motor_Type,
    kp: float,
    kd: float,
    position: float,
    velocity: float,
    torque: float,
) -> tuple[int, bytes]:
    limits = get_motor_limits(motor_type)
    kp_uint = float_to_uint(kp, 0, 500, 12)
    kd_uint = float_to_uint(kd, 0, 5, 12)
    q_uint = float_to_uint(position, -limits.pmax, limits.pmax, 16)
    dq_uint = float_to_uint(velocity, -limits.vmax, limits.vmax, 12)
    tau_uint = float_to_uint(torque, -limits.tmax, limits.tmax, 12)
    data = bytes(
        [
            (q_uint >> 8) & 0xFF,
            q_uint & 0xFF,
            (dq_uint >> 4) & 0xFF,
            ((dq_uint & 0x0F) << 4) | ((kp_uint >> 8) & 0x0F),
            kp_uint & 0xFF,
            (kd_uint >> 4) & 0xFF,
            ((kd_uint & 0x0F) << 4) | ((tau_uint >> 8) & 0x0F),
            tau_uint & 0xFF,
        ]
    )
    return int(can_id) + Control_Mode.MIT_MODE, data


def decode_feedback(data: bytes, motor_type: DM_Motor_Type) -> MotorFeedback:
    if len(data) < 8:
        raise ValueError("Motor feedback requires 8 bytes")
    limits = get_motor_limits(motor_type)
    controller_id = data[0] & 0x0F
    state_code = (data[0] >> 4) & 0x0F
    q_uint = (data[1] << 8) | data[2]
    dq_uint = (data[3] << 4) | (data[4] >> 4)
    tau_uint = ((data[4] & 0x0F) << 8) | data[5]
    return MotorFeedback(
        position=uint_to_float(q_uint, -limits.pmax, limits.pmax, 16),
        velocity=uint_to_float(dq_uint, -limits.vmax, limits.vmax, 12),
        torque=uint_to_float(tau_uint, -limits.tmax, limits.tmax, 12),
        controller_id=controller_id,
        state_code=state_code,
        mos_temp=float(data[6]),
        rotor_temp=float(data[7]),
    )


class SocketCanTransport:
    def __init__(self, interface: str, force_fd: bool = True, fd_flags: int = CANFD_BRS):
        self.interface = str(interface)
        self.force_fd = bool(force_fd)
        self.fd_flags = int(fd_flags)
        self.socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.socket.setsockopt(SOL_CAN_RAW, CAN_RAW_FD_FRAMES, 1)
        self.socket.settimeout(0.1)
        self.socket.bind((self.interface,))

    def close(self) -> None:
        self.socket.close()

    def send(self, can_id: int, payload: bytes) -> None:
        if self.force_fd:
            packet = pack_canfd_frame(int(can_id), payload, flags=self.fd_flags)
        else:
            packet = pack_can_frame(int(can_id), payload) if len(payload) <= 8 else pack_canfd_frame(int(can_id), payload)
        self.socket.send(packet)

    def recv(self, timeout: float = 0.1) -> Optional[tuple[int, bytes]]:
        try:
            ready, _, _ = select.select([self.socket], [], [], float(timeout))
            if not ready:
                return None
            packet = self.socket.recv(CANFD_MTU)
        except socket.timeout:
            return None
        return unpack_can_packet(packet)


class SocketCanMotorController:
    def __init__(self, interface: str, motor: MotorConfig):
        self.transport = SocketCanTransport(interface)
        self.motor = motor
        self.feedback = MotorFeedback()
        self._rx_running = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._rx_running.set()
        self._rx_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._rx_thread.start()

    def close(self) -> None:
        self.disable()
        self._rx_running.clear()
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
        self.transport.close()

    def _recv_loop(self) -> None:
        while self._rx_running.is_set():
            packet = self.transport.recv()
            if packet is None:
                continue
            can_id, data = packet
            if can_id not in (self.motor.can_id, self.motor.mst_id):
                continue
            if len(data) >= 6 and data[2] not in (0x33, 0x55, 0xAA):
                decoded = decode_feedback(data, self.motor.motor_type)
                with self._lock:
                    self.feedback.update(
                        decoded.position,
                        decoded.velocity,
                        decoded.torque,
                        controller_id=decoded.controller_id,
                        state_code=decoded.state_code,
                        mos_temp=decoded.mos_temp,
                        rotor_temp=decoded.rotor_temp,
                    )

    def send(self, can_id: int, payload: bytes) -> None:
        self.transport.send(can_id, payload)

    def switch_control_mode(self, mode: Control_Mode) -> None:
        can_id, payload = build_param_write_frame(self.motor.can_id, 10, bytes([mode_to_code(mode), 0x00, 0x00, 0x00]))
        self.send(can_id, payload)
        self.motor.mode = mode
        time.sleep(0.01)

    def enable(self) -> None:
        self.switch_control_mode(self.motor.mode)
        control_id = self.motor.can_id + self.motor.mode
        can_id, payload = build_control_cmd_frame(control_id, ENABLE_CMD)
        for _ in range(5):
            self.send(can_id, payload)
            time.sleep(0.002)

    def disable(self) -> None:
        control_id = self.motor.can_id + self.motor.mode
        can_id, payload = build_control_cmd_frame(control_id, DISABLE_CMD)
        for _ in range(5):
            try:
                self.send(can_id, payload)
            except OSError:
                break
            time.sleep(0.002)

    def set_zero_position(self) -> None:
        can_id, payload = build_control_cmd_frame(self.motor.can_id + self.motor.mode, ZERO_CMD)
        self.send(can_id, payload)
        time.sleep(0.002)

    def control_velocity(self, velocity: float) -> None:
        can_id, payload = build_vel_frame(self.motor.can_id, velocity)
        self.send(can_id, payload)

    def control_position_velocity(self, position: float, velocity: float) -> None:
        can_id, payload = build_pos_vel_frame(self.motor.can_id, position, velocity)
        self.send(can_id, payload)

    def control_mit(self, kp: float, kd: float, position: float, velocity: float, torque: float) -> None:
        can_id, payload = build_mit_frame(self.motor.can_id, self.motor.motor_type, kp, kd, position, velocity, torque)
        self.send(can_id, payload)

    def read_feedback(self) -> MotorFeedback:
        with self._lock:
            return MotorFeedback(
                position=self.feedback.position,
                velocity=self.feedback.velocity,
                torque=self.feedback.torque,
                delta_time=self.feedback.delta_time,
                last_time=self.feedback.last_time,
                controller_id=self.feedback.controller_id,
                state_code=self.feedback.state_code,
                mos_temp=self.feedback.mos_temp,
                rotor_temp=self.feedback.rotor_temp,
            )


class DamiaoSocketCanTransport:
    def __init__(self, config: Any, *, can_transport: Any | None = None) -> None:
        self._root_config = config
        self._config = getattr(config, "transport", config)
        self._pending = bytearray()
        self._decoded_frames: deque[DecodedFeedbackFrame] = deque()
        self._motor_mappings = self._build_motor_mappings()
        self._feedback_mapping = self._build_feedback_mapping(self._motor_mappings)
        self._current_modes: dict[int, Control_Mode] = {}
        self._last_mit_velocity_kd: dict[int, float] = {}
        self._transport = can_transport
        self._closed = False
        if self._transport is None:
            if bool(self._config.configure_interface):
                configure_can_interface(
                    self._config.interface,
                    self._config.nominal_bitrate,
                    self._config.data_bitrate,
                )
            ensure_interface_ready(
                self._config.interface,
                self._config.nominal_bitrate,
                self._config.data_bitrate,
            )
            self._transport = SocketCanTransport(
                self._config.interface,
                force_fd=bool(self._config.force_fd),
            )

    def _motor_ids(self) -> tuple[int, ...]:
        if hasattr(self._root_config, "motor_ids"):
            return tuple(int(motor_id) for motor_id in self._root_config.motor_ids)
        return tuple(range(1, len(self._config.motor_can_ids) + 1))

    def _build_motor_mappings(self) -> tuple[_DamiaoMotorMapping, ...]:
        motor_ids = self._motor_ids()
        mappings: list[_DamiaoMotorMapping] = []
        for index, motor_id in enumerate(motor_ids):
            mappings.append(
                _DamiaoMotorMapping(
                    motor_id=int(motor_id),
                    can_id=int(self._config.motor_can_ids[index]),
                    mst_id=int(self._config.motor_mst_ids[index]),
                    motor_type=DM_Motor_Type[str(self._config.motor_types[index])],
                )
            )
        return tuple(mappings)

    @staticmethod
    def _build_feedback_mapping(mappings: tuple[_DamiaoMotorMapping, ...]) -> dict[int, _DamiaoMotorMapping]:
        feedback_mapping: dict[int, _DamiaoMotorMapping] = {}
        for mapping in mappings:
            feedback_mapping[int(mapping.can_id)] = mapping
            feedback_mapping[int(mapping.mst_id)] = mapping
        return feedback_mapping

    def _trace_packet(self, can_id: int, payload: bytes) -> bytes:
        if bool(self._config.force_fd):
            return pack_canfd_frame(int(can_id), payload, flags=CANFD_BRS)
        if len(payload) <= 8:
            return pack_can_frame(int(can_id), payload)
        return pack_canfd_frame(int(can_id), payload)

    def _send_with_retry(self, can_id: int, payload: bytes) -> bytes:
        send_frame(self._transport, (int(can_id), bytes(payload)))
        return self._trace_packet(int(can_id), bytes(payload))

    def _mapping_for_motor_id(self, motor_id: int) -> _DamiaoMotorMapping:
        target_motor_id = self._validate_motor_id(int(motor_id))
        for mapping in self._motor_mappings:
            if int(mapping.motor_id) == target_motor_id:
                return mapping
        raise ValueError(f"Unknown motor_id: {target_motor_id}")

    def _mode_for_semantic(self, semantic_mode: str) -> Control_Mode:
        if semantic_mode in {"mit_torque", "mit_velocity"}:
            return Control_Mode.MIT_MODE
        if semantic_mode == "velocity_mode":
            return Control_Mode.VEL_MODE
        raise ValueError(f"Unsupported semantic_mode: {semantic_mode}")

    def _get_active_mode(self, mapping: _DamiaoMotorMapping, fallback: Control_Mode = Control_Mode.MIT_MODE) -> Control_Mode:
        return self._current_modes.get(int(mapping.motor_id), fallback)

    def _ensure_mode(self, mapping: _DamiaoMotorMapping, mode: Control_Mode) -> bytes:
        if self._get_active_mode(mapping, mode) == mode and int(mapping.motor_id) in self._current_modes:
            return b""
        can_id, payload = build_param_write_frame(
            int(mapping.can_id),
            10,
            bytes([int(mode_to_code(mode)), 0x00, 0x00, 0x00]),
        )
        packet = self._send_with_retry(can_id, payload)
        self._current_modes[int(mapping.motor_id)] = Control_Mode(mode)
        time.sleep(DEFAULT_PARAM_WRITE_SETTLE)
        return packet

    def _send_control(self, mapping: _DamiaoMotorMapping, cmd: int, *, mode: Control_Mode | None = None) -> bytes:
        active_mode = self._get_active_mode(mapping) if mode is None else Control_Mode(mode)
        can_id, payload = build_control_cmd_frame(int(mapping.can_id) + int(active_mode), int(cmd))
        packets: list[bytes] = []
        for _ in range(DEFAULT_CONTROL_COMMAND_REPEAT):
            try:
                packets.append(self._send_with_retry(can_id, payload))
            except OSError:
                break
            time.sleep(DEFAULT_CONTROL_COMMAND_INTERVAL)
        return b"".join(packets)

    def _validate_motor_id(self, motor_id: int) -> int:
        valid_ids = self._motor_ids()
        motor_id = int(motor_id)
        if motor_id not in valid_ids:
            raise ValueError(f"motor_id must be within {valid_ids}.")
        return motor_id

    def _append_feedback_frame(self, can_id: int, payload: bytes) -> None:
        if len(payload) < 6:
            return
        if len(payload) >= 3 and payload[2] in (0x33, 0x55, 0xAA):
            return
        mapping = self._feedback_mapping.get(int(can_id))
        if mapping is None:
            return
        try:
            decoded = decode_feedback(payload, mapping.motor_type)
        except Exception as exc:
            raise ValueError(
                f"feedback_decode_error motor_id={int(mapping.motor_id)} can_id=0x{int(can_id):03X}: {exc}"
            ) from exc
        if int(decoded.state_code) not in VALID_FEEDBACK_STATE_CODES:
            raise ValueError(
                f"feedback_state_error motor_id={int(mapping.motor_id)} can_id=0x{int(can_id):03X} "
                f"state=0x{int(decoded.state_code):X}"
            )
        frame = DecodedFeedbackFrame(
            motor_id=int(mapping.motor_id),
            can_id=int(mapping.can_id),
            mst_id=int(mapping.mst_id),
            state=int(decoded.state_code),
            controller_id=int(decoded.controller_id),
            position=float(decoded.position),
            velocity=float(decoded.velocity),
            torque=float(decoded.torque),
            mos_temperature=float(decoded.mos_temp),
            rotor_temperature=float(decoded.rotor_temp),
        )
        self._decoded_frames.append(frame)
        self._pending.extend(
            SERIALIZED_FEEDBACK_STRUCT.pack(
                SERIALIZED_FEEDBACK_HEAD,
                int(frame.motor_id),
                int(frame.state),
                float(frame.position),
                float(frame.velocity),
                float(frame.torque),
                float(frame.mos_temperature),
            )
        )

    def read(self, size: int) -> bytes:
        target_size = max(int(size), 1)
        while len(self._pending) < target_size:
            packet = self._transport.recv(timeout=float(self._config.read_timeout))
            if packet is None:
                break
            can_id, payload = packet
            self._append_feedback_frame(int(can_id), bytes(payload))
        chunk = bytes(self._pending[:target_size])
        del self._pending[:target_size]
        return chunk

    def pop_feedback_frame(self) -> DecodedFeedbackFrame | None:
        if not self._decoded_frames:
            return None
        return self._decoded_frames.popleft()

    def motor_type_name(self, motor_id: int) -> str:
        return self._mapping_for_motor_id(motor_id).motor_type.name

    def motor_limits(self, motor_id: int) -> MotorLimits:
        return get_motor_limits(self._mapping_for_motor_id(motor_id).motor_type)

    def limit_torque_command(self, motor_id: int, torque: float) -> float:
        limits = self.motor_limits(motor_id)
        return float(np.clip(float(torque), -float(limits.tmax), float(limits.tmax)))

    def send_mit_torque(self, motor_id: int, torque: float) -> bytes:
        if not np.isfinite(float(torque)):
            raise ValueError("torque must be finite")
        mapping = self._mapping_for_motor_id(motor_id)
        packet = bytearray()
        packet.extend(self._ensure_mode(mapping, Control_Mode.MIT_MODE))
        limited_torque = self.limit_torque_command(motor_id, float(torque))
        can_id, payload = build_mit_frame(
            int(mapping.can_id),
            mapping.motor_type,
            0.0,
            0.0,
            0.0,
            0.0,
            float(limited_torque),
        )
        packet.extend(self._send_with_retry(can_id, payload))
        return bytes(packet)

    def send_mit_velocity(
        self,
        motor_id: int,
        velocity: float,
        kd: float,
        *,
        kp: float = 0.0,
        torque_ff: float = 0.0,
        position: float = 0.0,
    ) -> bytes:
        if not np.isfinite(float(velocity)):
            raise ValueError("velocity must be finite")
        mapping = self._mapping_for_motor_id(motor_id)
        packet = bytearray()
        packet.extend(self._ensure_mode(mapping, Control_Mode.MIT_MODE))
        limited_torque = self.limit_torque_command(motor_id, float(torque_ff))
        can_id, payload = build_mit_frame(
            int(mapping.can_id),
            mapping.motor_type,
            float(kp),
            float(kd),
            float(position),
            float(velocity),
            float(limited_torque),
        )
        packet.extend(self._send_with_retry(can_id, payload))
        self._last_mit_velocity_kd[int(mapping.motor_id)] = float(kd)
        return bytes(packet)

    def send_velocity_mode(self, motor_id: int, velocity: float) -> bytes:
        if not np.isfinite(float(velocity)):
            raise ValueError("velocity must be finite")
        mapping = self._mapping_for_motor_id(motor_id)
        packet = bytearray()
        packet.extend(self._ensure_mode(mapping, Control_Mode.VEL_MODE))
        can_id, payload = build_vel_frame(int(mapping.can_id), float(velocity))
        packet.extend(self._send_with_retry(can_id, payload))
        return bytes(packet)

    def send_zero_command(self, motor_id: int, semantic_mode: str) -> bytes:
        if semantic_mode == "mit_torque":
            return self.send_mit_torque(motor_id, 0.0)
        if semantic_mode == "mit_velocity":
            kd = float(self._last_mit_velocity_kd.get(int(motor_id), 0.0))
            return self.send_mit_velocity(motor_id, 0.0, kd, kp=0.0, torque_ff=0.0, position=0.0)
        if semantic_mode == "velocity_mode":
            return self.send_velocity_mode(motor_id, 0.0)
        raise ValueError(f"Unsupported semantic_mode: {semantic_mode}")

    def enable_motor(self, motor_id: int) -> bytes:
        mapping = self._mapping_for_motor_id(motor_id)
        packet = bytearray()
        packet.extend(self._ensure_mode(mapping, self._get_active_mode(mapping)))
        packet.extend(self._send_control(mapping, ENABLE_CMD))
        return bytes(packet)

    def disable_motor(self, motor_id: int) -> bytes:
        mapping = self._mapping_for_motor_id(motor_id)
        return self._send_control(mapping, DISABLE_CMD)

    def clear_error(self, motor_id: int) -> bytes:
        mapping = self._mapping_for_motor_id(motor_id)
        return self._send_control(mapping, CLEAR_ERROR_CMD)

    def send_motor_torque(self, motor_id: int, torque: float) -> bytes:
        return self.send_mit_torque(motor_id, torque)

    def reset_input_buffer(self) -> None:
        self._pending.clear()
        self._decoded_frames.clear()
        while True:
            packet = self._transport.recv(timeout=0.0)
            if packet is None:
                break

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for mapping in self._motor_mappings:
                try:
                    self._send_control(mapping, DISABLE_CMD)
                except OSError:
                    continue
        finally:
            self._transport.close()


def configure_can_interface(interface: str, nominal_bitrate: int, data_bitrate: int) -> None:
    commands = [
        ["ip", "link", "set", interface, "down"],
        ["ip", "link", "set", interface, "type", "can", "bitrate", str(nominal_bitrate), "dbitrate", str(data_bitrate), "fd", "on"],
        ["ip", "link", "set", interface, "up"],
    ]
    for cmd in commands:
        subprocess.run(cmd, check=True)


def ensure_interface_ready(interface: str, nominal_bitrate: int, data_bitrate: int) -> None:
    path = f"/sys/class/net/{interface}/operstate"
    if not os.path.exists(path):
        raise RuntimeError(f"CAN interface {interface} does not exist")
    with open(path, "r", encoding="utf-8") as file:
        state = file.read().strip()
    if state != "up":
        raise RuntimeError(
            f"{interface} 当前不是 UP 状态。先执行:\n"
            f"  sudo ip link set {interface} down\n"
            f"  sudo ip link set {interface} type can bitrate {nominal_bitrate} dbitrate {data_bitrate} fd on\n"
            f"  sudo ip link set {interface} up"
        )


def send_frame(transport: SocketCanTransport, frame: tuple[int, bytes]) -> None:
    can_id, payload = frame
    backpressure_sleep = DEFAULT_BACKPRESSURE_SLEEP
    while True:
        try:
            transport.send(can_id, payload)
            return
        except OSError as exc:
            if exc.errno != errno.ENOBUFS:
                raise
            time.sleep(backpressure_sleep)
            backpressure_sleep = min(backpressure_sleep * 2.0, MAX_BACKPRESSURE_SLEEP)


def send_repeated_frame(transport: SocketCanTransport, frame: tuple[int, bytes], count: int, interval_seconds: float) -> None:
    for _ in range(int(count)):
        send_frame(transport, frame)
        time.sleep(float(interval_seconds))


def build_enable_frame(can_id: int) -> tuple[int, bytes]:
    return build_control_cmd_frame(int(can_id) + Control_Mode.MIT_MODE, ENABLE_CMD)


def build_disable_frame(can_id: int) -> tuple[int, bytes]:
    return build_control_cmd_frame(int(can_id) + Control_Mode.MIT_MODE, DISABLE_CMD)


def build_clear_error_frame(can_id: int) -> tuple[int, bytes]:
    return build_control_cmd_frame(int(can_id) + Control_Mode.MIT_MODE, CLEAR_ERROR_CMD)


def build_zero_mit_frame(can_id: int, motor_type: DM_Motor_Type) -> tuple[int, bytes]:
    return build_mit_frame(int(can_id), motor_type, 0.0, 0.0, 0.0, 0.0, 0.0)


def build_feedback_views(base_path: str) -> list:
    return [
        rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/position"], name="Position"),
        rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/velocity"], name="Velocity"),
        rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/torque"], name="Torque"),
        rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/state_code"], name="State Code"),
        rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/send_rate_hz"], name="Send Rate"),
        rrb.TimeSeriesView(
            origin="/",
            contents=[f"{base_path}/mos_temp", f"{base_path}/rotor_temp"],
            name="Temperatures",
        ),
        rrb.TextLogView(origin=f"{base_path}/events", name="Motor Events"),
    ]


def build_rerun_blueprint(base_path: str) -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(contents=build_feedback_views(base_path)),
        collapse_panels=True,
    )


def setup_rerun(base_path: str, application_id: str) -> None:
    rr.init(application_id, spawn=True, default_blueprint=build_rerun_blueprint(base_path))
    rr.log(f"{base_path}/position", rr.SeriesLines(colors=[[255, 99, 71]], names=["position"], widths=[2]), static=True)
    rr.log(f"{base_path}/velocity", rr.SeriesLines(colors=[[30, 144, 255]], names=["velocity"], widths=[2]), static=True)
    rr.log(f"{base_path}/torque", rr.SeriesLines(colors=[[60, 179, 113]], names=["torque"], widths=[2]), static=True)
    rr.log(f"{base_path}/state_code", rr.SeriesLines(colors=[[255, 215, 0]], names=["state_code"], widths=[2]), static=True)
    rr.log(f"{base_path}/send_rate_hz", rr.SeriesLines(colors=[[138, 43, 226]], names=["send_rate_hz"], widths=[2]), static=True)
    rr.log(f"{base_path}/mos_temp", rr.SeriesLines(colors=[[255, 140, 0]], names=["mos_temp"], widths=[2]), static=True)
    rr.log(f"{base_path}/rotor_temp", rr.SeriesLines(colors=[[220, 20, 60]], names=["rotor_temp"], widths=[2]), static=True)


def log_feedback_to_rerun(base_path: str, elapsed_seconds: float, feedback: MotorFeedback | DecodedFeedbackFrame) -> None:
    rr.set_time("feedback_time", duration=elapsed_seconds)
    rr.log(f"{base_path}/position", rr.Scalars([feedback.position]))
    rr.log(f"{base_path}/velocity", rr.Scalars([feedback.velocity]))
    rr.log(f"{base_path}/torque", rr.Scalars([feedback.torque]))
    state_code = getattr(feedback, "state_code", getattr(feedback, "state"))
    mos_temp = getattr(feedback, "mos_temp", getattr(feedback, "mos_temperature"))
    rotor_temp = getattr(feedback, "rotor_temp", getattr(feedback, "rotor_temperature"))
    controller_id = int(getattr(feedback, "controller_id", 0))
    rr.log(f"{base_path}/state_code", rr.Scalars([state_code]))
    rr.log(f"{base_path}/mos_temp", rr.Scalars([mos_temp]))
    rr.log(f"{base_path}/rotor_temp", rr.Scalars([rotor_temp]))
    state_label = STATE_CODE_LABELS.get(int(state_code), f"unknown_{int(state_code):X}")
    rr.log(
        f"{base_path}/events",
        rr.TextLog(
            (
                f"state={state_label} controller_id=0x{controller_id:02X} "
                f"pos={float(feedback.position):.4f} vel={float(feedback.velocity):.4f} "
                f"tau={float(feedback.torque):.4f} mos_temp={float(mos_temp):.1f} "
                f"rotor_temp={float(rotor_temp):.1f}"
            ),
            level="INFO",
        ),
    )


def log_send_rate_to_rerun(base_path: str, elapsed_seconds: float, send_rate_hz: float) -> None:
    rr.set_time("feedback_time", duration=elapsed_seconds)
    rr.log(f"{base_path}/send_rate_hz", rr.Scalars([float(send_rate_hz)]))


def build_socketcan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过 Linux SocketCAN 直接控制达妙电机")
    parser.add_argument("--interface", default="can0", help="SocketCAN 接口名，默认 can0")
    parser.add_argument("--can-id", type=int_auto, required=True, help="电机 CAN ID，例如 0x01")
    parser.add_argument("--mst-id", type=int_auto, required=True, help="电机反馈 ID，例如 0x11")
    parser.add_argument("--motor-type", default="DM4310", choices=[member.name for member in DM_Motor_Type], help="电机型号")
    parser.add_argument("--mode", default="vel", choices=["mit", "pos-vel", "vel", "pos-force"], help="控制模式")
    parser.add_argument("--vel", type=float, default=2.0, help="速度模式目标值")
    parser.add_argument("--pos", type=float, default=0.0, help="位置模式目标值")
    parser.add_argument("--kp", type=float, default=0.0, help="MIT 模式 kp")
    parser.add_argument("--kd", type=float, default=0.0, help="MIT 模式 kd")
    parser.add_argument("--tau", type=float, default=0.0, help="MIT 模式力矩")
    parser.add_argument("--rate", type=float, default=1000.0, help="发送频率，默认 1000Hz")
    parser.add_argument("--duration", type=float, default=5.0, help="持续时间，单位秒")
    parser.add_argument("--print-interval", type=float, default=0.2, help="打印反馈间隔，单位秒")
    parser.add_argument("--nom-bitrate", type=int, default=1000000, help="CAN 仲裁域波特率")
    parser.add_argument("--data-bitrate", type=int, default=5000000, help="CAN FD 数据域波特率")
    parser.add_argument("--configure-interface", action="store_true", help="启动前尝试配置并拉起接口")
    parser.add_argument("--set-zero", action="store_true", help="使能后发送零点设置命令")
    return parser


def socketcan_main(argv: Optional[list[str]] = None) -> int:
    args = build_socketcan_parser().parse_args(argv)
    if args.configure_interface:
        configure_can_interface(args.interface, args.nom_bitrate, args.data_bitrate)
    ensure_interface_ready(args.interface, args.nom_bitrate, args.data_bitrate)
    motor = MotorConfig(
        motor_type=parse_motor_type(args.motor_type),
        can_id=args.can_id,
        mst_id=args.mst_id,
        mode=parse_mode(args.mode),
    )
    controller = SocketCanMotorController(args.interface, motor)
    try:
        controller.start()
        controller.enable()
        if args.set_zero:
            controller.set_zero_position()
        period = 1.0 / args.rate
        end_time = time.perf_counter() + args.duration
        next_print = time.perf_counter()
        while RUNNING.is_set() and time.perf_counter() < end_time:
            loop_start = time.perf_counter()
            if motor.mode == Control_Mode.VEL_MODE:
                controller.control_velocity(args.vel)
            elif motor.mode == Control_Mode.POS_VEL_MODE:
                controller.control_position_velocity(args.pos, args.vel)
            elif motor.mode == Control_Mode.MIT_MODE:
                controller.control_mit(args.kp, args.kd, args.pos, args.vel, args.tau)
            else:
                raise RuntimeError("POS_FORCE_MODE 暂未实现直接控制命令")
            now = time.perf_counter()
            if now >= next_print:
                feedback = controller.read_feedback()
                print(
                    f"canid={motor.can_id} pos={feedback.position:.4f} "
                    f"vel={feedback.velocity:.4f} tau={feedback.torque:.4f} "
                    f"dt={feedback.delta_time:.6f}"
                )
                next_print = now + args.print_interval
            sleep_time = period - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        controller.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="持续向 DM8009 id=0x01 发送使能帧并实时监听反馈（默认无明确命令内容时持续发送使能）"
    )
    parser.add_argument("--mst-id", type=lambda value: int(value, 0), default=DEFAULT_MST_ID, help="反馈帧 ID，默认 0x11")
    parser.add_argument(
        "--mode",
        default="mit",
        choices=["mit", "pos-vel", "vel", "pos-force"],
        help="保留兼容参数；当前脚本仅支持 mit，默认 mit",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="兼容旧参数；当前默认使能脚本会忽略它",
    )
    parser.add_argument("--count", type=int, default=5, help="初始使能帧重复发送次数，默认 5 次")
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=DEFAULT_COMMAND_INTERVAL_MS,
        help="默认使能帧发送周期；<=0 表示不主动休眠、尽可能高频发送，默认 0ms",
    )
    parser.add_argument("--nom-bitrate", type=int, default=1000000, help="CAN 仲裁域波特率")
    parser.add_argument("--data-bitrate", type=int, default=5000000, help="CAN FD 数据域波特率")
    parser.add_argument("--configure-interface", action="store_true", help="启动前尝试配置并拉起接口")
    parser.add_argument(
        "--listen-duration",
        type=float,
        default=DEFAULT_LISTEN_DURATION,
        help="连续发送控制帧的时长；<=0 表示一直运行直到 Ctrl+C，默认 0 秒",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=DEFAULT_PRINT_INTERVAL,
        help="终端反馈打印间隔，单位秒；<=0 表示收到反馈就打印，默认 0.1 秒",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将要发送的帧，不真正发包")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode != "mit":
        raise RuntimeError("当前脚本仅支持 mit 模式连续使能流")
    if args.configure_interface:
        configure_can_interface(DEFAULT_INTERFACE, args.nom_bitrate, args.data_bitrate)
    if not args.dry_run:
        ensure_interface_ready(DEFAULT_INTERFACE, args.nom_bitrate, args.data_bitrate)
    base_path = f"/motor/dm8009_id_{DEFAULT_CAN_ID:02X}"
    motor_type = DM_Motor_Type[DEFAULT_MOTOR_TYPE]
    enable_frame = build_enable_frame(DEFAULT_CAN_ID)
    disable_frame = build_disable_frame(DEFAULT_CAN_ID)
    duration_label = "forever" if args.listen_duration <= 0 else str(args.listen_duration)
    print(
        f"target_motor={DEFAULT_MOTOR_TYPE} can_id=0x{DEFAULT_CAN_ID:02X} "
        f"mst_id=0x{args.mst_id:02X} interface={DEFAULT_INTERFACE} "
        f"nominal_bitrate={args.nom_bitrate} data_bitrate={args.data_bitrate} frame_type=canfd_brs "
        f"control_mode=mit command_interval_ms={args.interval_ms} listen_duration={duration_label} "
        f"default_behavior=enable_only"
    )
    if args.all_modes:
        print("warning: --all-modes is ignored in continuous enable mode")
    print(f"enable_frame id=0x{enable_frame[0]:03X} data={enable_frame[1].hex()}")
    print("default_command=enable_only")
    if args.dry_run:
        return 0
    setup_rerun(base_path, application_id="dm8009_enable_feedback")
    transport = SocketCanTransport(DEFAULT_INTERFACE)
    start_time = time.monotonic()
    end_time = None if args.listen_duration <= 0 else start_time + args.listen_duration
    command_interval_seconds = args.interval_ms / 1000.0 if args.interval_ms > 0 else None
    last_state_code = None
    next_feedback_print_at = start_time
    last_send_rate_log_time = start_time
    sends_since_rate_log = 0
    try:
        send_repeated_frame(transport, enable_frame, args.count, 0.002)
        while RUNNING.is_set() and (end_time is None or time.monotonic() < end_time):
            loop_start = time.perf_counter()
            send_frame(transport, enable_frame)
            sends_since_rate_log += 1
            loop_time = time.monotonic()
            send_rate_window = loop_time - last_send_rate_log_time
            if send_rate_window >= DEFAULT_SEND_RATE_LOG_INTERVAL:
                log_send_rate_to_rerun(base_path, loop_time - start_time, sends_since_rate_log / send_rate_window)
                last_send_rate_log_time = loop_time
                sends_since_rate_log = 0
            while True:
                packet = transport.recv(timeout=0.0)
                if packet is None:
                    break
                can_id, payload = packet
                if can_id != args.mst_id:
                    continue
                feedback = decode_feedback(payload, motor_type)
                feedback_time = time.monotonic()
                log_feedback_to_rerun(base_path, feedback_time - start_time, feedback)
                state_label = STATE_CODE_LABELS.get(feedback.state_code, f"unknown_{feedback.state_code:X}")
                if feedback.state_code != last_state_code:
                    print(f"feedback_state={state_label} controller_id=0x{feedback.controller_id:02X}")
                    last_state_code = feedback.state_code
                if args.print_interval <= 0 or feedback_time >= next_feedback_print_at:
                    print(
                        f"feedback pos={feedback.position:.4f} vel={feedback.velocity:.4f} "
                        f"tau={feedback.torque:.4f} mos={feedback.mos_temp:.1f} rotor={feedback.rotor_temp:.1f}"
                    )
                    if args.print_interval > 0:
                        next_feedback_print_at = feedback_time + args.print_interval
            if command_interval_seconds is not None:
                sleep_time = command_interval_seconds - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        try:
            send_repeated_frame(transport, disable_frame, args.count, 0.002)
        except OSError:
            pass
        transport.close()
    return 0


__all__ = [
    "CAN_MTU",
    "CANFD_MTU",
    "CANFD_BRS",
    "CLEAR_ERROR_CMD",
    "Control_Mode",
    "Control_Mode_Code",
    "DM_Motor_Type",
    "DamiaoSocketCanTransport",
    "DecodedFeedbackFrame",
    "DEFAULT_BACKPRESSURE_SLEEP",
    "DEFAULT_CAN_ID",
    "DEFAULT_COMMAND_INTERVAL_MS",
    "DEFAULT_INTERFACE",
    "DEFAULT_LISTEN_DURATION",
    "DEFAULT_MOTOR_TYPE",
    "DEFAULT_MST_ID",
    "DEFAULT_PRINT_INTERVAL",
    "ENABLE_CMD",
    "DISABLE_CMD",
    "LIMIT_PARAM",
    "MAX_BACKPRESSURE_SLEEP",
    "MotorConfig",
    "MotorFeedback",
    "RUNNING",
    "SERIALIZED_FEEDBACK_HEAD",
    "SERIALIZED_FEEDBACK_STRUCT",
    "SocketCanMotorController",
    "SocketCanTransport",
    "STATE_CODE_LABELS",
    "VALID_FEEDBACK_STATE_CODES",
    "build_control_cmd_frame",
    "build_clear_error_frame",
    "build_disable_frame",
    "build_enable_frame",
    "build_feedback_views",
    "build_mit_frame",
    "build_param_read_frame",
    "build_param_write_frame",
    "build_parser",
    "build_pos_vel_frame",
    "build_rerun_blueprint",
    "build_socketcan_parser",
    "build_vel_frame",
    "build_zero_mit_frame",
    "configure_can_interface",
    "decode_feedback",
    "ensure_interface_ready",
    "int_auto",
    "log_feedback_to_rerun",
    "log_send_rate_to_rerun",
    "main",
    "mode_to_code",
    "get_motor_limits",
    "MotorLimits",
    "MOTOR_LIMITS",
    "pack_can_frame",
    "pack_canfd_frame",
    "parse_mode",
    "parse_motor_type",
    "send_frame",
    "send_repeated_frame",
    "setup_rerun",
    "socketcan_main",
    "uint_to_float",
    "unpack_can_packet",
]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    except subprocess.CalledProcessError as exc:
        print(f"命令执行失败: {' '.join(exc.cmd)}", file=sys.stderr)
        raise SystemExit(exc.returncode) from None
