from __future__ import annotations

import errno
import glob
import math
import os
import select
import socket
import struct
import subprocess
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

CAN_MTU = 16
CANFD_MTU = 72
CAN_RAW_FD_FRAMES = getattr(socket, "CAN_RAW_FD_FRAMES", 5)
SOL_CAN_RAW = getattr(socket, "SOL_CAN_RAW", socket.SOL_CAN_BASE + socket.CAN_RAW)
CANFD_BRS = getattr(socket, "CANFD_BRS", 0x01)

CLEAR_ERROR_CMD = 0xFB
ENABLE_CMD = 0xFC
DISABLE_CMD = 0xFD
ZERO_POSITION_CMD = 0xFE
MIT_MODE = 0x000
MIT_MODE_CODE = 1

PARAM_MST_ID_RID = 7
PARAM_ESC_ID_RID = 8
PARAM_CTRL_MODE_RID = 10
PARAM_WRITE_CMD = 0x55
PARAM_STORE_CMD = 0xAA
PARAM_STORE_TO_FLASH = 0x01

ALLOWED_INTERFACES = ("can0", "can1")
DEFAULT_INTERFACE = "can0"
DEFAULT_NOMINAL_BITRATE = 1_000_000
DEFAULT_DATA_BITRATE = 5_000_000
DEFAULT_MOTOR_CAN_IDS = (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07)
DEFAULT_MOTOR_MST_IDS = (0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17)
DEFAULT_MOTOR_TYPES = ("DM8009", "DM8009", "DM4340", "DM4340", "DM4310", "DM4310", "DM4310")

CONTROL_REPEAT = 5
CONTROL_INTERVAL_SECONDS = 0.002
PARAM_WRITE_SETTLE_SECONDS = 0.002
PARAM_STORE_SETTLE_SECONDS = 0.05
BACKPRESSURE_SLEEP_SECONDS = 0.0005
MAX_BACKPRESSURE_SLEEP_SECONDS = 0.01


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


LIMIT_PARAM = (
    (12.566, 50.0, 5.0),
    (12.5, 30.0, 10.0),
    (12.5, 50.0, 10.0),
    (12.5, 10.0, 28.0),
    (12.5, 20.0, 28.0),
    (12.5, 45.0, 12.0),
    (12.566, 20.0, 120.0),
    (12.5, 45.0, 20.0),
    (12.5, 45.0, 54.0),
    (12.5, 25.0, 200.0),
    (12.5, 20.0, 200.0),
    (12.5, 280.0, 1.0),
    (12.5, 45.0, 10.0),
    (12.5, 2000.0, 2.0),
    (12.5, 45.0, 10.0),
)


@dataclass(frozen=True)
class MotorLimits:
    pmax: float
    vmax: float
    tmax: float


@dataclass(frozen=True)
class MotorSpec:
    motor_id: int
    can_id: int
    mst_id: int
    motor_type: DM_Motor_Type


@dataclass(frozen=True)
class MitCommand:
    position: float
    velocity: float
    kp: float
    kd: float
    torque_ff: float


@dataclass(frozen=True)
class MotorFeedback:
    motor_id: int
    can_id: int
    state: int
    controller_id: int
    position: float
    velocity: float
    torque: float
    mos_temperature: float
    rotor_temperature: float


class CanTransport(Protocol):
    def send(self, can_id: int, payload: bytes) -> None:
        ...

    def close(self) -> None:
        ...


MOTOR_LIMITS = {
    motor_type: MotorLimits(*LIMIT_PARAM[motor_type.value])
    for motor_type in DM_Motor_Type
}


def default_motor_specs() -> list[MotorSpec]:
    return [
        MotorSpec(
            motor_id=index + 1,
            can_id=can_id,
            mst_id=DEFAULT_MOTOR_MST_IDS[index],
            motor_type=parse_motor_type(DEFAULT_MOTOR_TYPES[index]),
        )
        for index, can_id in enumerate(DEFAULT_MOTOR_CAN_IDS)
    ]


def parse_motor_type(value: str | DM_Motor_Type) -> DM_Motor_Type:
    if isinstance(value, DM_Motor_Type):
        return value
    return DM_Motor_Type[str(value)]


def get_motor_limits(motor_type: str | DM_Motor_Type) -> MotorLimits:
    return MOTOR_LIMITS[parse_motor_type(motor_type)]


def float_to_uint(value: float, xmin: float, xmax: float, bits: int) -> int:
    if xmax <= xmin:
        raise ValueError("xmax must be larger than xmin")
    clamped = min(max(float(value), xmin), xmax)
    scale = (1 << bits) - 1
    return int((clamped - xmin) / (xmax - xmin) * scale)


def uint_to_float(value: int, xmin: float, xmax: float, bits: int) -> float:
    scale = (1 << bits) - 1
    return ((float(value) / scale) * (xmax - xmin)) + xmin


def pack_canfd_frame(can_id: int, payload: bytes, flags: int = CANFD_BRS) -> bytes:
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


def build_param_write_frame(can_id: int, rid: int, data: bytes) -> tuple[int, bytes]:
    if len(data) != 4:
        raise ValueError("Motor parameter writes require exactly 4 data bytes")
    return 0x7FF, bytes([can_id & 0xFF, (can_id >> 8) & 0xFF, PARAM_WRITE_CMD, rid, *data])


def build_param_write_uint32_frame(can_id: int, rid: int, value: int) -> tuple[int, bytes]:
    if int(value) < 0:
        raise ValueError("Motor uint32 parameter writes must be >= 0")
    return build_param_write_frame(can_id, rid, int(value).to_bytes(4, byteorder="little", signed=False))


def build_param_store_frame(can_id: int) -> tuple[int, bytes]:
    _validate_standard_can_id(can_id, "can_id")
    return 0x7FF, bytes(
        [
            int(can_id) & 0xFF,
            (int(can_id) >> 8) & 0xFF,
            PARAM_STORE_CMD,
            PARAM_STORE_TO_FLASH,
        ]
    )


def build_zero_position_frame(can_id: int, mode_offset: int = MIT_MODE) -> tuple[int, bytes]:
    return build_control_cmd_frame(int(can_id) + int(mode_offset), ZERO_POSITION_CMD)


def build_mit_frame(
    can_id: int,
    motor_type: DM_Motor_Type | str,
    kp: float,
    kd: float,
    position: float,
    velocity: float,
    torque: float,
) -> tuple[int, bytes]:
    limits = get_motor_limits(motor_type)
    kp_uint = float_to_uint(kp, 0.0, 500.0, 12)
    kd_uint = float_to_uint(kd, 0.0, 5.0, 12)
    q_uint = float_to_uint(position, -limits.pmax, limits.pmax, 16)
    dq_uint = float_to_uint(velocity, -limits.vmax, limits.vmax, 12)
    tau_uint = float_to_uint(torque, -limits.tmax, limits.tmax, 12)
    payload = bytes(
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
    return int(can_id) + MIT_MODE, payload


def decode_feedback_frame(can_id: int, payload: bytes, motor_specs: list[MotorSpec]) -> MotorFeedback | None:
    if len(payload) < 8:
        return None
    spec_by_feedback_id = {spec.mst_id: spec for spec in motor_specs}
    spec = spec_by_feedback_id.get(int(can_id))
    if spec is None:
        return None
    limits = get_motor_limits(spec.motor_type)
    state = (payload[0] >> 4) & 0x0F
    controller_id = payload[0] & 0x0F
    position_uint = (payload[1] << 8) | payload[2]
    velocity_uint = (payload[3] << 4) | (payload[4] >> 4)
    torque_uint = ((payload[4] & 0x0F) << 8) | payload[5]
    return MotorFeedback(
        motor_id=spec.motor_id,
        can_id=int(can_id),
        state=state,
        controller_id=controller_id,
        position=uint_to_float(position_uint, -limits.pmax, limits.pmax, 16),
        velocity=uint_to_float(velocity_uint, -limits.vmax, limits.vmax, 12),
        torque=uint_to_float(torque_uint, -limits.tmax, limits.tmax, 12),
        mos_temperature=float(payload[6]),
        rotor_temperature=float(payload[7]),
    )


def validate_mit_command(command: MitCommand, limits: MotorLimits) -> None:
    fields = {
        "position": command.position,
        "velocity": command.velocity,
        "kp": command.kp,
        "kd": command.kd,
        "torque_ff": command.torque_ff,
    }
    for name, value in fields.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if command.kp < 0.0:
        raise ValueError("kp must be >= 0")
    if command.kd < 0.0:
        raise ValueError("kd must be >= 0")
    if command.kp > 500.0:
        raise ValueError("kp must be <= 500")
    if command.kd > 5.0:
        raise ValueError("kd must be <= 5")
    if abs(command.position) > limits.pmax:
        raise ValueError(f"position exceeds +/-{limits.pmax:g}")
    if abs(command.velocity) > limits.vmax:
        raise ValueError(f"velocity exceeds +/-{limits.vmax:g}")
    if abs(command.torque_ff) > limits.tmax:
        raise ValueError(f"torque_ff exceeds +/-{limits.tmax:g}")


class SocketCanTransport:
    def __init__(self, interface: str = DEFAULT_INTERFACE) -> None:
        self.socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.socket.setsockopt(SOL_CAN_RAW, CAN_RAW_FD_FRAMES, 1)
        self.socket.bind((str(interface),))

    def send(self, can_id: int, payload: bytes) -> None:
        self.socket.send(pack_canfd_frame(int(can_id), bytes(payload)))

    def recv(self, timeout: float = 0.0) -> tuple[int, bytes] | None:
        ready, _, _ = select.select([self.socket], [], [], float(timeout))
        if not ready:
            return None
        return unpack_can_packet(self.socket.recv(CANFD_MTU))

    def close(self) -> None:
        self.socket.close()


def send_frame(transport: CanTransport, can_id: int, payload: bytes) -> None:
    sleep_seconds = BACKPRESSURE_SLEEP_SECONDS
    while True:
        try:
            transport.send(int(can_id), bytes(payload))
            return
        except OSError as exc:
            if exc.errno != errno.ENOBUFS:
                raise
            time.sleep(sleep_seconds)
            sleep_seconds = min(sleep_seconds * 2.0, MAX_BACKPRESSURE_SLEEP_SECONDS)


class DamiaoMitController:
    def __init__(self, transport: CanTransport, motor_specs: list[MotorSpec]) -> None:
        self.transport = transport
        self.motor_specs = {spec.motor_id: spec for spec in motor_specs}
        self.enabled_motor_ids: set[int] = set()

    def set_mit_mode(self, motor_id: int) -> None:
        spec = self._spec(motor_id)
        self.set_mit_mode_raw(spec.can_id)

    def clear_error(self, motor_id: int) -> None:
        self._send_control(motor_id, CLEAR_ERROR_CMD)

    def enable_motor(self, motor_id: int) -> None:
        self.set_mit_mode(motor_id)
        self._send_control(motor_id, ENABLE_CMD)
        self.enabled_motor_ids.add(int(motor_id))

    def disable_motor(self, motor_id: int) -> None:
        self._send_control(motor_id, DISABLE_CMD)
        self.enabled_motor_ids.discard(int(motor_id))

    def save_zero_position(self, motor_id: int) -> None:
        spec = self._spec(motor_id)
        self.save_zero_position_raw(spec.can_id, MIT_MODE)

    def save_zero_position_raw(self, can_id: int, mode_offset: int = MIT_MODE) -> None:
        frame_can_id, payload = build_zero_position_frame(int(can_id), int(mode_offset))
        send_frame(self.transport, frame_can_id, payload)
        time.sleep(CONTROL_INTERVAL_SECONDS)

    def set_mit_mode_raw(self, can_id: int) -> None:
        frame_can_id, payload = build_param_write_uint32_frame(int(can_id), PARAM_CTRL_MODE_RID, MIT_MODE_CODE)
        send_frame(self.transport, frame_can_id, payload)
        time.sleep(PARAM_WRITE_SETTLE_SECONDS)

    def store_parameters_raw(self, can_id: int, mode_offset: int = MIT_MODE) -> None:
        self._send_control_raw(int(can_id), int(mode_offset), DISABLE_CMD)
        frame_can_id, payload = build_param_store_frame(int(can_id))
        send_frame(self.transport, frame_can_id, payload)
        time.sleep(PARAM_STORE_SETTLE_SECONDS)

    def set_mit_mode_persistent_raw(self, can_id: int, mode_offset: int = MIT_MODE) -> None:
        self._send_control_raw(int(can_id), int(mode_offset), DISABLE_CMD)
        self.set_mit_mode_raw(int(can_id))
        self.store_parameters_raw(int(can_id), MIT_MODE)

    def set_motor_ids_raw(self, current_can_id: int, new_can_id: int, new_mst_id: int) -> None:
        for label, value in (("current_can_id", current_can_id), ("new_can_id", new_can_id), ("new_mst_id", new_mst_id)):
            _validate_standard_can_id(value, label)
        # Write feedback ID first because writing ESC_ID changes the address used for further parameter writes.
        for rid, value in ((PARAM_MST_ID_RID, new_mst_id), (PARAM_ESC_ID_RID, new_can_id)):
            frame_can_id, payload = build_param_write_uint32_frame(int(current_can_id), rid, int(value))
            send_frame(self.transport, frame_can_id, payload)
            time.sleep(PARAM_WRITE_SETTLE_SECONDS)

    def set_motor_ids_persistent_raw(
        self,
        current_can_id: int,
        current_mode_offset: int,
        new_can_id: int,
        new_mst_id: int,
    ) -> None:
        self._send_control_raw(int(current_can_id), int(current_mode_offset), DISABLE_CMD)
        self.set_motor_ids_raw(int(current_can_id), int(new_can_id), int(new_mst_id))
        self.store_parameters_raw(int(new_can_id), MIT_MODE)

    def configure_single_motor_raw(
        self,
        current_can_id: int,
        current_mode_offset: int,
        new_can_id: int,
        new_mst_id: int,
    ) -> None:
        _validate_standard_can_id(current_can_id, "current_can_id")
        _validate_standard_can_id(new_can_id, "new_can_id")
        _validate_standard_can_id(new_mst_id, "new_mst_id")
        self._send_control_raw(int(current_can_id), int(current_mode_offset), DISABLE_CMD)
        self.save_zero_position_raw(int(current_can_id), int(current_mode_offset))
        self.set_mit_mode_raw(int(current_can_id))
        self.set_motor_ids_raw(int(current_can_id), int(new_can_id), int(new_mst_id))
        self.store_parameters_raw(int(new_can_id), MIT_MODE)

    def prepare_motor(self, motor_id: int) -> None:
        self.clear_error(motor_id)
        self.enable_motor(motor_id)

    def send_mit(self, motor_id: int, command: MitCommand) -> None:
        spec = self._spec(motor_id)
        validate_mit_command(command, get_motor_limits(spec.motor_type))
        can_id, payload = build_mit_frame(
            spec.can_id,
            spec.motor_type,
            command.kp,
            command.kd,
            command.position,
            command.velocity,
            command.torque_ff,
        )
        send_frame(self.transport, can_id, payload)

    def prepare_and_send_mit(self, motor_id: int, command: MitCommand) -> None:
        self.prepare_motor(motor_id)
        self.send_mit(motor_id, command)

    def close(self) -> None:
        self.transport.close()

    def _send_control(self, motor_id: int, cmd: int) -> None:
        spec = self._spec(motor_id)
        self._send_control_raw(spec.can_id, MIT_MODE, cmd)

    def _send_control_raw(self, can_id: int, mode_offset: int, cmd: int) -> None:
        frame_can_id, payload = build_control_cmd_frame(int(can_id) + int(mode_offset), cmd)
        for _ in range(CONTROL_REPEAT):
            send_frame(self.transport, frame_can_id, payload)
            time.sleep(CONTROL_INTERVAL_SECONDS)

    def _spec(self, motor_id: int) -> MotorSpec:
        try:
            return self.motor_specs[int(motor_id)]
        except KeyError as exc:
            valid_ids = ", ".join(str(key) for key in sorted(self.motor_specs))
            raise ValueError(f"Unknown motor id {motor_id}; valid ids: {valid_ids}") from exc


def _validate_standard_can_id(value: int, label: str = "can_id") -> None:
    if int(value) < 0x001 or int(value) > 0x7FE:
        raise ValueError(f"{label} must be within 0x001..0x7FE")


def configure_can_interface(
    interface: str,
    nominal_bitrate: int = DEFAULT_NOMINAL_BITRATE,
    data_bitrate: int = DEFAULT_DATA_BITRATE,
) -> None:
    commands = [
        ["ip", "link", "set", interface, "down"],
        [
            "ip",
            "link",
            "set",
            interface,
            "type",
            "can",
            "bitrate",
            str(int(nominal_bitrate)),
            "dbitrate",
            str(int(data_bitrate)),
            "fd",
            "on",
        ],
        ["ip", "link", "set", interface, "up"],
    ]
    for command in commands:
        subprocess.run(command, check=True)


def format_can_setup_commands(
    interface: str,
    nominal_bitrate: int = DEFAULT_NOMINAL_BITRATE,
    data_bitrate: int = DEFAULT_DATA_BITRATE,
) -> str:
    return "\n".join(
        [
            f"sudo ip link set {interface} down",
            f"sudo ip link set {interface} type can bitrate {int(nominal_bitrate)} dbitrate {int(data_bitrate)} fd on",
            f"sudo ip link set {interface} up",
        ]
    )


def can_setup_hint(
    interface: str,
    nominal_bitrate: int = DEFAULT_NOMINAL_BITRATE,
    data_bitrate: int = DEFAULT_DATA_BITRATE,
) -> str:
    return "可手动执行以下三行命令:\n" + format_can_setup_commands(interface, nominal_bitrate, data_bitrate)


def get_can_interface_state(interface: str) -> str | None:
    state_path = f"/sys/class/net/{interface}/operstate"
    if not os.path.exists(state_path):
        return None
    with open(state_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def list_available_can_interfaces() -> tuple[str, ...]:
    return tuple(sorted(os.path.basename(path) for path in glob.glob("/sys/class/net/can*")))


def format_available_can_interfaces() -> str:
    interfaces = list_available_can_interfaces()
    return "检测到: " + (", ".join(interfaces) if interfaces else "无")


def ensure_interface_ready(
    interface: str,
    nominal_bitrate: int = DEFAULT_NOMINAL_BITRATE,
    data_bitrate: int = DEFAULT_DATA_BITRATE,
) -> None:
    state = get_can_interface_state(interface)
    if state is None:
        raise RuntimeError(
            f"CAN interface {interface} does not exist\n"
            f"{format_available_can_interfaces()}\n"
            f"{can_setup_hint(interface, nominal_bitrate, data_bitrate)}"
        )
    if state != "up":
        raise RuntimeError(
            f"{interface} is not UP.\n{can_setup_hint(interface, nominal_bitrate, data_bitrate)}"
        )


__all__ = [
    "CANFD_BRS",
    "ALLOWED_INTERFACES",
    "CLEAR_ERROR_CMD",
    "CONTROL_REPEAT",
    "DEFAULT_DATA_BITRATE",
    "DEFAULT_INTERFACE",
    "DEFAULT_MOTOR_CAN_IDS",
    "DEFAULT_MOTOR_MST_IDS",
    "DEFAULT_MOTOR_TYPES",
    "DEFAULT_NOMINAL_BITRATE",
    "DISABLE_CMD",
    "DM_Motor_Type",
    "DamiaoMitController",
    "ENABLE_CMD",
    "MIT_MODE",
    "MitCommand",
    "MotorFeedback",
    "MotorLimits",
    "MotorSpec",
    "PARAM_CTRL_MODE_RID",
    "PARAM_ESC_ID_RID",
    "PARAM_MST_ID_RID",
    "PARAM_STORE_CMD",
    "PARAM_STORE_SETTLE_SECONDS",
    "PARAM_STORE_TO_FLASH",
    "PARAM_WRITE_CMD",
    "SocketCanTransport",
    "ZERO_POSITION_CMD",
    "build_control_cmd_frame",
    "build_mit_frame",
    "build_param_store_frame",
    "build_param_write_frame",
    "build_param_write_uint32_frame",
    "build_zero_position_frame",
    "configure_can_interface",
    "decode_feedback_frame",
    "default_motor_specs",
    "ensure_interface_ready",
    "format_available_can_interfaces",
    "format_can_setup_commands",
    "float_to_uint",
    "get_can_interface_state",
    "get_motor_limits",
    "list_available_can_interfaces",
    "pack_canfd_frame",
    "parse_motor_type",
    "send_frame",
    "uint_to_float",
    "unpack_can_packet",
    "validate_mit_command",
]
