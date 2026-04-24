import argparse
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


CAN_MTU = 16
CANFD_MTU = 72
CAN_RAW_FD_FRAMES = getattr(socket, "CAN_RAW_FD_FRAMES", 5)
SOL_CAN_RAW = getattr(socket, "SOL_CAN_RAW", socket.SOL_CAN_BASE + socket.CAN_RAW)
CANFD_BRS = getattr(socket, "CANFD_BRS", 0x01)


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


RUNNING = threading.Event()
RUNNING.set()

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


def signal_handler(signum, frame):
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
    ):
        now = time.monotonic()
        self.delta_time = 0.0 if self.last_time == 0.0 else now - self.last_time
        self.last_time = now
        self.position = position
        self.velocity = velocity
        self.torque = torque
        self.controller_id = controller_id
        self.state_code = state_code
        self.mos_temp = mos_temp
        self.rotor_temp = rotor_temp


def int_auto(value: str) -> int:
    return int(value, 0)


def float_to_uint(value: float, xmin: float, xmax: float, bits: int) -> int:
    if xmax <= xmin:
        raise ValueError("xmax must be larger than xmin")

    clamped = min(max(value, xmin), xmax)
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


def pack_can_frame(can_id: int, payload: bytes) -> bytes:
    if len(payload) > 8:
        raise ValueError("Classic CAN payload must be 8 bytes or fewer")
    return struct.pack("=IB3x8s", can_id, len(payload), payload.ljust(8, b"\x00"))


def pack_canfd_frame(can_id: int, payload: bytes, flags: int = 0) -> bytes:
    if len(payload) > 64:
        raise ValueError("CAN FD payload must be 64 bytes or fewer")
    return struct.pack("=IBB2x64s", can_id, len(payload), flags, payload.ljust(64, b"\x00"))


def unpack_can_packet(packet: bytes) -> tuple[int, bytes]:
    if len(packet) == CAN_MTU:
        can_id, can_dlc, data = struct.unpack("=IB3x8s", packet)
        return can_id & socket.CAN_SFF_MASK, data[:can_dlc]

    if len(packet) == CANFD_MTU:
        can_id, length, _, data = struct.unpack("=IBB2x64s", packet)
        return can_id & socket.CAN_SFF_MASK, data[:length]

    raise ValueError(f"Unsupported CAN packet size: {len(packet)}")


def build_control_cmd_frame(can_id: int, cmd: int) -> tuple[int, bytes]:
    return can_id, bytes([0xFF] * 7 + [cmd])


def build_param_read_frame(can_id: int, rid: int) -> tuple[int, bytes]:
    return 0x7FF, bytes([can_id & 0xFF, (can_id >> 8) & 0xFF, 0x33, rid, 0x00, 0x00, 0x00, 0x00])


def build_param_write_frame(can_id: int, rid: int, data: bytes) -> tuple[int, bytes]:
    if len(data) != 4:
        raise ValueError("Motor parameter writes require exactly 4 data bytes")
    return 0x7FF, bytes([can_id & 0xFF, (can_id >> 8) & 0xFF, 0x55, rid, *data])


def build_vel_frame(can_id: int, velocity: float) -> tuple[int, bytes]:
    return can_id + Control_Mode.VEL_MODE, struct.pack("<f", velocity)


def build_pos_vel_frame(can_id: int, position: float, velocity: float) -> tuple[int, bytes]:
    return can_id + Control_Mode.POS_VEL_MODE, struct.pack("<ff", position, velocity)


def build_mit_frame(can_id: int, motor_type: DM_Motor_Type, kp: float, kd: float, position: float, velocity: float, torque: float) -> tuple[int, bytes]:
    position_limit, velocity_limit, torque_limit = LIMIT_PARAM[motor_type.value]

    kp_uint = float_to_uint(kp, 0, 500, 12)
    kd_uint = float_to_uint(kd, 0, 5, 12)
    q_uint = float_to_uint(position, -position_limit, position_limit, 16)
    dq_uint = float_to_uint(velocity, -velocity_limit, velocity_limit, 12)
    tau_uint = float_to_uint(torque, -torque_limit, torque_limit, 12)

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
    return can_id + Control_Mode.MIT_MODE, data


def decode_feedback(data: bytes, motor_type: DM_Motor_Type) -> MotorFeedback:
    if len(data) < 8:
        raise ValueError("Motor feedback requires 8 bytes")

    position_limit, velocity_limit, torque_limit = LIMIT_PARAM[motor_type.value]
    controller_id = data[0] & 0x0F
    state_code = (data[0] >> 4) & 0x0F
    q_uint = (data[1] << 8) | data[2]
    dq_uint = (data[3] << 4) | (data[4] >> 4)
    tau_uint = ((data[4] & 0x0F) << 8) | data[5]

    return MotorFeedback(
        position=uint_to_float(q_uint, -position_limit, position_limit, 16),
        velocity=uint_to_float(dq_uint, -velocity_limit, velocity_limit, 12),
        torque=uint_to_float(tau_uint, -torque_limit, torque_limit, 12),
        controller_id=controller_id,
        state_code=state_code,
        mos_temp=float(data[6]),
        rotor_temp=float(data[7]),
    )


class SocketCanTransport:
    def __init__(self, interface: str, force_fd: bool = True, fd_flags: int = CANFD_BRS):
        self.interface = interface
        self.force_fd = force_fd
        self.fd_flags = fd_flags
        self.socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.socket.setsockopt(SOL_CAN_RAW, CAN_RAW_FD_FRAMES, 1)
        self.socket.settimeout(0.1)
        self.socket.bind((interface,))

    def close(self):
        self.socket.close()

    def send(self, can_id: int, payload: bytes):
        if self.force_fd:
            packet = pack_canfd_frame(can_id, payload, flags=self.fd_flags)
        else:
            packet = pack_can_frame(can_id, payload) if len(payload) <= 8 else pack_canfd_frame(can_id, payload)
        self.socket.send(packet)

    def recv(self, timeout: float = 0.1) -> Optional[tuple[int, bytes]]:
        try:
            ready, _, _ = select.select([self.socket], [], [], timeout)
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

    def start(self):
        self._rx_running.set()
        self._rx_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._rx_thread.start()

    def close(self):
        self.disable()
        self._rx_running.clear()
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
        self.transport.close()

    def _recv_loop(self):
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

    def send(self, can_id: int, payload: bytes):
        self.transport.send(can_id, payload)

    def switch_control_mode(self, mode: Control_Mode):
        can_id, payload = build_param_write_frame(self.motor.can_id, 10, bytes([mode_to_code(mode), 0x00, 0x00, 0x00]))
        self.send(can_id, payload)
        self.motor.mode = mode
        time.sleep(0.01)

    def enable(self):
        self.switch_control_mode(self.motor.mode)
        control_id = self.motor.can_id + self.motor.mode
        can_id, payload = build_control_cmd_frame(control_id, 0xFC)
        for _ in range(5):
            self.send(can_id, payload)
            time.sleep(0.002)

    def disable(self):
        control_id = self.motor.can_id + self.motor.mode
        can_id, payload = build_control_cmd_frame(control_id, 0xFD)
        for _ in range(5):
            try:
                self.send(can_id, payload)
            except OSError:
                break
            time.sleep(0.002)

    def set_zero_position(self):
        can_id, payload = build_control_cmd_frame(self.motor.can_id + self.motor.mode, 0xFE)
        self.send(can_id, payload)
        time.sleep(0.002)

    def control_velocity(self, velocity: float):
        can_id, payload = build_vel_frame(self.motor.can_id, velocity)
        self.send(can_id, payload)

    def control_position_velocity(self, position: float, velocity: float):
        can_id, payload = build_pos_vel_frame(self.motor.can_id, position, velocity)
        self.send(can_id, payload)

    def control_mit(self, kp: float, kd: float, position: float, velocity: float, torque: float):
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


def configure_can_interface(interface: str, nominal_bitrate: int, data_bitrate: int):
    commands = [
        ["ip", "link", "set", interface, "down"],
        ["ip", "link", "set", interface, "type", "can", "bitrate", str(nominal_bitrate), "dbitrate", str(data_bitrate), "fd", "on"],
        ["ip", "link", "set", interface, "up"],
    ]
    for cmd in commands:
        subprocess.run(cmd, check=True)


def ensure_interface_ready(interface: str, nominal_bitrate: int, data_bitrate: int):
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


def build_parser() -> argparse.ArgumentParser:
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


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    except subprocess.CalledProcessError as exc:
        print(f"命令执行失败: {' '.join(exc.cmd)}", file=sys.stderr)
        raise SystemExit(exc.returncode) from None
