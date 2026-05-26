from __future__ import annotations

from dataclasses import dataclass

from mit_sender.damiao import MIT_MODE, MitCommand


MIT_COMMAND_DEFAULTS = {
    "position": "0.0",
    "velocity": "0.0",
    "kp": "0.0",
    "kd": "0.0",
    "torque_ff": "0.0",
}

MIT_FIELD_LABELS = {
    "position": "P",
    "velocity": "V",
    "kp": "Kp",
    "kd": "Kd",
    "torque_ff": "Tau",
}

MIT_FIELD_TOOLTIPS = {
    "position": "position: 目标位置",
    "velocity": "velocity: 目标速度",
    "kp": "kp: 位置增益，范围 0..500",
    "kd": "kd: 速度增益，范围 0..5",
    "torque_ff": "torque_ff: 前馈力矩",
}

POSITION_VELOCITY_MODE = 0x100
VELOCITY_MODE = 0x200
DEBUG_MODE_OPTIONS = (
    ("MIT", MIT_MODE),
    ("位置速度", POSITION_VELOCITY_MODE),
    ("速度", VELOCITY_MODE),
)
DEBUG_MODE_OFFSETS = tuple(offset for _label, offset in DEBUG_MODE_OPTIONS)
DEFAULT_DEBUG_FEEDBACK_OFFSET = 0x10


@dataclass(frozen=True)
class SelectedMotorCommand:
    motor_id: int
    position: float
    velocity: float
    kp: float
    kd: float
    torque_ff: float

    def as_mit_command(self) -> MitCommand:
        return MitCommand(
            position=self.position,
            velocity=self.velocity,
            kp=self.kp,
            kd=self.kd,
            torque_ff=self.torque_ff,
        )


@dataclass(frozen=True)
class TransportSettings:
    interface: str
    nominal_bitrate: int
    data_bitrate: int
    configure_interface: bool


@dataclass(frozen=True)
class SingleMotorDebugCommand:
    current_can_id: int
    current_mode_offset: int
    new_can_id: int
    new_mst_id: int


def default_single_motor_debug_command() -> SingleMotorDebugCommand:
    default_can_id = 0x01
    return SingleMotorDebugCommand(
        current_can_id=default_can_id,
        current_mode_offset=MIT_MODE,
        new_can_id=default_can_id,
        new_mst_id=default_can_id + DEFAULT_DEBUG_FEEDBACK_OFFSET,
    )


def build_uniform_commands(
    motor_ids: list[int],
    command: SelectedMotorCommand,
) -> list[SelectedMotorCommand]:
    return [
        SelectedMotorCommand(
            motor_id=motor_id,
            position=command.position,
            velocity=command.velocity,
            kp=command.kp,
            kd=command.kd,
            torque_ff=command.torque_ff,
        )
        for motor_id in motor_ids
    ]
