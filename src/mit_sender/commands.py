from __future__ import annotations

from dataclasses import dataclass

from mit_sender.damiao import MitCommand


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
