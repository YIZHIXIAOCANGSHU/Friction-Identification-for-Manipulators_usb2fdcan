from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from friction_identification_core.runtime_config import Config
from send import damiao as send_damiao


SemanticMode = Literal["mit_torque", "mit_velocity", "velocity_mode"]

RECV_FRAME_HEAD = send_damiao.SERIALIZED_FEEDBACK_HEAD
RECV_FRAME_FORMAT = send_damiao.SERIALIZED_FEEDBACK_FORMAT
RECV_FRAME_STRUCT = send_damiao.SERIALIZED_FEEDBACK_STRUCT
RECV_FRAME_SIZE = RECV_FRAME_STRUCT.size
VALID_FEEDBACK_STATE_CODES = send_damiao.VALID_FEEDBACK_STATE_CODES


@dataclass(frozen=True)
class FeedbackFrame:
    motor_id: int
    state: int
    position: float
    velocity: float
    torque: float
    mos_temperature: float


class FeedbackFrameParser:
    def __init__(self, *, max_motor_id: int = 7) -> None:
        self._buffer = bytearray()
        self._max_motor_id = max(int(max_motor_id), 1)

    def feed(self, chunk: bytes) -> None:
        if chunk:
            self._buffer.extend(chunk)

    def reset(self) -> None:
        self._buffer.clear()

    def _is_valid_candidate(self, parsed: tuple[object, ...]) -> bool:
        if int(parsed[0]) != RECV_FRAME_HEAD:
            return False
        motor_id = int(parsed[1])
        if not 1 <= motor_id <= self._max_motor_id:
            return False
        values = np.asarray(parsed[3:], dtype=np.float32)
        if not np.all(np.isfinite(values)):
            return False
        return True

    def pop_frame(self) -> FeedbackFrame | None:
        while True:
            if len(self._buffer) < RECV_FRAME_SIZE:
                return None

            head_index = self._buffer.find(RECV_FRAME_HEAD)
            if head_index < 0:
                self._buffer.clear()
                return None
            if head_index > 0:
                del self._buffer[:head_index]
                if len(self._buffer) < RECV_FRAME_SIZE:
                    return None

            try:
                parsed = RECV_FRAME_STRUCT.unpack_from(self._buffer)
            except struct.error:
                return None

            if not self._is_valid_candidate(parsed):
                del self._buffer[0]
                continue

            del self._buffer[:RECV_FRAME_SIZE]
            return FeedbackFrame(
                motor_id=int(parsed[1]),
                state=int(parsed[2]),
                position=float(parsed[3]),
                velocity=float(parsed[4]),
                torque=float(parsed[5]),
                mos_temperature=float(parsed[6]),
            )


class CommandTransport(Protocol):
    def read(self, size: int) -> bytes:
        ...

    def pop_feedback_frame(self) -> FeedbackFrame | None:
        ...

    def send_mit_torque(self, motor_id: int, torque: float) -> bytes:
        ...

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
        ...

    def send_velocity_mode(self, motor_id: int, velocity: float) -> bytes:
        ...

    def send_zero_command(self, motor_id: int, semantic_mode: SemanticMode) -> bytes:
        ...

    def limit_torque_command(self, motor_id: int, torque: float) -> float:
        ...

    def enable_motor(self, motor_id: int) -> bytes:
        ...

    def disable_motor(self, motor_id: int) -> bytes:
        ...

    def clear_error(self, motor_id: int) -> bytes:
        ...

    def reset_input_buffer(self) -> None:
        ...

    def close(self) -> None:
        ...


DamiaoSocketCanTransport = send_damiao.DamiaoSocketCanTransport


def open_transport(config: Config) -> CommandTransport:
    return DamiaoSocketCanTransport(config)


__all__ = [
    "CommandTransport",
    "DamiaoSocketCanTransport",
    "FeedbackFrame",
    "FeedbackFrameParser",
    "RECV_FRAME_HEAD",
    "RECV_FRAME_SIZE",
    "RECV_FRAME_STRUCT",
    "SemanticMode",
    "VALID_FEEDBACK_STATE_CODES",
    "open_transport",
]
