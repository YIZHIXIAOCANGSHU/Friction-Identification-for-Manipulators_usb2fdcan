from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from friction_identification_core.core import RoundCapture
from friction_identification_core.io import CommandTransport, FeedbackFrameParser, SemanticMode
from friction_identification_core.runtime_config import Config
from friction_identification_core.results import log_info
from friction_identification_core.visualization import RerunRecorder


@dataclass
class CaptureBuffer:
    target_motor_id: int
    motor_name: str
    start_monotonic: float = field(default_factory=time.monotonic)
    time_log: list[float] = field(default_factory=list)
    motor_id_log: list[int] = field(default_factory=list)
    position_log: list[float] = field(default_factory=list)
    velocity_log: list[float] = field(default_factory=list)
    torque_log: list[float] = field(default_factory=list)
    command_raw_log: list[float] = field(default_factory=list)
    command_log: list[float] = field(default_factory=list)
    position_cmd_log: list[float] = field(default_factory=list)
    velocity_cmd_log: list[float] = field(default_factory=list)
    acceleration_cmd_log: list[float] = field(default_factory=list)
    kp_cmd_log: list[float] = field(default_factory=list)
    kd_cmd_log: list[float] = field(default_factory=list)
    torque_ff_cmd_log: list[float] = field(default_factory=list)
    position_error_log: list[float] = field(default_factory=list)
    velocity_error_log: list[float] = field(default_factory=list)
    tracking_ok_log: list[bool] = field(default_factory=list)
    safety_ok_log: list[bool] = field(default_factory=list)
    state_ok_log: list[bool] = field(default_factory=list)
    saturated_log: list[bool] = field(default_factory=list)
    used_for_fit_log: list[bool] = field(default_factory=list)
    tau_mit_est_log: list[float] = field(default_factory=list)
    phase_log: list[str] = field(default_factory=list)
    state_log: list[int] = field(default_factory=list)
    mos_temperature_log: list[float] = field(default_factory=list)
    id_match_log: list[bool] = field(default_factory=list)
    filtered_velocity_log: list[float] = field(default_factory=list)
    estimated_acceleration_log: list[float] = field(default_factory=list)
    friction_term_log: list[float] = field(default_factory=list)
    inertia_term_log: list[float] = field(default_factory=list)
    guard_scale_log: list[float] = field(default_factory=list)
    stiction_evidence_log: list[bool] = field(default_factory=list)

    def append(
        self,
        *,
        frame,
        command_raw: float,
        command: float,
        position_cmd: float,
        velocity_cmd: float,
        acceleration_cmd: float,
        phase_name: str,
        kp_cmd: float = 0.0,
        kd_cmd: float = 0.0,
        torque_ff_cmd: float = 0.0,
        position_error: float = float("nan"),
        velocity_error: float = float("nan"),
        tracking_ok: bool = True,
        safety_ok: bool = True,
        state_ok: bool = True,
        saturated: bool = False,
        used_for_fit: bool = False,
        tau_mit_est: float = float("nan"),
        filtered_velocity: float = float("nan"),
        estimated_acceleration: float = float("nan"),
        friction_term: float = float("nan"),
        inertia_term: float = float("nan"),
        guard_scale: float = float("nan"),
        stiction_evidence: bool = False,
    ) -> None:
        self.time_log.append(time.monotonic() - self.start_monotonic)
        self.motor_id_log.append(int(frame.motor_id))
        self.position_log.append(float(frame.position))
        self.velocity_log.append(float(frame.velocity))
        self.torque_log.append(float(frame.torque))
        self.command_raw_log.append(float(command_raw))
        self.command_log.append(float(command))
        self.position_cmd_log.append(float(position_cmd))
        self.velocity_cmd_log.append(float(velocity_cmd))
        self.acceleration_cmd_log.append(float(acceleration_cmd))
        self.kp_cmd_log.append(float(kp_cmd))
        self.kd_cmd_log.append(float(kd_cmd))
        self.torque_ff_cmd_log.append(float(torque_ff_cmd))
        self.position_error_log.append(float(position_error))
        self.velocity_error_log.append(float(velocity_error))
        self.tracking_ok_log.append(bool(tracking_ok))
        self.safety_ok_log.append(bool(safety_ok))
        self.state_ok_log.append(bool(state_ok))
        self.saturated_log.append(bool(saturated))
        self.used_for_fit_log.append(bool(used_for_fit))
        self.tau_mit_est_log.append(float(tau_mit_est))
        self.phase_log.append(str(phase_name))
        self.state_log.append(int(frame.state))
        self.mos_temperature_log.append(float(frame.mos_temperature))
        self.id_match_log.append(True)
        self.filtered_velocity_log.append(float(filtered_velocity))
        self.estimated_acceleration_log.append(float(estimated_acceleration))
        self.friction_term_log.append(float(friction_term))
        self.inertia_term_log.append(float(inertia_term))
        self.guard_scale_log.append(float(guard_scale))
        self.stiction_evidence_log.append(bool(stiction_evidence))

    def build(self, *, group_index: int, round_index: int, metadata: dict[str, object]) -> RoundCapture:
        return RoundCapture(
            group_index=int(group_index),
            round_index=int(round_index),
            target_motor_id=int(self.target_motor_id),
            motor_name=str(self.motor_name),
            time=np.asarray(self.time_log, dtype=np.float64),
            motor_id=np.asarray(self.motor_id_log, dtype=np.int64),
            position=np.asarray(self.position_log, dtype=np.float64),
            velocity=np.asarray(self.velocity_log, dtype=np.float64),
            torque_feedback=np.asarray(self.torque_log, dtype=np.float64),
            command_raw=np.asarray(self.command_raw_log, dtype=np.float64),
            command=np.asarray(self.command_log, dtype=np.float64),
            position_cmd=np.asarray(self.position_cmd_log, dtype=np.float64),
            velocity_cmd=np.asarray(self.velocity_cmd_log, dtype=np.float64),
            acceleration_cmd=np.asarray(self.acceleration_cmd_log, dtype=np.float64),
            phase_name=np.asarray(self.phase_log),
            state=np.asarray(self.state_log, dtype=np.uint8),
            mos_temperature=np.asarray(self.mos_temperature_log, dtype=np.float64),
            id_match_ok=np.asarray(self.id_match_log, dtype=bool),
            filtered_velocity=np.asarray(self.filtered_velocity_log, dtype=np.float64),
            estimated_acceleration=np.asarray(self.estimated_acceleration_log, dtype=np.float64),
            friction_term=np.asarray(self.friction_term_log, dtype=np.float64),
            inertia_term=np.asarray(self.inertia_term_log, dtype=np.float64),
            guard_scale=np.asarray(self.guard_scale_log, dtype=np.float64),
            stiction_evidence=np.asarray(self.stiction_evidence_log, dtype=bool),
            kp_cmd=np.asarray(self.kp_cmd_log, dtype=np.float64),
            kd_cmd=np.asarray(self.kd_cmd_log, dtype=np.float64),
            torque_ff_cmd=np.asarray(self.torque_ff_cmd_log, dtype=np.float64),
            position_error=np.asarray(self.position_error_log, dtype=np.float64),
            velocity_error=np.asarray(self.velocity_error_log, dtype=np.float64),
            tracking_ok=np.asarray(self.tracking_ok_log, dtype=bool),
            safety_ok=np.asarray(self.safety_ok_log, dtype=bool),
            state_ok=np.asarray(self.state_ok_log, dtype=bool),
            saturated=np.asarray(self.saturated_log, dtype=bool),
            used_for_fit=np.asarray(self.used_for_fit_log, dtype=bool),
            tau_mit_est=np.asarray(self.tau_mit_est_log, dtype=np.float64),
            metadata=dict(metadata),
        )


def sent_command_vector(config: Config, *, target_index: int, target_command: float) -> np.ndarray:
    sent_commands = np.zeros(config.motor_count, dtype=np.float64)
    sent_commands[target_index] = float(target_command)
    return sent_commands


def expected_position_vector(config: Config, *, target_index: int, target_position: float) -> np.ndarray:
    expected = np.zeros(config.motor_count, dtype=np.float64)
    expected[target_index] = float(target_position)
    return expected


def expected_velocity_vector(config: Config, *, target_index: int, target_velocity: float) -> np.ndarray:
    expected = np.zeros(config.motor_count, dtype=np.float64)
    expected[target_index] = float(target_velocity)
    return expected


def poll_feedback_frames(
    *,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    read_chunk_size: int,
) -> tuple[tuple, bool]:
    chunk = transport.read(read_chunk_size)
    pop_feedback_frame = getattr(transport, "pop_feedback_frame", None)
    if callable(pop_feedback_frame):
        frames = []
        while True:
            frame = pop_feedback_frame()
            if frame is None:
                break
            frames.append(frame)
        return tuple(frames), bool(chunk)

    if chunk:
        parser.feed(chunk)
    frames = []
    while True:
        frame = parser.pop_frame()
        if frame is None:
            break
        frames.append(frame)
    return tuple(frames), bool(chunk)


def safety_margin_text(config: Config, observed_velocity: float, command_value: float) -> str:
    return (
        f"velocity_margin={float(config.safety.hard_speed_abort_abs) - abs(float(observed_velocity)):+.6f}, "
        f"command={float(command_value):+.6f}"
    )


def log_stage_transition(stage: str, *, target_motor_id: int, detail: str = "") -> None:
    message = f"Stage {str(stage)}: motor_id={int(target_motor_id)}"
    if detail:
        message += f", {str(detail)}"
    log_info(message)


def send_command(
    *,
    config: Config,
    transport: CommandTransport,
    rerun_recorder: RerunRecorder,
    target_motor_id: int,
    target_index: int,
    semantic_mode: SemanticMode,
    command_value: float,
    kd_speed: float = 0.0,
    position_cmd: float = 0.0,
    velocity_cmd: float = 0.0,
    kp_cmd: float = 0.0,
    torque_ff_cmd: float = 0.0,
) -> bytes:
    if semantic_mode == "mit_torque":
        packet = transport.send_mit_torque(int(target_motor_id), float(command_value))
    elif semantic_mode == "mit_velocity":
        packet = transport.send_mit_velocity(
            int(target_motor_id),
            float(command_value),
            float(kd_speed),
            kp=0.0,
            torque_ff=0.0,
            position=0.0,
        )
    elif semantic_mode == "mit_state":
        packet = transport.send_mit_state(
            int(target_motor_id),
            float(position_cmd),
            float(velocity_cmd),
            float(kp_cmd),
            float(kd_speed),
            torque_ff=float(torque_ff_cmd),
        )
    elif semantic_mode == "velocity_mode":
        packet = transport.send_velocity_mode(int(target_motor_id), float(command_value))
    else:  # pragma: no cover
        raise ValueError(f"Unsupported semantic_mode: {semantic_mode}")

    rerun_recorder.log_live_command_packet(
        sent_commands=sent_command_vector(config, target_index=target_index, target_command=float(command_value)),
        expected_positions=expected_position_vector(config, target_index=target_index, target_position=float(position_cmd)),
        expected_velocities=expected_velocity_vector(config, target_index=target_index, target_velocity=float(velocity_cmd)),
        raw_packet=packet,
    )
    return packet


def record_target_frame(
    *,
    config: Config,
    rerun_recorder: RerunRecorder,
    capture_buffer: CaptureBuffer | None,
    group_index: int,
    round_index: int,
    target_motor_id: int,
    frame,
    command_raw: float,
    command: float,
    position_cmd: float,
    velocity_cmd: float,
    acceleration_cmd: float,
    phase_name: str,
    stage: str,
    kp_cmd: float = 0.0,
    kd_cmd: float = 0.0,
    torque_ff_cmd: float = 0.0,
    tracking_ok: bool | None = None,
    safety_ok: bool | None = None,
    state_ok: bool | None = None,
    saturated: bool = False,
    used_for_fit: bool = False,
    tau_mit_est: float = float("nan"),
    filtered_velocity: float = float("nan"),
    estimated_acceleration: float = float("nan"),
    friction_term: float = float("nan"),
    inertia_term: float = float("nan"),
    guard_scale: float = float("nan"),
    stiction_evidence: bool = False,
) -> None:
    position_error = float(position_cmd) - float(frame.position)
    velocity_error = float(velocity_cmd) - float(frame.velocity)
    if tracking_ok is None:
        tracking_ok = True
    if safety_ok is None:
        safety_ok = abs(float(frame.velocity)) < float(config.safety.hard_speed_abort_abs)
    if state_ok is None:
        state_ok = int(frame.state) in {0, 1}
    if capture_buffer is not None:
        capture_buffer.append(
            frame=frame,
            command_raw=float(command_raw),
            command=float(command),
            position_cmd=float(position_cmd),
            velocity_cmd=float(velocity_cmd),
            acceleration_cmd=float(acceleration_cmd),
            phase_name=str(phase_name),
            kp_cmd=float(kp_cmd),
            kd_cmd=float(kd_cmd),
            torque_ff_cmd=float(torque_ff_cmd),
            position_error=float(position_error),
            velocity_error=float(velocity_error),
            tracking_ok=bool(tracking_ok),
            safety_ok=bool(safety_ok),
            state_ok=bool(state_ok),
            saturated=bool(saturated),
            used_for_fit=bool(used_for_fit),
            tau_mit_est=float(tau_mit_est),
            filtered_velocity=float(filtered_velocity),
            estimated_acceleration=float(estimated_acceleration),
            friction_term=float(friction_term),
            inertia_term=float(inertia_term),
            guard_scale=float(guard_scale),
            stiction_evidence=bool(stiction_evidence),
        )

    rerun_recorder.log_live_motor_sample(
        group_index=int(group_index),
        round_index=int(round_index),
        active_motor_id=int(target_motor_id),
        motor_id=int(frame.motor_id),
        position=float(frame.position),
        velocity=float(frame.velocity),
        feedback_torque=float(frame.torque),
        command_raw=float(command_raw),
        command=float(command),
        reference_position=float(position_cmd),
        reference_velocity=float(velocity_cmd),
        reference_acceleration=float(acceleration_cmd),
        velocity_limit=float(config.safety.hard_speed_abort_abs),
        torque_limit=float(abs(command)) if np.isfinite(float(command)) else float("nan"),
        position_limit=float("nan"),
        phase_name=str(phase_name),
        stage=str(stage),
        safety_margin_text=safety_margin_text(config, float(frame.velocity), float(command)),
        filtered_velocity=float(filtered_velocity),
        estimated_acceleration=float(estimated_acceleration),
        friction_term=float(friction_term),
        inertia_term=float(inertia_term),
        guard_scale=float(guard_scale),
        stiction_evidence=bool(stiction_evidence),
        tracking_ok=bool(tracking_ok),
        safety_ok=bool(safety_ok),
        state_ok=bool(state_ok),
        saturated=bool(saturated),
        used_for_fit=bool(used_for_fit),
    )


__all__ = [
    "CaptureBuffer",
    "expected_position_vector",
    "expected_velocity_vector",
    "log_stage_transition",
    "poll_feedback_frames",
    "record_target_frame",
    "safety_margin_text",
    "send_command",
    "sent_command_vector",
]
