from __future__ import annotations

from collections import deque
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from friction_identification_core.core import (
    AbortEvent,
    BreakawayIdentificationResult,
    FrictionIdentificationResult,
    InertiaIdentificationResult,
    MotorIdentificationResult,
    RoundCapture,
    RunResult,
    ValidationResult,
)
from friction_identification_core.identification import (
    build_validation_result,
    estimate_filtered_velocity_and_acceleration,
    fit_friction_model,
    fit_inertia_model,
)
from friction_identification_core.io import (
    CommandTransport,
    FeedbackFrameParser,
    SemanticMode,
    open_transport,
)
from friction_identification_core.results import (
    ResultStore,
    RoundArtifact,
    latest_parameters_path,
    load_latest_parameters,
    log_info,
)
from friction_identification_core.runtime_config import Config
from friction_identification_core.visualization import RerunRecorder


ABORT_ZERO_COMMAND_REPEAT = 5


class _RuntimeAbortError(RuntimeError):
    def __init__(self, event: AbortEvent) -> None:
        self.event = event
        super().__init__(event.error_message())


@dataclass
class _CaptureBuffer:
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
    phase_log: list[str] = field(default_factory=list)
    state_log: list[int] = field(default_factory=list)
    mos_temperature_log: list[float] = field(default_factory=list)
    id_match_log: list[bool] = field(default_factory=list)

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
        self.phase_log.append(str(phase_name))
        self.state_log.append(int(frame.state))
        self.mos_temperature_log.append(float(frame.mos_temperature))
        self.id_match_log.append(True)

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
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class _CompensationParameters:
    motor_id: int
    motor_name: str
    identified_at: str
    source_run_label: str
    tau_static: float
    tau_bias: float
    tau_c: float
    viscous: float
    inertia: float
    friction_validation_rmse: float
    inertia_validation_rmse: float
    repeat_consistency_score: float
    recommended_for_compensation: bool


def _sent_command_vector(config: Config, *, target_index: int, target_command: float) -> np.ndarray:
    sent_commands = np.zeros(config.motor_count, dtype=np.float64)
    sent_commands[target_index] = float(target_command)
    return sent_commands


def _expected_position_vector(config: Config, *, target_index: int, target_position: float) -> np.ndarray:
    expected = np.zeros(config.motor_count, dtype=np.float64)
    expected[target_index] = float(target_position)
    return expected


def _expected_velocity_vector(config: Config, *, target_index: int, target_velocity: float) -> np.ndarray:
    expected = np.zeros(config.motor_count, dtype=np.float64)
    expected[target_index] = float(target_velocity)
    return expected


def _poll_feedback_frames(
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


def _safety_margin_text(config: Config, observed_velocity: float, command_value: float) -> str:
    return (
        f"velocity_margin={float(config.safety.hard_speed_abort_abs) - abs(float(observed_velocity)):+.6f}, "
        f"command={float(command_value):+.6f}"
    )


def _log_stage_transition(stage: str, *, target_motor_id: int, detail: str = "") -> None:
    message = f"Stage {str(stage)}: motor_id={int(target_motor_id)}"
    if detail:
        message += f", {str(detail)}"
    log_info(message)


def _load_compensation_parameters(config: Config, *, target_motor_id: int) -> _CompensationParameters:
    payload = load_latest_parameters(config)
    motors = payload.get("motors", {})
    entry = motors.get(str(int(target_motor_id)))
    latest_path = latest_parameters_path(config)
    if not isinstance(entry, dict):
        raise ValueError(f"latest motor parameters file does not contain motor_id={int(target_motor_id)}: {latest_path}")

    required_fields = (
        "motor_id",
        "motor_name",
        "identified_at",
        "source_run_label",
        "tau_static",
        "tau_bias",
        "tau_c",
        "viscous",
        "inertia",
        "friction_validation_rmse",
        "inertia_validation_rmse",
        "repeat_consistency_score",
        "recommended_for_compensation",
    )
    missing_fields = [field_name for field_name in required_fields if field_name not in entry]
    if missing_fields:
        raise ValueError(
            "latest motor parameters entry is missing required field(s) for "
            f"motor_id={int(target_motor_id)}: {', '.join(missing_fields)}"
        )

    return _CompensationParameters(
        motor_id=int(entry["motor_id"]),
        motor_name=str(entry["motor_name"]),
        identified_at=str(entry["identified_at"]),
        source_run_label=str(entry["source_run_label"]),
        tau_static=float(entry["tau_static"]),
        tau_bias=float(entry["tau_bias"]),
        tau_c=float(entry["tau_c"]),
        viscous=float(entry["viscous"]),
        inertia=float(entry["inertia"]),
        friction_validation_rmse=float(entry["friction_validation_rmse"]),
        inertia_validation_rmse=float(entry["inertia_validation_rmse"]),
        repeat_consistency_score=float(entry["repeat_consistency_score"]),
        recommended_for_compensation=bool(entry["recommended_for_compensation"]),
    )


def _limit_torque_command(transport: CommandTransport, *, target_motor_id: int, torque: float) -> float:
    limiter = getattr(transport, "limit_torque_command", None)
    if callable(limiter):
        return float(limiter(int(target_motor_id), float(torque)))
    return float(torque)


def _compensation_history_window(config: Config) -> int:
    window = max(
        int(config.identification.savgol_window),
        int(config.identification.savgol_polyorder) + 2,
        3,
    )
    if window % 2 == 0:
        window += 1
    return window


def _compute_compensation_state(
    *,
    time_history: deque[float],
    velocity_history: deque[float],
    config: Config,
) -> tuple[float, float]:
    if not velocity_history:
        return 0.0, 0.0
    if len(time_history) < 2:
        return float(velocity_history[-1]), 0.0
    filtered_velocity, acceleration = estimate_filtered_velocity_and_acceleration(
        np.asarray(tuple(time_history), dtype=np.float64),
        np.asarray(tuple(velocity_history), dtype=np.float64),
        savgol_window=int(config.identification.savgol_window),
        savgol_polyorder=int(config.identification.savgol_polyorder),
    )
    return float(filtered_velocity[-1]), float(acceleration[-1])


def _compensation_torque(
    parameters: _CompensationParameters,
    *,
    filtered_velocity: float,
    acceleration: float,
) -> float:
    return float(
        float(parameters.tau_c) * float(np.sign(float(filtered_velocity)))
        + float(parameters.viscous) * float(filtered_velocity)
        + float(parameters.tau_bias)
        + float(parameters.inertia) * float(acceleration)
    )


def _send_command(
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
    elif semantic_mode == "velocity_mode":
        packet = transport.send_velocity_mode(int(target_motor_id), float(command_value))
    else:  # pragma: no cover - guarded by Literal type
        raise ValueError(f"Unsupported semantic_mode: {semantic_mode}")

    rerun_recorder.log_live_command_packet(
        sent_commands=_sent_command_vector(config, target_index=target_index, target_command=float(command_value)),
        expected_positions=_expected_position_vector(config, target_index=target_index, target_position=float(position_cmd)),
        expected_velocities=_expected_velocity_vector(config, target_index=target_index, target_velocity=float(velocity_cmd)),
        raw_packet=packet,
    )
    return packet


def _record_target_frame(
    *,
    config: Config,
    rerun_recorder: RerunRecorder,
    capture_buffer: _CaptureBuffer | None,
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
) -> None:
    if capture_buffer is not None:
        capture_buffer.append(
            frame=frame,
            command_raw=float(command_raw),
            command=float(command),
            position_cmd=float(position_cmd),
            velocity_cmd=float(velocity_cmd),
            acceleration_cmd=float(acceleration_cmd),
            phase_name=str(phase_name),
        )

    target_index = config.motor_index(target_motor_id)
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
        safety_margin_text=_safety_margin_text(config, float(frame.velocity), float(command)),
    )


def _build_abort_event(
    *,
    config: Config,
    stage: str,
    group_index: int,
    round_index: int,
    phase_name: str,
    target_motor_id: int,
    frame,
) -> AbortEvent | None:
    if abs(float(frame.velocity)) < float(config.safety.hard_speed_abort_abs):
        return None
    return AbortEvent(
        reason="hard_speed_abort",
        stage=str(stage),
        motor_id=int(target_motor_id),
        group_index=int(group_index),
        round_index=int(round_index),
        phase_name=str(phase_name),
        observed_velocity=float(frame.velocity),
        velocity_limit=float(config.safety.hard_speed_abort_abs),
        detail=f"abs_velocity={abs(float(frame.velocity)):.6f}",
    )


def _perform_hard_abort(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    target_motor_id: int,
    semantic_mode: SemanticMode,
) -> None:
    for _ in range(ABORT_ZERO_COMMAND_REPEAT):
        transport.send_zero_command(int(target_motor_id), semantic_mode)
    time.sleep(float(config.safety.post_abort_disable_delay_ms) / 1000.0)

    recent_velocity = float("inf")
    deadline = time.monotonic() + float(config.safety.post_abort_disable_delay_ms) / 1000.0
    while time.monotonic() < deadline:
        frames, saw_chunk = _poll_feedback_frames(
            transport=transport,
            parser=parser,
            read_chunk_size=config.transport.read_chunk_size,
        )
        for frame in frames:
            if int(frame.motor_id) != int(target_motor_id):
                continue
            recent_velocity = abs(float(frame.velocity))
        if not frames and not saw_chunk:
            time.sleep(max(float(config.transport.read_timeout), 1.0e-3))
    if not np.isfinite(recent_velocity) or recent_velocity >= float(config.safety.moving_velocity_threshold):
        transport.disable_motor(int(target_motor_id))


def _wait_for_stationary(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    phase_name: str,
    stage: str,
    semantic_mode: SemanticMode,
    capture_buffer: _CaptureBuffer | None = None,
    timeout_s: float | None = None,
) -> None:
    target_index = config.motor_index(target_motor_id)
    timeout_s = float(config.transport.sync_timeout if timeout_s is None else timeout_s)
    settle_required_s = float(config.safety.moving_hold_ms) / 1000.0
    send_interval_s = max(float(config.transport.read_timeout), 5.0e-3)
    deadline = time.monotonic() + timeout_s
    last_send = 0.0
    stable_started_at: float | None = None

    while time.monotonic() < deadline:
        now = time.monotonic()
        if (now - last_send) >= send_interval_s:
            _send_command(
                config=config,
                transport=transport,
                rerun_recorder=rerun_recorder,
                target_motor_id=int(target_motor_id),
                target_index=target_index,
                semantic_mode=semantic_mode,
                command_value=0.0,
                kd_speed=float(getattr(config.mit_velocity, "kd_speed")[target_index]),
                position_cmd=0.0,
                velocity_cmd=0.0,
            )
            last_send = now

        frames, saw_chunk = _poll_feedback_frames(
            transport=transport,
            parser=parser,
            read_chunk_size=config.transport.read_chunk_size,
        )
        saw_target = False
        for frame in frames:
            rerun_recorder.log_live_feedback_frame(
                group_index=int(group_index),
                round_index=int(round_index),
                active_motor_id=int(target_motor_id),
                motor_id=int(frame.motor_id),
                state=int(frame.state),
                position=float(frame.position),
                velocity=float(frame.velocity),
                feedback_torque=float(frame.torque),
                mos_temperature=float(frame.mos_temperature),
                phase_name=str(phase_name),
                stage=str(stage),
            )
            if int(frame.motor_id) != int(target_motor_id):
                continue
            saw_target = True
            abort_event = _build_abort_event(
                config=config,
                stage=stage,
                group_index=group_index,
                round_index=round_index,
                phase_name=phase_name,
                target_motor_id=target_motor_id,
                frame=frame,
            )
            if abort_event is not None:
                _perform_hard_abort(
                    config=config,
                    transport=transport,
                    parser=parser,
                    target_motor_id=target_motor_id,
                    semantic_mode=semantic_mode,
                )
                raise _RuntimeAbortError(abort_event)

            _record_target_frame(
                config=config,
                rerun_recorder=rerun_recorder,
                capture_buffer=capture_buffer,
                group_index=group_index,
                round_index=round_index,
                target_motor_id=target_motor_id,
                frame=frame,
                command_raw=0.0,
                command=0.0,
                position_cmd=0.0,
                velocity_cmd=0.0,
                acceleration_cmd=0.0,
                phase_name=phase_name,
                stage=stage,
            )
            if abs(float(frame.velocity)) <= float(config.safety.moving_velocity_threshold):
                stable_started_at = now if stable_started_at is None else stable_started_at
                if (now - stable_started_at) >= settle_required_s:
                    return
            else:
                stable_started_at = None
        if not saw_target and not saw_chunk:
            time.sleep(max(float(config.transport.read_timeout), 1.0e-3))

    raise _RuntimeAbortError(
        AbortEvent(
            reason="stationary_timeout",
            stage=str(stage),
            motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
            phase_name=str(phase_name),
            detail=f"timeout_s={timeout_s:.3f}",
        )
    )


def _run_velocity_segment(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    capture_buffer: _CaptureBuffer,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    phase_name: str,
    stage: str,
    start_velocity: float,
    end_velocity: float,
    duration_s: float,
    kd_speed: float,
    semantic_mode: SemanticMode = "mit_velocity",
) -> float:
    target_index = config.motor_index(target_motor_id)
    send_interval_s = max(float(config.transport.read_timeout), 5.0e-3)
    start_monotonic = time.monotonic()
    last_send = 0.0
    duration_s = max(float(duration_s), 0.0)
    acceleration_cmd = 0.0 if duration_s <= 0.0 else (float(end_velocity) - float(start_velocity)) / duration_s
    current_velocity_cmd = float(start_velocity)

    while True:
        now = time.monotonic()
        elapsed = now - start_monotonic
        progress = 1.0 if duration_s <= 0.0 else min(elapsed / duration_s, 1.0)
        current_velocity_cmd = float(start_velocity) + (float(end_velocity) - float(start_velocity)) * progress

        if (now - last_send) >= send_interval_s:
            _send_command(
                config=config,
                transport=transport,
                rerun_recorder=rerun_recorder,
                target_motor_id=int(target_motor_id),
                target_index=target_index,
                semantic_mode=semantic_mode,
                command_value=float(current_velocity_cmd),
                kd_speed=float(kd_speed),
                position_cmd=0.0,
                velocity_cmd=float(current_velocity_cmd),
            )
            last_send = now

        frames, saw_chunk = _poll_feedback_frames(
            transport=transport,
            parser=parser,
            read_chunk_size=config.transport.read_chunk_size,
        )
        saw_target = False
        for frame in frames:
            rerun_recorder.log_live_feedback_frame(
                group_index=int(group_index),
                round_index=int(round_index),
                active_motor_id=int(target_motor_id),
                motor_id=int(frame.motor_id),
                state=int(frame.state),
                position=float(frame.position),
                velocity=float(frame.velocity),
                feedback_torque=float(frame.torque),
                mos_temperature=float(frame.mos_temperature),
                phase_name=str(phase_name),
                stage=str(stage),
            )
            if int(frame.motor_id) != int(target_motor_id):
                continue
            saw_target = True
            abort_event = _build_abort_event(
                config=config,
                stage=stage,
                group_index=group_index,
                round_index=round_index,
                phase_name=phase_name,
                target_motor_id=target_motor_id,
                frame=frame,
            )
            if abort_event is not None:
                _perform_hard_abort(
                    config=config,
                    transport=transport,
                    parser=parser,
                    target_motor_id=target_motor_id,
                    semantic_mode=semantic_mode,
                )
                raise _RuntimeAbortError(abort_event)
            _record_target_frame(
                config=config,
                rerun_recorder=rerun_recorder,
                capture_buffer=capture_buffer,
                group_index=group_index,
                round_index=round_index,
                target_motor_id=target_motor_id,
                frame=frame,
                command_raw=float(current_velocity_cmd),
                command=float(current_velocity_cmd),
                position_cmd=0.0,
                velocity_cmd=float(current_velocity_cmd),
                acceleration_cmd=float(acceleration_cmd),
                phase_name=phase_name,
                stage=stage,
            )

        if duration_s <= 0.0 or elapsed >= duration_s:
            break
        if not saw_target and not saw_chunk:
            time.sleep(max(float(config.transport.read_timeout), 1.0e-3))

    return float(current_velocity_cmd)


def _scan_breakaway_direction(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    capture_buffer: _CaptureBuffer,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    direction: int,
) -> float:
    direction_label = "pos" if int(direction) > 0 else "neg"
    target_index = config.motor_index(target_motor_id)
    scan_limit = float(config.breakaway.scan_max_torque[target_index])
    send_interval_s = max(float(config.transport.read_timeout), 5.0e-3)
    moving_hold_s = float(config.safety.moving_hold_ms) / 1000.0
    torque_step = float(config.breakaway.torque_step)
    hold_duration = float(config.breakaway.hold_duration)
    torque_values = np.arange(torque_step, scan_limit + torque_step * 0.5, torque_step, dtype=np.float64)

    _wait_for_stationary(
        config=config,
        transport=transport,
        parser=parser,
        rerun_recorder=rerun_recorder,
        target_motor_id=target_motor_id,
        group_index=group_index,
        round_index=round_index,
        phase_name=f"breakaway_{direction_label}_settle",
        stage="breakaway",
        semantic_mode="mit_torque",
        capture_buffer=capture_buffer,
    )

    for step_index, torque_value in enumerate(torque_values, start=1):
        phase_name = f"breakaway_{direction_label}_step_{int(step_index):03d}"
        command_value = float(direction) * float(torque_value)
        start_monotonic = time.monotonic()
        last_send = 0.0
        moving_started_at: float | None = None
        while True:
            now = time.monotonic()
            elapsed = now - start_monotonic
            if (now - last_send) >= send_interval_s:
                _send_command(
                    config=config,
                    transport=transport,
                    rerun_recorder=rerun_recorder,
                    target_motor_id=int(target_motor_id),
                    target_index=target_index,
                    semantic_mode="mit_torque",
                    command_value=float(command_value),
                )
                last_send = now

            frames, saw_chunk = _poll_feedback_frames(
                transport=transport,
                parser=parser,
                read_chunk_size=config.transport.read_chunk_size,
            )
            saw_target = False
            for frame in frames:
                rerun_recorder.log_live_feedback_frame(
                    group_index=int(group_index),
                    round_index=int(round_index),
                    active_motor_id=int(target_motor_id),
                    motor_id=int(frame.motor_id),
                    state=int(frame.state),
                    position=float(frame.position),
                    velocity=float(frame.velocity),
                    feedback_torque=float(frame.torque),
                    mos_temperature=float(frame.mos_temperature),
                    phase_name=str(phase_name),
                    stage="breakaway",
                )
                if int(frame.motor_id) != int(target_motor_id):
                    continue
                saw_target = True
                abort_event = _build_abort_event(
                    config=config,
                    stage="breakaway",
                    group_index=group_index,
                    round_index=round_index,
                    phase_name=phase_name,
                    target_motor_id=target_motor_id,
                    frame=frame,
                )
                if abort_event is not None:
                    _perform_hard_abort(
                        config=config,
                        transport=transport,
                        parser=parser,
                        target_motor_id=target_motor_id,
                        semantic_mode="mit_torque",
                    )
                    raise _RuntimeAbortError(abort_event)
                _record_target_frame(
                    config=config,
                    rerun_recorder=rerun_recorder,
                    capture_buffer=capture_buffer,
                    group_index=group_index,
                    round_index=round_index,
                    target_motor_id=target_motor_id,
                    frame=frame,
                    command_raw=float(command_value),
                    command=float(command_value),
                    position_cmd=0.0,
                    velocity_cmd=0.0,
                    acceleration_cmd=0.0,
                    phase_name=phase_name,
                    stage="breakaway",
                )
                if abs(float(frame.velocity)) > float(config.safety.moving_velocity_threshold):
                    moving_started_at = now if moving_started_at is None else moving_started_at
                    if (now - moving_started_at) >= moving_hold_s:
                        return float(command_value)
                else:
                    moving_started_at = None
            if elapsed >= hold_duration:
                break
            if not saw_target and not saw_chunk:
                time.sleep(max(float(config.transport.read_timeout), 1.0e-3))

    return float(direction) * float(scan_limit)


def _run_breakaway_phase(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    capture_buffer: _CaptureBuffer,
    target_motor_id: int,
    group_index: int,
    round_index: int,
) -> BreakawayIdentificationResult:
    _log_stage_transition("breakaway", target_motor_id=target_motor_id)
    rerun_recorder.log_phase_event(motor_id=int(target_motor_id), phase_name="breakaway", detail="start")
    positive = _scan_breakaway_direction(
        config=config,
        transport=transport,
        parser=parser,
        rerun_recorder=rerun_recorder,
        capture_buffer=capture_buffer,
        target_motor_id=target_motor_id,
        group_index=group_index,
        round_index=round_index,
        direction=1,
    )
    negative = _scan_breakaway_direction(
        config=config,
        transport=transport,
        parser=parser,
        rerun_recorder=rerun_recorder,
        capture_buffer=capture_buffer,
        target_motor_id=target_motor_id,
        group_index=group_index,
        round_index=round_index,
        direction=-1,
    )
    tau_static = 0.5 * (float(positive) - float(negative))
    tau_bias = 0.5 * (float(positive) + float(negative))
    rerun_recorder.log_phase_event(
        motor_id=int(target_motor_id),
        phase_name="breakaway",
        detail=f"positive={positive:+.4f}, negative={negative:+.4f}",
    )
    return BreakawayIdentificationResult(
        torque_positive=float(positive),
        torque_negative=float(negative),
        tau_static=float(tau_static),
        tau_bias=float(tau_bias),
        metadata={
            "scan_max_torque": float(config.breakaway.scan_max_torque[config.motor_index(target_motor_id)]),
            "torque_step": float(config.breakaway.torque_step),
            "hold_duration": float(config.breakaway.hold_duration),
        },
    )


def _run_speed_hold_phase(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    capture_buffer: _CaptureBuffer,
    target_motor_id: int,
    group_index: int,
    round_index: int,
) -> None:
    _log_stage_transition("speed-hold", target_motor_id=target_motor_id)
    rerun_recorder.log_phase_event(motor_id=int(target_motor_id), phase_name="speed-hold", detail="start")
    target_index = config.motor_index(target_motor_id)
    kd_speed = float(config.mit_velocity.kd_speed[target_index])
    ramp_acceleration = float(config.mit_velocity.ramp_acceleration)
    hold_duration = float(config.mit_velocity.steady_hold_duration)
    holdout_speed = max(float(item) for item in config.identification.steady_speed_points)
    current_velocity = 0.0

    speed_points: list[float] = [float(point) for point in config.identification.steady_speed_points]
    speed_points.extend([-float(point) for point in config.identification.steady_speed_points])
    for target_velocity in speed_points:
        bucket = "valid" if np.isclose(abs(float(target_velocity)), holdout_speed) else "train"
        ramp_duration = abs(float(target_velocity) - float(current_velocity)) / ramp_acceleration
        current_velocity = _run_velocity_segment(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=target_motor_id,
            group_index=group_index,
            round_index=round_index,
            phase_name=f"speed_ramp_{bucket}_{float(target_velocity):+0.2f}",
            stage="speed-hold",
            start_velocity=float(current_velocity),
            end_velocity=float(target_velocity),
            duration_s=float(ramp_duration),
            kd_speed=kd_speed,
        )
        current_velocity = _run_velocity_segment(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=target_motor_id,
            group_index=group_index,
            round_index=round_index,
            phase_name=f"speed_hold_{bucket}_{float(target_velocity):+0.2f}",
            stage="speed-hold",
            start_velocity=float(current_velocity),
            end_velocity=float(target_velocity),
            duration_s=hold_duration,
            kd_speed=kd_speed,
        )

    current_velocity = _run_velocity_segment(
        config=config,
        transport=transport,
        parser=parser,
        rerun_recorder=rerun_recorder,
        capture_buffer=capture_buffer,
        target_motor_id=target_motor_id,
        group_index=group_index,
        round_index=round_index,
        phase_name="speed_ramp_return_0.00",
        stage="speed-hold",
        start_velocity=float(current_velocity),
        end_velocity=0.0,
        duration_s=abs(float(current_velocity)) / ramp_acceleration,
        kd_speed=kd_speed,
    )
    _wait_for_stationary(
        config=config,
        transport=transport,
        parser=parser,
        rerun_recorder=rerun_recorder,
        target_motor_id=target_motor_id,
        group_index=group_index,
        round_index=round_index,
        phase_name="speed_hold_settle",
        stage="speed-hold",
        semantic_mode="mit_velocity",
        capture_buffer=capture_buffer,
    )


def _run_inertia_phase(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    capture_buffer: _CaptureBuffer,
    target_motor_id: int,
    group_index: int,
    round_index: int,
) -> None:
    _log_stage_transition("inertia", target_motor_id=target_motor_id)
    rerun_recorder.log_phase_event(motor_id=int(target_motor_id), phase_name="inertia", detail="start")
    target_index = config.motor_index(target_motor_id)
    kd_speed = float(config.mit_velocity.kd_speed[target_index])
    ramp_acceleration = float(config.mit_velocity.ramp_acceleration)
    waypoints = [0.0, 2.0, 4.0, 6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0, -4.0, -2.0, 0.0]
    current_velocity = float(waypoints[0])
    midpoint = 6
    for segment_index, target_velocity in enumerate(waypoints[1:], start=1):
        bucket = "train" if segment_index <= midpoint else "valid"
        current_velocity = _run_velocity_segment(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=target_motor_id,
            group_index=group_index,
            round_index=round_index,
            phase_name=f"inertia_{bucket}_{segment_index:02d}",
            stage="inertia",
            start_velocity=float(current_velocity),
            end_velocity=float(target_velocity),
            duration_s=abs(float(target_velocity) - float(current_velocity)) / ramp_acceleration,
            kd_speed=kd_speed,
        )
    _wait_for_stationary(
        config=config,
        transport=transport,
        parser=parser,
        rerun_recorder=rerun_recorder,
        target_motor_id=target_motor_id,
        group_index=group_index,
        round_index=round_index,
        phase_name="inertia_settle",
        stage="inertia",
        semantic_mode="mit_velocity",
        capture_buffer=capture_buffer,
    )


def _send_zero_then_disable(
    *,
    config: Config,
    transport: CommandTransport,
    target_motor_id: int,
    semantic_mode: SemanticMode,
) -> None:
    for _ in range(ABORT_ZERO_COMMAND_REPEAT):
        transport.send_zero_command(int(target_motor_id), semantic_mode)
    time.sleep(float(config.safety.post_abort_disable_delay_ms) / 1000.0)
    transport.disable_motor(int(target_motor_id))


def _run_compensation_phase(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    capture_buffer: _CaptureBuffer,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    parameters: _CompensationParameters,
    max_runtime_s: float | None,
) -> None:
    _log_stage_transition(
        "compensation",
        target_motor_id=target_motor_id,
        detail=f"source_run_label={parameters.source_run_label}",
    )
    rerun_recorder.log_phase_event(
        motor_id=int(target_motor_id),
        phase_name="compensation",
        detail=f"start source_run_label={parameters.source_run_label}",
    )
    target_index = config.motor_index(target_motor_id)
    phase_name = "compensation_active"
    started_at = time.monotonic()
    runtime_limit = None if max_runtime_s is None else max(float(max_runtime_s), 0.0)
    history_window = _compensation_history_window(config)
    time_history: deque[float] = deque(maxlen=history_window)
    velocity_history: deque[float] = deque(maxlen=history_window)

    while True:
        if runtime_limit is not None and (time.monotonic() - started_at) >= runtime_limit:
            break

        frames, saw_chunk = _poll_feedback_frames(
            transport=transport,
            parser=parser,
            read_chunk_size=config.transport.read_chunk_size,
        )
        saw_target = False
        for frame in frames:
            rerun_recorder.log_live_feedback_frame(
                group_index=int(group_index),
                round_index=int(round_index),
                active_motor_id=int(target_motor_id),
                motor_id=int(frame.motor_id),
                state=int(frame.state),
                position=float(frame.position),
                velocity=float(frame.velocity),
                feedback_torque=float(frame.torque),
                mos_temperature=float(frame.mos_temperature),
                phase_name=phase_name,
                stage="compensation",
            )
            if int(frame.motor_id) != int(target_motor_id):
                continue
            saw_target = True
            abort_event = _build_abort_event(
                config=config,
                stage="compensation",
                group_index=group_index,
                round_index=round_index,
                phase_name=phase_name,
                target_motor_id=target_motor_id,
                frame=frame,
            )
            if abort_event is not None:
                _perform_hard_abort(
                    config=config,
                    transport=transport,
                    parser=parser,
                    target_motor_id=target_motor_id,
                    semantic_mode="mit_torque",
                )
                raise _RuntimeAbortError(abort_event)

            sample_time = float(time.monotonic() - capture_buffer.start_monotonic)
            time_history.append(sample_time)
            velocity_history.append(float(frame.velocity))
            filtered_velocity, acceleration = _compute_compensation_state(
                time_history=time_history,
                velocity_history=velocity_history,
                config=config,
            )
            command_raw = _compensation_torque(
                parameters,
                filtered_velocity=float(filtered_velocity),
                acceleration=float(acceleration),
            )
            command = _limit_torque_command(
                transport,
                target_motor_id=target_motor_id,
                torque=float(command_raw),
            )
            _send_command(
                config=config,
                transport=transport,
                rerun_recorder=rerun_recorder,
                target_motor_id=int(target_motor_id),
                target_index=target_index,
                semantic_mode="mit_torque",
                command_value=float(command),
                position_cmd=0.0,
                velocity_cmd=0.0,
            )
            _record_target_frame(
                config=config,
                rerun_recorder=rerun_recorder,
                capture_buffer=capture_buffer,
                group_index=group_index,
                round_index=round_index,
                target_motor_id=target_motor_id,
                frame=frame,
                command_raw=float(command_raw),
                command=float(command),
                position_cmd=0.0,
                velocity_cmd=0.0,
                acceleration_cmd=float(acceleration),
                phase_name=phase_name,
                stage="compensation",
            )
        if not saw_target and not saw_chunk:
            time.sleep(max(float(config.transport.read_timeout), 1.0e-3))


def _late_portion_mask(phase_names: np.ndarray, *, prefix: str, ratio: float) -> np.ndarray:
    phase_names = np.asarray(phase_names).astype(str)
    mask = np.zeros(phase_names.size, dtype=bool)
    ordered_phase_names = list(dict.fromkeys(phase_names.tolist()))
    for phase_name in ordered_phase_names:
        if not str(phase_name).startswith(prefix):
            continue
        indices = np.flatnonzero(phase_names == phase_name)
        if indices.size == 0:
            continue
        start_index = int(np.floor((1.0 - float(ratio)) * indices.size))
        mask[indices[start_index:]] = True
    return mask


def _empty_friction_result(size: int, *, status: str) -> FrictionIdentificationResult:
    return FrictionIdentificationResult(
        tau_c=float("nan"),
        viscous=float("nan"),
        tau_bias=float("nan"),
        train_rmse=float("nan"),
        valid_rmse=float("nan"),
        train_mask=np.zeros(size, dtype=bool),
        valid_mask=np.zeros(size, dtype=bool),
        torque_pred=np.full(size, np.nan, dtype=np.float64),
        torque_target=np.full(size, np.nan, dtype=np.float64),
        metadata={"status": status},
    )


def _empty_inertia_result(size: int, *, status: str) -> InertiaIdentificationResult:
    return InertiaIdentificationResult(
        inertia=float("nan"),
        train_rmse=float("nan"),
        valid_rmse=float("nan"),
        train_mask=np.zeros(size, dtype=bool),
        valid_mask=np.zeros(size, dtype=bool),
        torque_pred=np.full(size, np.nan, dtype=np.float64),
        torque_target=np.full(size, np.nan, dtype=np.float64),
        filtered_velocity=np.full(size, np.nan, dtype=np.float64),
        acceleration=np.full(size, np.nan, dtype=np.float64),
        metadata={"status": status},
    )


def _empty_validation_result(*, status: str) -> ValidationResult:
    return ValidationResult(
        friction_rmse=float("nan"),
        inertia_rmse=float("nan"),
        recommended_for_compensation=False,
        detail=status,
        metadata={"status": status},
    )


def _identify_round(
    *,
    config: Config,
    capture: RoundCapture,
    mode: str,
    breakaway_result: BreakawayIdentificationResult,
) -> MotorIdentificationResult:
    sample_count = capture.sample_count
    phase_names = np.asarray(capture.phase_name).astype(str)
    friction_result = _empty_friction_result(sample_count, status="not_run")
    inertia_result = _empty_inertia_result(sample_count, status="not_run")
    validation_result = _empty_validation_result(status="not_run")

    if mode in {"identify-all", "speed-hold", "inertia"}:
        friction_train_mask = _late_portion_mask(
            phase_names,
            prefix="speed_hold_train_",
            ratio=float(config.mit_velocity.steady_window_ratio),
        )
        friction_valid_mask = _late_portion_mask(
            phase_names,
            prefix="speed_hold_valid_",
            ratio=float(config.mit_velocity.steady_window_ratio),
        )
        friction_result = fit_friction_model(
            capture.velocity,
            capture.torque_feedback,
            train_mask=friction_train_mask,
            valid_mask=friction_valid_mask,
        )

    if mode in {"identify-all", "inertia"}:
        inertia_train_mask = np.asarray([name.startswith("inertia_train_") for name in phase_names], dtype=bool)
        inertia_valid_mask = np.asarray([name.startswith("inertia_valid_") for name in phase_names], dtype=bool)
        inertia_result = fit_inertia_model(
            capture.time,
            capture.velocity,
            capture.torque_feedback,
            friction_result=friction_result,
            train_mask=inertia_train_mask,
            valid_mask=inertia_valid_mask,
            savgol_window=int(config.identification.savgol_window),
            savgol_polyorder=int(config.identification.savgol_polyorder),
        )
        validation_result = build_validation_result(friction_result, inertia_result)
    elif mode == "speed-hold":
        validation_result = ValidationResult(
            friction_rmse=float(friction_result.valid_rmse),
            inertia_rmse=float("nan"),
            recommended_for_compensation=False,
            detail="speed-hold debug mode",
            metadata={"status": "partial"},
        )
    elif mode == "breakaway":
        validation_result = _empty_validation_result(status="breakaway_only")

    return MotorIdentificationResult(
        motor_id=int(capture.target_motor_id),
        motor_name=str(capture.motor_name),
        breakaway=breakaway_result,
        friction=friction_result,
        inertia=inertia_result,
        validation=validation_result,
        metadata={
            "mode": str(mode),
            "steady_window_ratio": float(config.mit_velocity.steady_window_ratio),
            "repeat_index": int(capture.group_index),
            "round_index": int(capture.round_index),
        },
    )


def _empty_breakaway_result(*, status: str) -> BreakawayIdentificationResult:
    return BreakawayIdentificationResult(
        torque_positive=float("nan"),
        torque_negative=float("nan"),
        tau_static=float("nan"),
        tau_bias=float("nan"),
        metadata={"status": status},
    )


def _precheck_transport(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
) -> None:
    for motor_id in config.enabled_motor_ids:
        motor_type_name = getattr(transport, "motor_type_name", lambda current_motor_id: config.transport.motor_types[config.motor_index(current_motor_id)])
        motor_limits = getattr(transport, "motor_limits", lambda current_motor_id: None)
        description = f"type={motor_type_name(int(motor_id))}"
        limits = motor_limits(int(motor_id))
        if limits is not None:
            description += f" pmax={float(limits.pmax):.3f} vmax={float(limits.vmax):.3f} tmax={float(limits.tmax):.3f}"
        _log_stage_transition("precheck", target_motor_id=int(motor_id), detail=description)
        transport.clear_error(int(motor_id))
        transport.enable_motor(int(motor_id))
        _wait_for_stationary(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            target_motor_id=int(motor_id),
            group_index=0,
            round_index=0,
            phase_name="precheck_zero",
            stage="precheck",
            semantic_mode="mit_torque",
            capture_buffer=None,
        )


def _run_motor_round(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    mode: str,
) -> tuple[RoundCapture, MotorIdentificationResult]:
    if config.transport.flush_input_before_round:
        transport.reset_input_buffer()
        parser.reset()

    capture_buffer = _CaptureBuffer(
        target_motor_id=int(target_motor_id),
        motor_name=config.motors.name_for(int(target_motor_id)),
    )
    breakaway_result = _empty_breakaway_result(status="not_run")
    if mode in {"identify-all", "breakaway"}:
        breakaway_result = _run_breakaway_phase(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
        )
    if mode in {"identify-all", "speed-hold", "inertia"}:
        _run_speed_hold_phase(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
        )
    if mode in {"identify-all", "inertia"}:
        _run_inertia_phase(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
        )

    capture = capture_buffer.build(
        group_index=int(group_index),
        round_index=int(round_index),
        metadata={
            "mode": str(mode),
            "enabled_motor_ids": list(config.enabled_motor_ids),
            "hard_speed_abort_abs": float(config.safety.hard_speed_abort_abs),
            "moving_velocity_threshold": float(config.safety.moving_velocity_threshold),
            "repeat_count": int(config.identification.repeat_count),
        },
    )
    identification = _identify_round(
        config=config,
        capture=capture,
        mode=mode,
        breakaway_result=breakaway_result,
    )
    rerun_recorder.log_round_stop(
        group_index=int(group_index),
        round_index=int(round_index),
        motor_id=int(target_motor_id),
        phase_name="completed",
        stage=mode,
    )
    return capture, identification


def _run_mode(
    config: Config,
    *,
    mode: str,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    store = ResultStore(config, mode=mode)
    parser = FeedbackFrameParser(max_motor_id=max(config.motor_ids))
    rerun_recorder = RerunRecorder(
        store.rerun_recording_path,
        motor_ids=config.motor_ids,
        motor_names={motor_id: config.motors.name_for(motor_id) for motor_id in config.motor_ids},
        mode=mode,
        show_viewer=show_rerun_viewer,
    )
    artifacts: list[RoundArtifact] = []
    transport = transport_factory() if transport_factory is not None else open_transport(config)

    try:
        _precheck_transport(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
        )
        total_rounds = int(config.identification.repeat_count) * len(config.enabled_motor_ids)
        current_round = 0
        for group_index in range(1, int(config.identification.repeat_count) + 1):
            for target_motor_id in config.enabled_motor_ids:
                current_round += 1
                rerun_recorder.log_round_timing(
                    group_index=int(group_index),
                    round_index=int(current_round),
                    active_motor_id=int(target_motor_id),
                    planned_duration_s=0.0,
                    actual_capture_duration_s=0.0,
                    sync_wait_duration_s=0.0,
                    round_total_duration_s=0.0,
                )
                log_info(
                    f"Starting {mode} round {current_round}/{total_rounds}: "
                    f"repeat={group_index}, motor_id={target_motor_id}"
                )
                round_started = time.monotonic()
                capture, identification = _run_motor_round(
                    config=config,
                    transport=transport,
                    parser=parser,
                    rerun_recorder=rerun_recorder,
                    target_motor_id=int(target_motor_id),
                    group_index=int(group_index),
                    round_index=int(current_round),
                    mode=mode,
                )
                rerun_recorder.log_round_timing(
                    group_index=int(group_index),
                    round_index=int(current_round),
                    active_motor_id=int(target_motor_id),
                    planned_duration_s=float(capture.time[-1]) if capture.sample_count else 0.0,
                    actual_capture_duration_s=float(capture.time[-1]) if capture.sample_count else 0.0,
                    sync_wait_duration_s=0.0,
                    round_total_duration_s=float(time.monotonic() - round_started),
                )
                capture_path = store.save_capture(capture)
                identification_path = store.save_identification(capture, identification)
                artifacts.append(
                    RoundArtifact(
                        capture=capture,
                        identification=identification,
                        capture_path=capture_path,
                        identification_path=identification_path,
                    )
                )
                log_info(
                    f"motor_id={target_motor_id} finished: "
                    f"tau_static={float(identification.breakaway.tau_static):+.4f}, "
                    f"tau_c={float(identification.friction.tau_c):+.4f}, "
                    f"viscous={float(identification.friction.viscous):+.4f}, "
                    f"inertia={float(identification.inertia.inertia):+.4f}"
                )

        summary_paths = store.save_summary(artifacts)
        if mode == "identify-all":
            store.save_latest_parameters(artifacts)
        rerun_recorder.log_summary(
            summary_path=summary_paths.run_summary_path,
            report_path=summary_paths.run_summary_report_path,
        )
        return RunResult(
            artifacts=tuple(artifacts),
            summary_paths=summary_paths,
            manifest_path=store.manifest_path,
        )
    except _RuntimeAbortError as exc:
        rerun_recorder.log_abort_event(exc.event.to_payload())
        store.record_abort_event(exc.event.to_payload())
        store.finalize()
        raise
    finally:
        rerun_recorder.close()
        transport.close()


def run_identify_all(
    config: Config,
    *,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    return _run_mode(
        config,
        mode="identify-all",
        transport_factory=transport_factory,
        show_rerun_viewer=show_rerun_viewer,
    )


def run_breakaway(
    config: Config,
    *,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    return _run_mode(
        config,
        mode="breakaway",
        transport_factory=transport_factory,
        show_rerun_viewer=show_rerun_viewer,
    )


def run_speed_hold(
    config: Config,
    *,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    return _run_mode(
        config,
        mode="speed-hold",
        transport_factory=transport_factory,
        show_rerun_viewer=show_rerun_viewer,
    )


def run_inertia(
    config: Config,
    *,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    return _run_mode(
        config,
        mode="inertia",
        transport_factory=transport_factory,
        show_rerun_viewer=show_rerun_viewer,
    )


def run_compensation(
    config: Config,
    *,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
    max_runtime_s: float | None = None,
) -> RunResult:
    if len(config.enabled_motor_ids) != 1:
        raise ValueError("compensation mode requires exactly one enabled motor_id.")

    target_motor_id = int(config.enabled_motor_ids[0])
    parameters = _load_compensation_parameters(config, target_motor_id=target_motor_id)
    if not bool(parameters.recommended_for_compensation):
        log_info(
            f"Warning: motor_id={int(target_motor_id)} latest model is not recommended_for_compensation, "
            f"source_run_label={parameters.source_run_label}"
        )

    store = ResultStore(config, mode="compensation")
    parser = FeedbackFrameParser(max_motor_id=max(config.motor_ids))
    rerun_recorder = RerunRecorder(
        store.rerun_recording_path,
        motor_ids=config.motor_ids,
        motor_names={motor_id: config.motors.name_for(motor_id) for motor_id in config.motor_ids},
        mode="compensation",
        show_viewer=show_rerun_viewer,
    )
    transport = transport_factory() if transport_factory is not None else open_transport(config)
    capture_buffer = _CaptureBuffer(
        target_motor_id=int(target_motor_id),
        motor_name=config.motors.name_for(int(target_motor_id)),
    )
    hard_aborted = False

    try:
        _precheck_transport(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
        )
        rerun_recorder.log_round_timing(
            group_index=1,
            round_index=1,
            active_motor_id=int(target_motor_id),
            planned_duration_s=0.0 if max_runtime_s is None else float(max_runtime_s),
            actual_capture_duration_s=0.0,
            sync_wait_duration_s=0.0,
            round_total_duration_s=0.0,
        )
        log_info(f"Starting compensation round 1/1: motor_id={int(target_motor_id)}")
        round_started = time.monotonic()
        _run_compensation_phase(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=int(target_motor_id),
            group_index=1,
            round_index=1,
            parameters=parameters,
            max_runtime_s=max_runtime_s,
        )
        capture = capture_buffer.build(
            group_index=1,
            round_index=1,
            metadata={
                "mode": "compensation",
                "enabled_motor_ids": list(config.enabled_motor_ids),
                "hard_speed_abort_abs": float(config.safety.hard_speed_abort_abs),
                "moving_velocity_threshold": float(config.safety.moving_velocity_threshold),
                "latest_parameters_path": str(latest_parameters_path(config)),
                "identified_at": str(parameters.identified_at),
                "source_run_label": str(parameters.source_run_label),
                "recommended_for_compensation": bool(parameters.recommended_for_compensation),
            },
        )
        rerun_recorder.log_round_timing(
            group_index=1,
            round_index=1,
            active_motor_id=int(target_motor_id),
            planned_duration_s=0.0 if max_runtime_s is None else float(max_runtime_s),
            actual_capture_duration_s=float(capture.time[-1]) if capture.sample_count else 0.0,
            sync_wait_duration_s=0.0,
            round_total_duration_s=float(time.monotonic() - round_started),
        )
        capture_path = store.save_capture(capture)
        rerun_recorder.log_round_stop(
            group_index=1,
            round_index=1,
            motor_id=int(target_motor_id),
            phase_name="completed",
            stage="compensation",
        )
        store.finalize()
        log_info(
            f"motor_id={int(target_motor_id)} compensation finished: "
            f"source_run_label={parameters.source_run_label}, capture_samples={int(capture.sample_count)}"
        )
        return RunResult(
            artifacts=(capture_path,),
            summary_paths=None,
            manifest_path=store.manifest_path,
        )
    except _RuntimeAbortError as exc:
        hard_aborted = True
        rerun_recorder.log_abort_event(exc.event.to_payload())
        store.record_abort_event(exc.event.to_payload())
        store.finalize()
        raise
    finally:
        try:
            if not hard_aborted:
                _send_zero_then_disable(
                    config=config,
                    transport=transport,
                    target_motor_id=int(target_motor_id),
                    semantic_mode="mit_torque",
                )
        finally:
            rerun_recorder.close()
            transport.close()


__all__ = [
    "run_breakaway",
    "run_compensation",
    "run_identify_all",
    "run_inertia",
    "run_speed_hold",
]
