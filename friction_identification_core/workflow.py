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
    filtered_velocity_log: list[float] = field(default_factory=list)
    estimated_acceleration_log: list[float] = field(default_factory=list)
    friction_term_log: list[float] = field(default_factory=list)
    inertia_term_log: list[float] = field(default_factory=list)
    guard_scale_log: list[float] = field(default_factory=list)

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
        filtered_velocity: float = float("nan"),
        estimated_acceleration: float = float("nan"),
        friction_term: float = float("nan"),
        inertia_term: float = float("nan"),
        guard_scale: float = float("nan"),
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
        self.filtered_velocity_log.append(float(filtered_velocity))
        self.estimated_acceleration_log.append(float(estimated_acceleration))
        self.friction_term_log.append(float(friction_term))
        self.inertia_term_log.append(float(inertia_term))
        self.guard_scale_log.append(float(guard_scale))

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
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class _CompensationParameters:
    motor_id: int
    motor_name: str
    identified_at: str
    source_run_label: str
    model_kind: str
    publish_status: str
    publish_detail: str
    accepted_round_count: int
    selected_rounds: tuple[int, ...]
    tau_static: float
    tau_bias: float
    tau_c: float
    viscous: float
    inertia: float
    friction_validation_rmse: float
    inertia_validation_rmse: float
    repeat_consistency_score: float
    recommended_for_compensation: bool


@dataclass(frozen=True)
class _CompensationCommand:
    raw_torque: float
    direction: float
    friction_term: float
    inertia_term: float
    guard_scale: float


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

    model_kind = str(entry.get("model_kind", "static_v1"))
    publish_status = str(
        entry.get(
            "publish_status",
            "published" if bool(entry.get("recommended_for_compensation", False)) else "not_published",
        )
    )
    publish_detail = str(entry.get("publish_detail", "legacy entry"))
    selected_rounds_raw = entry.get("selected_rounds", ())
    if not isinstance(selected_rounds_raw, (list, tuple)):
        selected_rounds_raw = ()
    selected_rounds = tuple(int(item) for item in selected_rounds_raw)
    accepted_round_count = int(entry.get("accepted_round_count", len(selected_rounds) if selected_rounds else 0))
    if bool(config.compensation.require_published_model) and publish_status != "published":
        raise ValueError(
            f"compensation requires a published model for motor_id={int(target_motor_id)}: "
            f"publish_status={publish_status}, latest_parameters_path={latest_path}"
        )

    return _CompensationParameters(
        motor_id=int(entry["motor_id"]),
        motor_name=str(entry["motor_name"]),
        identified_at=str(entry["identified_at"]),
        source_run_label=str(entry["source_run_label"]),
        model_kind=model_kind,
        publish_status=publish_status,
        publish_detail=publish_detail,
        accepted_round_count=accepted_round_count,
        selected_rounds=selected_rounds,
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


def _compensation_torque_limit_abs(
    transport: CommandTransport,
    *,
    target_motor_id: int,
    config: Config,
) -> float:
    motor_limits = getattr(transport, "motor_limits", None)
    if callable(motor_limits):
        limits = motor_limits(int(target_motor_id))
        torque_limit = getattr(limits, "tmax", None) if limits is not None else None
        if torque_limit is not None and np.isfinite(float(torque_limit)) and float(torque_limit) > 0.0:
            return float(config.compensation.torque_limit_ratio) * abs(float(torque_limit))

    probe_limit = abs(
        _limit_torque_command(
            transport,
            target_motor_id=int(target_motor_id),
            torque=1.0e9,
        )
    )
    if np.isfinite(probe_limit) and probe_limit > 0.0:
        return float(config.compensation.torque_limit_ratio) * float(probe_limit)
    return float("inf")


def _compensation_soft_guard_scale(filtered_velocity: float, config: Config) -> float:
    hard_speed_limit = float(config.safety.hard_speed_abort_abs)
    start_speed = float(config.compensation.soft_abort_start_ratio) * hard_speed_limit
    stop_speed = float(config.compensation.soft_abort_stop_ratio) * hard_speed_limit
    abs_velocity = abs(float(filtered_velocity))
    if abs_velocity <= start_speed:
        return 1.0
    if abs_velocity >= stop_speed:
        return 0.0
    return float((stop_speed - abs_velocity) / max(stop_speed - start_speed, 1.0e-9))


def _compensation_torque(
    parameters: _CompensationParameters,
    *,
    filtered_velocity: float,
    acceleration: float,
    direction: float,
    static_assist_enabled: bool,
    torque_limit_abs: float,
    config: Config,
) -> _CompensationCommand:
    friction_level = _compensation_friction_level(
        parameters,
        filtered_velocity=float(filtered_velocity),
        static_assist_enabled=bool(static_assist_enabled),
        config=config,
    )
    friction_term = (
        float(direction) * float(friction_level)
        + float(parameters.viscous) * float(filtered_velocity)
        + float(parameters.tau_bias)
    )
    inertia_limit_abs = (
        float(config.compensation.max_inertia_torque_ratio) * float(torque_limit_abs)
        if np.isfinite(float(torque_limit_abs))
        else float("inf")
    )
    raw_inertia_term = float(parameters.inertia) * float(acceleration)
    inertia_term = (
        float(np.clip(raw_inertia_term, -inertia_limit_abs, inertia_limit_abs))
        if np.isfinite(float(inertia_limit_abs))
        else float(raw_inertia_term)
    )
    soft_guard_scale = _compensation_soft_guard_scale(float(filtered_velocity), config)
    torque_before_limit = (float(friction_term) + float(inertia_term)) * float(soft_guard_scale)
    hard_guard_scale = 1.0
    limited_torque = float(torque_before_limit)
    if np.isfinite(float(torque_limit_abs)) and float(torque_limit_abs) > 0.0:
        hard_guard_scale = min(1.0, float(torque_limit_abs) / max(abs(float(torque_before_limit)), 1.0e-9))
        limited_torque = float(np.clip(torque_before_limit, -float(torque_limit_abs), float(torque_limit_abs)))
    return _CompensationCommand(
        raw_torque=float(limited_torque),
        direction=float(direction),
        friction_term=float(friction_term),
        inertia_term=float(inertia_term),
        guard_scale=float(soft_guard_scale * hard_guard_scale),
    )


def _compensation_direction(
    *,
    filtered_velocity: float,
    acceleration: float,
    feedback_torque: float,
    feedback_torque_epsilon: float,
    last_direction: float,
    config: Config,
) -> float:
    velocity_epsilon = max(min(float(config.safety.moving_velocity_threshold), 0.05), 1.0e-3)
    acceleration_epsilon = 1.0e-2
    if abs(float(filtered_velocity)) >= velocity_epsilon:
        return float(np.sign(float(filtered_velocity)))
    if abs(float(feedback_torque)) >= feedback_torque_epsilon:
        return float(np.sign(float(feedback_torque)))
    if abs(float(acceleration)) >= acceleration_epsilon:
        return float(np.sign(float(acceleration)))
    return float(np.sign(float(last_direction)))


def _compensation_friction_level(
    parameters: _CompensationParameters,
    *,
    filtered_velocity: float,
    static_assist_enabled: bool,
    config: Config,
) -> float:
    transition_speed = max(float(config.safety.moving_velocity_threshold), 1.0e-3)
    blend = min(abs(float(filtered_velocity)) / transition_speed, 1.0)
    effective_tau_static = min(
        float(parameters.tau_static),
        float(config.compensation.static_assist_ratio_cap) * max(abs(float(parameters.tau_c)), 0.0),
    )
    low_speed_level = effective_tau_static if bool(static_assist_enabled) else float(parameters.tau_c)
    return float(low_speed_level) + (float(parameters.tau_c) - float(low_speed_level)) * float(blend)


def _limit_compensation_slew(*, previous_command: float, desired_command: float, dt_s: float, config: Config) -> float:
    if dt_s <= 0.0:
        return float(desired_command)
    max_delta = float(config.compensation.torque_slew_rate_nm_s) * float(dt_s)
    return float(np.clip(float(desired_command), float(previous_command) - max_delta, float(previous_command) + max_delta))


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
    filtered_velocity: float = float("nan"),
    estimated_acceleration: float = float("nan"),
    friction_term: float = float("nan"),
    inertia_term: float = float("nan"),
    guard_scale: float = float("nan"),
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
            filtered_velocity=float(filtered_velocity),
            estimated_acceleration=float(estimated_acceleration),
            friction_term=float(friction_term),
            inertia_term=float(inertia_term),
            guard_scale=float(guard_scale),
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


def _build_soft_abort_event(
    *,
    config: Config,
    stage: str,
    group_index: int,
    round_index: int,
    phase_name: str,
    target_motor_id: int,
    frame,
) -> AbortEvent | None:
    soft_velocity_limit = float(config.compensation.soft_abort_stop_ratio) * float(config.safety.hard_speed_abort_abs)
    if abs(float(frame.velocity)) < soft_velocity_limit:
        return None
    return AbortEvent(
        reason="soft_speed_abort",
        stage=str(stage),
        motor_id=int(target_motor_id),
        group_index=int(group_index),
        round_index=int(round_index),
        phase_name=str(phase_name),
        observed_velocity=float(frame.velocity),
        velocity_limit=float(soft_velocity_limit),
        detail=(
            f"abs_velocity={abs(float(frame.velocity)):.6f}, "
            f"hard_speed_limit={float(config.safety.hard_speed_abort_abs):.6f}"
        ),
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
    total_frame_count = 0
    target_frame_count = 0
    saw_any_chunk = False
    other_motor_ids: set[int] = set()
    last_target_position: float | None = None
    last_target_velocity: float | None = None
    last_target_torque: float | None = None
    last_target_state: int | None = None

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
        saw_any_chunk = saw_any_chunk or bool(saw_chunk)
        saw_target = False
        for frame in frames:
            total_frame_count += 1
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
                other_motor_ids.add(int(frame.motor_id))
                continue
            saw_target = True
            target_frame_count += 1
            last_target_position = float(frame.position)
            last_target_velocity = float(frame.velocity)
            last_target_torque = float(frame.torque)
            last_target_state = int(frame.state)
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

    if target_frame_count == 0:
        other_motor_text = ",".join(str(motor_id) for motor_id in sorted(other_motor_ids)) or "-"
        raise _RuntimeAbortError(
            AbortEvent(
                reason="feedback_timeout",
                stage=str(stage),
                motor_id=int(target_motor_id),
                group_index=int(group_index),
                round_index=int(round_index),
                phase_name=str(phase_name),
                detail=(
                    f"timeout_s={timeout_s:.3f}, target_feedback_count=0, total_frames={int(total_frame_count)}, "
                    f"other_motor_ids={other_motor_text}, saw_any_chunk={str(bool(saw_any_chunk)).lower()}"
                ),
            )
        )

    stationary_detail_parts = [
        f"timeout_s={timeout_s:.3f}",
        f"target_feedback_count={int(target_frame_count)}",
        f"velocity_threshold={float(config.safety.moving_velocity_threshold):.6f}",
        f"hold_required_s={settle_required_s:.3f}",
    ]
    if last_target_velocity is not None:
        stationary_detail_parts.append(f"last_velocity={float(last_target_velocity):+.6f}")
    if last_target_position is not None:
        stationary_detail_parts.append(f"last_position={float(last_target_position):+.6f}")
    if last_target_torque is not None:
        stationary_detail_parts.append(f"last_torque={float(last_target_torque):+.6f}")
    if last_target_state is not None:
        stationary_detail_parts.append(f"last_state=0x{int(last_target_state):X}")

    raise _RuntimeAbortError(
        AbortEvent(
            reason="stationary_timeout",
            stage=str(stage),
            motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
            phase_name=str(phase_name),
            detail=", ".join(stationary_detail_parts),
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
        # Use active zero-velocity damping here so the motor does not keep coasting
        # between the positive/negative breakaway scans.
        semantic_mode="mit_velocity",
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
    send_interval_s = max(float(config.transport.read_timeout), 5.0e-3)
    feedback_timeout_s = max(float(config.transport.sync_timeout), send_interval_s)
    last_send = 0.0
    last_target_feedback_at = started_at
    torque_limit_abs = _compensation_torque_limit_abs(
        transport,
        target_motor_id=int(target_motor_id),
        config=config,
    )
    command_raw = 0.0
    command = 0.0
    last_direction = 0.0
    pending_direction = 0.0
    direction_hold_count = 0
    acceleration = 0.0
    last_command_sample_time: float | None = None

    while True:
        loop_now = time.monotonic()
        if runtime_limit is not None and (loop_now - started_at) >= runtime_limit:
            break
        if (loop_now - last_send) >= send_interval_s:
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
            last_send = loop_now

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
            last_target_feedback_at = time.monotonic()
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
            soft_abort_event = _build_soft_abort_event(
                config=config,
                stage="compensation",
                group_index=group_index,
                round_index=round_index,
                phase_name=phase_name,
                target_motor_id=target_motor_id,
                frame=frame,
            )
            if soft_abort_event is not None:
                _perform_hard_abort(
                    config=config,
                    transport=transport,
                    parser=parser,
                    target_motor_id=target_motor_id,
                    semantic_mode="mit_torque",
                )
                raise _RuntimeAbortError(soft_abort_event)

            sample_time = float(time.monotonic() - capture_buffer.start_monotonic)
            time_history.append(sample_time)
            velocity_history.append(float(frame.velocity))
            filtered_velocity, acceleration = _compute_compensation_state(
                time_history=time_history,
                velocity_history=velocity_history,
                config=config,
            )
            feedback_torque_epsilon = max(
                0.1 * max(abs(float(parameters.tau_static)), abs(float(parameters.tau_c))),
                1.0e-3,
            )
            current_direction = _compensation_direction(
                filtered_velocity=float(filtered_velocity),
                acceleration=float(acceleration),
                feedback_torque=float(frame.torque),
                feedback_torque_epsilon=float(feedback_torque_epsilon),
                last_direction=float(last_direction),
                config=config,
            )
            if abs(float(current_direction)) > 0.0:
                if np.sign(float(current_direction)) == np.sign(float(pending_direction)):
                    direction_hold_count += 1
                else:
                    pending_direction = float(current_direction)
                    direction_hold_count = 1
                last_direction = float(current_direction)
            static_assist_enabled = bool(
                abs(float(filtered_velocity)) < max(float(config.safety.moving_velocity_threshold), 1.0e-3)
                and direction_hold_count >= int(config.compensation.direction_hold_samples)
                and abs(float(current_direction)) > 0.0
            )
            compensation_command = _compensation_torque(
                parameters,
                filtered_velocity=float(filtered_velocity),
                acceleration=float(acceleration),
                direction=float(current_direction),
                static_assist_enabled=static_assist_enabled,
                torque_limit_abs=float(torque_limit_abs),
                config=config,
            )
            command_raw = float(compensation_command.raw_torque)
            dt_s = 0.0 if last_command_sample_time is None else max(float(sample_time) - float(last_command_sample_time), 0.0)
            command = _limit_compensation_slew(
                previous_command=float(command),
                desired_command=float(command_raw),
                dt_s=float(dt_s),
                config=config,
            )
            last_command_sample_time = float(sample_time)
            command = _limit_torque_command(
                transport,
                target_motor_id=target_motor_id,
                torque=float(command),
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
            last_send = time.monotonic()
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
                filtered_velocity=float(filtered_velocity),
                estimated_acceleration=float(acceleration),
                friction_term=float(compensation_command.friction_term),
                inertia_term=float(compensation_command.inertia_term),
                guard_scale=float(compensation_command.guard_scale),
            )
        if (time.monotonic() - last_target_feedback_at) >= feedback_timeout_s:
            abort_event = AbortEvent(
                reason="feedback_timeout",
                stage="compensation",
                motor_id=int(target_motor_id),
                group_index=int(group_index),
                round_index=int(round_index),
                phase_name=str(phase_name),
                detail=f"timeout_s={feedback_timeout_s:.3f}, last_command={float(command):+.6f}",
            )
            _perform_hard_abort(
                config=config,
                transport=transport,
                parser=parser,
                target_motor_id=target_motor_id,
                semantic_mode="mit_torque",
            )
            raise _RuntimeAbortError(abort_event)
        if not saw_target and not saw_chunk:
            time.sleep(max(float(config.transport.read_timeout), 1.0e-3))


@dataclass(frozen=True)
class _SpeedHoldPlatformStat:
    phase_name: str
    bucket: str
    commanded_velocity: float
    mean_velocity: float
    velocity_std: float
    mean_torque: float
    sample_count: int
    tracking_ratio: float
    velocity_std_ratio: float
    accepted: bool
    rejection_reason: str


def _parse_speed_hold_command_velocity(phase_name: str) -> float:
    try:
        return float(str(phase_name).rsplit("_", 1)[-1])
    except ValueError:
        return float("nan")


def _collect_speed_hold_platform_stats(capture: RoundCapture, config: Config) -> list[_SpeedHoldPlatformStat]:
    phase_names = np.asarray(capture.phase_name).astype(str)
    ordered_phase_names = list(dict.fromkeys(phase_names.tolist()))
    ratio = float(config.mit_velocity.steady_window_ratio)
    platforms: list[_SpeedHoldPlatformStat] = []
    for phase_name in ordered_phase_names:
        if not str(phase_name).startswith("speed_hold_"):
            continue
        if str(phase_name).startswith("speed_hold_train_"):
            bucket = "train"
        elif str(phase_name).startswith("speed_hold_valid_"):
            bucket = "valid"
        else:
            continue

        indices = np.flatnonzero(phase_names == phase_name)
        if indices.size == 0:
            continue
        start_index = int(np.floor((1.0 - ratio) * indices.size))
        selected = indices[start_index:]
        commanded_velocity = _parse_speed_hold_command_velocity(phase_name)
        mean_velocity = float(np.nanmean(capture.velocity[selected])) if selected.size else float("nan")
        velocity_std = float(np.nanstd(capture.velocity[selected])) if selected.size else float("nan")
        mean_torque = float(np.nanmean(capture.torque_feedback[selected])) if selected.size else float("nan")
        tracking_ratio = (
            abs(mean_velocity) / max(abs(commanded_velocity), 1.0e-6)
            if np.isfinite(commanded_velocity)
            else float("nan")
        )
        velocity_std_ratio = (
            velocity_std / max(abs(commanded_velocity), 1.0e-6)
            if np.isfinite(commanded_velocity)
            else float("nan")
        )
        direction_ok = bool(
            np.isfinite(commanded_velocity)
            and np.isfinite(mean_velocity)
            and abs(mean_velocity) > 1.0e-6
            and np.sign(mean_velocity) == np.sign(commanded_velocity)
        )
        accepted = True
        rejection_reason = ""
        if selected.size == 0:
            accepted = False
            rejection_reason = "empty_platform"
        elif not np.isfinite(commanded_velocity):
            accepted = False
            rejection_reason = "invalid_command_velocity"
        elif not np.isfinite(mean_velocity) or not np.isfinite(mean_torque):
            accepted = False
            rejection_reason = "non_finite_platform_statistics"
        elif not direction_ok:
            accepted = False
            rejection_reason = "direction_mismatch"
        elif tracking_ratio < float(config.identification.min_tracking_ratio):
            accepted = False
            rejection_reason = (
                f"tracking_ratio_below_min:{tracking_ratio:.3f}"
                f"<{float(config.identification.min_tracking_ratio):.3f}"
            )
        elif velocity_std_ratio > float(config.identification.max_steady_velocity_std_ratio):
            accepted = False
            rejection_reason = (
                f"velocity_std_ratio_above_max:{velocity_std_ratio:.3f}"
                f">{float(config.identification.max_steady_velocity_std_ratio):.3f}"
            )

        platforms.append(
            _SpeedHoldPlatformStat(
                phase_name=str(phase_name),
                bucket=str(bucket),
                commanded_velocity=float(commanded_velocity),
                mean_velocity=float(mean_velocity),
                velocity_std=float(velocity_std),
                mean_torque=float(mean_torque),
                sample_count=int(selected.size),
                tracking_ratio=float(tracking_ratio),
                velocity_std_ratio=float(velocity_std_ratio),
                accepted=bool(accepted),
                rejection_reason=str(rejection_reason),
            )
        )
    return platforms


def _annotate_friction_result(
    result: FrictionIdentificationResult,
    *,
    platforms: list[_SpeedHoldPlatformStat],
) -> FrictionIdentificationResult:
    metadata = dict(result.metadata)
    metadata["platforms"] = [
        {
            "phase_name": platform.phase_name,
            "bucket": platform.bucket,
            "commanded_velocity": float(platform.commanded_velocity),
            "mean_velocity": float(platform.mean_velocity),
            "velocity_std": float(platform.velocity_std),
            "mean_torque": float(platform.mean_torque),
            "sample_count": int(platform.sample_count),
            "tracking_ratio": float(platform.tracking_ratio),
            "velocity_std_ratio": float(platform.velocity_std_ratio),
            "accepted": bool(platform.accepted),
            "rejection_reason": str(platform.rejection_reason),
        }
        for platform in platforms
    ]
    metadata["accepted_train_platform_count"] = int(
        sum(1 for platform in platforms if platform.accepted and platform.bucket == "train")
    )
    metadata["accepted_valid_platform_count"] = int(
        sum(1 for platform in platforms if platform.accepted and platform.bucket == "valid")
    )
    metadata["rejected_platform_count"] = int(sum(1 for platform in platforms if not platform.accepted))
    return FrictionIdentificationResult(
        tau_c=float(result.tau_c),
        viscous=float(result.viscous),
        tau_bias=float(result.tau_bias),
        train_rmse=float(result.train_rmse),
        valid_rmse=float(result.valid_rmse),
        train_mask=np.asarray(result.train_mask, dtype=bool),
        valid_mask=np.asarray(result.valid_mask, dtype=bool),
        torque_pred=np.asarray(result.torque_pred, dtype=np.float64),
        torque_target=np.asarray(result.torque_target, dtype=np.float64),
        metadata=metadata,
    )


def _build_round_validation_result(
    friction_result: FrictionIdentificationResult,
    inertia_result: InertiaIdentificationResult,
) -> ValidationResult:
    accepted_train_platform_count = int(friction_result.metadata.get("accepted_train_platform_count", 0))
    accepted_valid_platform_count = int(friction_result.metadata.get("accepted_valid_platform_count", 0))
    rejected_platform_count = int(friction_result.metadata.get("rejected_platform_count", 0))
    friction_rmse = float(friction_result.valid_rmse)
    inertia_rmse = float(inertia_result.valid_rmse)
    reasons: list[str] = []

    if accepted_train_platform_count < 2:
        reasons.append(f"accepted_train_platform_count={accepted_train_platform_count}<2")
    if accepted_valid_platform_count < 2:
        reasons.append(f"accepted_valid_platform_count={accepted_valid_platform_count}<2")
    if not np.isfinite(float(friction_result.tau_c)) or float(friction_result.tau_c) < 0.0:
        reasons.append(f"invalid_tau_c={float(friction_result.tau_c):+.6f}")
    if not np.isfinite(float(friction_result.viscous)) or float(friction_result.viscous) < 0.0:
        reasons.append(f"invalid_viscous={float(friction_result.viscous):+.6f}")
    if not np.isfinite(float(inertia_result.inertia)) or float(inertia_result.inertia) < 0.0:
        reasons.append(f"invalid_inertia={float(inertia_result.inertia):+.6f}")
    if not np.isfinite(friction_rmse) or friction_rmse > 0.15:
        reasons.append(f"friction_rmse={friction_rmse:.6f}>0.150000")
    if not np.isfinite(inertia_rmse) or inertia_rmse > 0.20:
        reasons.append(f"inertia_rmse={inertia_rmse:.6f}>0.200000")

    recommended = not reasons
    status = "accepted" if recommended else "rejected"
    detail = (
        f"friction_rmse={friction_rmse:.6f}, inertia_rmse={inertia_rmse:.6f}"
        if recommended
        else "; ".join(reasons)
    )
    return ValidationResult(
        friction_rmse=friction_rmse,
        inertia_rmse=inertia_rmse,
        recommended_for_compensation=bool(recommended),
        detail=detail,
        metadata={
            "status": status,
            "accepted_train_platform_count": accepted_train_platform_count,
            "accepted_valid_platform_count": accepted_valid_platform_count,
            "rejected_platform_count": rejected_platform_count,
            "reasons": list(reasons),
        },
    )


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
    speed_hold_platforms = _collect_speed_hold_platform_stats(capture, config)

    if mode in {"identify-all", "speed-hold", "inertia"}:
        platform_velocity = np.asarray([platform.mean_velocity for platform in speed_hold_platforms], dtype=np.float64)
        platform_torque = np.asarray([platform.mean_torque for platform in speed_hold_platforms], dtype=np.float64)
        friction_train_mask = np.asarray(
            [platform.accepted and platform.bucket == "train" for platform in speed_hold_platforms],
            dtype=bool,
        )
        friction_valid_mask = np.asarray(
            [platform.accepted and platform.bucket == "valid" for platform in speed_hold_platforms],
            dtype=bool,
        )
        friction_result = fit_friction_model(
            platform_velocity,
            platform_torque,
            train_mask=friction_train_mask,
            valid_mask=friction_valid_mask,
        )
        friction_result = _annotate_friction_result(friction_result, platforms=speed_hold_platforms)

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
        validation_result = _build_round_validation_result(friction_result, inertia_result)
    elif mode == "speed-hold":
        validation_result = ValidationResult(
            friction_rmse=float(friction_result.valid_rmse),
            inertia_rmse=float("nan"),
            recommended_for_compensation=False,
            detail="speed-hold debug mode",
            metadata={
                "status": "partial",
                "accepted_train_platform_count": int(friction_result.metadata.get("accepted_train_platform_count", 0)),
                "accepted_valid_platform_count": int(friction_result.metadata.get("accepted_valid_platform_count", 0)),
                "rejected_platform_count": int(friction_result.metadata.get("rejected_platform_count", 0)),
            },
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
            # Use active zero-velocity damping during precheck so lightly damped
            # motors settle before the identification sequence starts.
            semantic_mode="mit_velocity",
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

        if mode == "identify-all":
            store.save_latest_parameters(artifacts)
        summary_paths = store.save_summary(artifacts)
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
    if str(parameters.publish_status) != "published":
        log_info(
            f"Warning: motor_id={int(target_motor_id)} using unpublished model because "
            f"compensation.require_published_model={str(bool(config.compensation.require_published_model)).lower()}, "
            f"publish_status={parameters.publish_status}"
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
        log_info(
            f"Starting compensation round 1/1: motor_id={int(target_motor_id)}, "
            f"source_run_label={parameters.source_run_label}, publish_status={parameters.publish_status}"
        )
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
                "model_kind": str(parameters.model_kind),
                "publish_status": str(parameters.publish_status),
                "publish_detail": str(parameters.publish_detail),
                "accepted_round_count": int(parameters.accepted_round_count),
                "selected_rounds": list(parameters.selected_rounds),
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
