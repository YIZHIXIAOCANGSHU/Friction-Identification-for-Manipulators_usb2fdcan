from __future__ import annotations

from collections import deque
import time
from dataclasses import dataclass

import numpy as np

from friction_identification_core.capture import CaptureBuffer, log_stage_transition, poll_feedback_frames, record_target_frame, send_command
from friction_identification_core.core import AbortEvent, IdentifiedMotorModel, piecewise_static_linear_level, piecewise_static_linear_torque
from friction_identification_core.identification import estimate_filtered_velocity_and_acceleration
from friction_identification_core.io import CommandTransport, FeedbackFrameParser
from friction_identification_core.results import latest_parameters_path, load_latest_parameters
from friction_identification_core.runtime_config import Config
from friction_identification_core.safety import RuntimeAbortError, build_abort_event, build_soft_abort_event, perform_hard_abort
from friction_identification_core.visualization import RerunRecorder


@dataclass(frozen=True)
class CompensationParameters:
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
class CompensationCommand:
    raw_torque: float
    direction: float
    friction_term: float
    inertia_term: float
    guard_scale: float


def load_compensation_parameters(config: Config, *, target_motor_id: int) -> CompensationParameters:
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

    model = IdentifiedMotorModel.from_latest_entry(entry)
    if bool(config.compensation.require_published_model) and model.publish_status != "published":
        previous = entry.get("previous_published_model")
        previous_detail = ""
        if isinstance(previous, dict):
            previous_source = previous.get("source_run_label", "unknown")
            previous_detail = f", previous_published_source_run_label={previous_source}"
        raise ValueError(
            f"compensation requires a published model for motor_id={int(target_motor_id)}: "
            f"publish_status={model.publish_status}, source_run_label={model.source_run_label}"
            f"{previous_detail}, latest_parameters_path={latest_path}"
        )

    return CompensationParameters(
        motor_id=int(model.motor_id),
        motor_name=str(model.motor_name),
        identified_at=str(model.identified_at),
        source_run_label=str(model.source_run_label),
        model_kind=str(model.model_kind),
        publish_status=str(model.publish_status),
        publish_detail=str(model.publish_detail),
        accepted_round_count=int(model.accepted_round_count),
        selected_rounds=tuple(model.selected_rounds),
        tau_static=float(model.tau_static),
        tau_bias=float(model.tau_bias),
        tau_c=float(model.tau_c),
        viscous=float(model.viscous),
        inertia=float(model.inertia),
        friction_validation_rmse=float(model.friction_validation_rmse),
        inertia_validation_rmse=float(model.inertia_validation_rmse),
        repeat_consistency_score=float(model.repeat_consistency_score),
        recommended_for_compensation=bool(model.recommended_for_compensation),
    )


def limit_torque_command(transport: CommandTransport, *, target_motor_id: int, torque: float) -> float:
    limiter = getattr(transport, "limit_torque_command", None)
    if callable(limiter):
        return float(limiter(int(target_motor_id), float(torque)))
    return float(torque)


def compensation_history_window(config: Config) -> int:
    window = max(
        int(config.identification.savgol_window),
        int(config.identification.savgol_polyorder) + 2,
        3,
    )
    if window % 2 == 0:
        window += 1
    return window


def compute_compensation_state(
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


def compensation_torque_limit_abs(
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
        limit_torque_command(
            transport,
            target_motor_id=int(target_motor_id),
            torque=1.0e9,
        )
    )
    if np.isfinite(probe_limit) and probe_limit > 0.0:
        return float(config.compensation.torque_limit_ratio) * float(probe_limit)
    return float("inf")


def compensation_soft_guard_scale(filtered_velocity: float, config: Config) -> float:
    hard_speed_limit = float(config.safety.hard_speed_abort_abs)
    start_speed = float(config.compensation.soft_abort_start_ratio) * hard_speed_limit
    stop_speed = float(config.compensation.soft_abort_stop_ratio) * hard_speed_limit
    abs_velocity = abs(float(filtered_velocity))
    if abs_velocity <= start_speed:
        return 1.0
    if abs_velocity >= stop_speed:
        return 0.0
    return float((stop_speed - abs_velocity) / max(stop_speed - start_speed, 1.0e-9))


def compensation_friction_level(
    parameters: CompensationParameters,
    *,
    filtered_velocity: float,
    config: Config,
) -> float:
    return float(
        piecewise_static_linear_level(
            abs(float(filtered_velocity)),
            tau_static=float(parameters.tau_static),
            tau_c=float(parameters.tau_c),
            static_velocity_threshold_rad_s=float(config.compensation.static_velocity_threshold_rad_s),
            static_transition_velocity_rad_s=float(config.compensation.static_transition_velocity_rad_s),
        )
    )


def compensation_torque(
    parameters: CompensationParameters,
    *,
    filtered_velocity: float,
    acceleration: float,
    direction: float,
    torque_limit_abs: float,
    config: Config,
) -> CompensationCommand:
    friction_level = compensation_friction_level(
        parameters,
        filtered_velocity=float(filtered_velocity),
        config=config,
    )
    model_torque = float(
        piecewise_static_linear_torque(
            float(filtered_velocity),
            acceleration=float(acceleration),
            direction=float(direction),
            tau_static=float(parameters.tau_static),
            tau_c=float(parameters.tau_c),
            viscous=float(parameters.viscous),
            tau_bias=float(parameters.tau_bias),
            inertia=float(parameters.inertia),
            static_velocity_threshold_rad_s=float(config.compensation.static_velocity_threshold_rad_s),
            static_transition_velocity_rad_s=float(config.compensation.static_transition_velocity_rad_s),
        )
    )
    inertia_term = float(parameters.inertia) * float(acceleration)
    friction_term = float(model_torque - inertia_term)
    soft_guard_scale = compensation_soft_guard_scale(float(filtered_velocity), config)
    torque_before_limit = float(model_torque) * float(soft_guard_scale)
    hard_guard_scale = 1.0
    limited_torque = float(torque_before_limit)
    if np.isfinite(float(torque_limit_abs)) and float(torque_limit_abs) > 0.0:
        hard_guard_scale = min(1.0, float(torque_limit_abs) / max(abs(float(torque_before_limit)), 1.0e-9))
        limited_torque = float(np.clip(torque_before_limit, -float(torque_limit_abs), float(torque_limit_abs)))
    return CompensationCommand(
        raw_torque=float(limited_torque),
        direction=float(direction),
        friction_term=float(friction_term),
        inertia_term=float(inertia_term),
        guard_scale=float(soft_guard_scale * hard_guard_scale),
    )


def compensation_direction(
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


def limit_compensation_slew(*, previous_command: float, desired_command: float, dt_s: float, config: Config) -> float:
    if dt_s <= 0.0:
        return float(desired_command)
    max_delta = float(config.compensation.torque_slew_rate_nm_s) * float(dt_s)
    return float(np.clip(float(desired_command), float(previous_command) - max_delta, float(previous_command) + max_delta))


def run_compensation_phase(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    capture_buffer: CaptureBuffer,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    parameters: CompensationParameters,
    max_runtime_s: float | None,
) -> None:
    log_stage_transition(
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
    history_window = compensation_history_window(config)
    time_history: deque[float] = deque(maxlen=history_window)
    velocity_history: deque[float] = deque(maxlen=history_window)
    send_interval_s = max(float(config.transport.read_timeout), 5.0e-3)
    feedback_timeout_s = max(float(config.transport.sync_timeout), send_interval_s)
    last_send = 0.0
    last_target_feedback_at = started_at
    torque_limit_abs = compensation_torque_limit_abs(
        transport,
        target_motor_id=int(target_motor_id),
        config=config,
    )
    command_raw = 0.0
    command = 0.0
    last_direction = 0.0
    pending_direction = 0.0
    direction_hold_count = 0
    last_command_sample_time: float | None = None

    while True:
        loop_now = time.monotonic()
        if runtime_limit is not None and (loop_now - started_at) >= runtime_limit:
            break
        if (loop_now - last_send) >= send_interval_s:
            send_command(
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

        frames, saw_chunk = poll_feedback_frames(
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
            abort_event = build_abort_event(
                config=config,
                stage="compensation",
                group_index=group_index,
                round_index=round_index,
                phase_name=phase_name,
                target_motor_id=target_motor_id,
                frame=frame,
            )
            if abort_event is not None:
                perform_hard_abort(
                    config=config,
                    transport=transport,
                    parser=parser,
                    target_motor_id=target_motor_id,
                    semantic_mode="mit_torque",
                )
                raise RuntimeAbortError(abort_event)
            soft_abort_event = build_soft_abort_event(
                config=config,
                stage="compensation",
                group_index=group_index,
                round_index=round_index,
                phase_name=phase_name,
                target_motor_id=target_motor_id,
                frame=frame,
            )
            if soft_abort_event is not None:
                perform_hard_abort(
                    config=config,
                    transport=transport,
                    parser=parser,
                    target_motor_id=target_motor_id,
                    semantic_mode="mit_torque",
                )
                raise RuntimeAbortError(soft_abort_event)

            sample_time = float(time.monotonic() - capture_buffer.start_monotonic)
            time_history.append(sample_time)
            velocity_history.append(float(frame.velocity))
            filtered_velocity, acceleration = compute_compensation_state(
                time_history=time_history,
                velocity_history=velocity_history,
                config=config,
            )
            feedback_torque_epsilon = max(
                0.1 * max(abs(float(parameters.tau_static)), abs(float(parameters.tau_c))),
                1.0e-3,
            )
            current_direction = compensation_direction(
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
            _ = direction_hold_count
            compensation_command = compensation_torque(
                parameters,
                filtered_velocity=float(filtered_velocity),
                acceleration=float(acceleration),
                direction=float(current_direction),
                torque_limit_abs=float(torque_limit_abs),
                config=config,
            )
            command_raw = float(compensation_command.raw_torque)
            dt_s = 0.0 if last_command_sample_time is None else max(float(sample_time) - float(last_command_sample_time), 0.0)
            command = limit_compensation_slew(
                previous_command=float(command),
                desired_command=float(command_raw),
                dt_s=float(dt_s),
                config=config,
            )
            last_command_sample_time = float(sample_time)
            command = limit_torque_command(
                transport,
                target_motor_id=target_motor_id,
                torque=float(command),
            )
            send_command(
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
            record_target_frame(
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
                torque_ff_cmd=float(command),
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
            perform_hard_abort(
                config=config,
                transport=transport,
                parser=parser,
                target_motor_id=target_motor_id,
                semantic_mode="mit_torque",
            )
            raise RuntimeAbortError(abort_event)
        if not saw_target and not saw_chunk:
            time.sleep(max(float(config.transport.read_timeout), 1.0e-3))


__all__ = [
    "CompensationCommand",
    "CompensationParameters",
    "load_compensation_parameters",
    "run_compensation_phase",
]
