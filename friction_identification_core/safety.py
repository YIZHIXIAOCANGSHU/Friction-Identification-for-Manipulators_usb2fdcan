from __future__ import annotations

import time

import numpy as np

from friction_identification_core.capture import (
    CaptureBuffer,
    log_stage_transition,
    poll_feedback_frames,
    record_target_frame,
    send_command,
)
from friction_identification_core.core import AbortEvent
from friction_identification_core.io import CommandTransport, FeedbackFrameParser, SemanticMode
from friction_identification_core.runtime_config import Config
from friction_identification_core.visualization import RerunRecorder


ABORT_ZERO_COMMAND_REPEAT = 5


class RuntimeAbortError(RuntimeError):
    def __init__(self, event: AbortEvent) -> None:
        self.event = event
        super().__init__(event.error_message())


def build_abort_event(
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


def build_soft_abort_event(
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


def perform_hard_abort(
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
        frames, saw_chunk = poll_feedback_frames(
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


def wait_for_stationary(
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
    capture_buffer: CaptureBuffer | None = None,
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
            send_command(
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

        frames, saw_chunk = poll_feedback_frames(
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
            abort_event = build_abort_event(
                config=config,
                stage=stage,
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
                    semantic_mode=semantic_mode,
                )
                raise RuntimeAbortError(abort_event)

            record_target_frame(
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
                kd_cmd=float(getattr(config.mit_velocity, "kd_speed")[target_index]) if semantic_mode == "mit_velocity" else 0.0,
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
        raise RuntimeAbortError(
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

    raise RuntimeAbortError(
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


def send_zero_then_disable(
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


def precheck_transport(
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
        log_stage_transition("precheck", target_motor_id=int(motor_id), detail=description)
        transport.clear_error(int(motor_id))
        transport.enable_motor(int(motor_id))
        wait_for_stationary(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            target_motor_id=int(motor_id),
            group_index=0,
            round_index=0,
            phase_name="precheck_zero",
            stage="precheck",
            semantic_mode="mit_velocity",
            capture_buffer=None,
        )


__all__ = [
    "ABORT_ZERO_COMMAND_REPEAT",
    "RuntimeAbortError",
    "build_abort_event",
    "build_soft_abort_event",
    "perform_hard_abort",
    "precheck_transport",
    "send_zero_then_disable",
    "wait_for_stationary",
]
