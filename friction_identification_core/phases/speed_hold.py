from __future__ import annotations

import time

import numpy as np

from friction_identification_core.capture import CaptureBuffer, log_stage_transition, poll_feedback_frames, record_target_frame, send_command
from friction_identification_core.io import CommandTransport, FeedbackFrameParser, SemanticMode
from friction_identification_core.limits import identification_limits_for_motor, validate_abs_less_than
from friction_identification_core.runtime_config import Config
from friction_identification_core.safety import RuntimeAbortError, build_abort_event, perform_hard_abort, wait_for_stationary
from friction_identification_core.visualization import RerunRecorder


def run_velocity_segment(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    capture_buffer: CaptureBuffer,
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
    limits = identification_limits_for_motor(config, target_motor_id=int(target_motor_id))
    validate_abs_less_than(
        (float(start_velocity), float(end_velocity)),
        limit_abs=float(limits.identification_speed_abs),
        name=f"{stage}.{phase_name}.velocity_cmd",
    )
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
            send_command(
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
                phase_name=str(phase_name),
                stage=str(stage),
            )
            if int(frame.motor_id) != int(target_motor_id):
                continue
            saw_target = True
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
                command_raw=float(current_velocity_cmd),
                command=float(current_velocity_cmd),
                position_cmd=0.0,
                velocity_cmd=float(current_velocity_cmd),
                acceleration_cmd=float(acceleration_cmd),
                phase_name=phase_name,
                stage=stage,
                kd_cmd=float(kd_speed) if semantic_mode == "mit_velocity" else 0.0,
            )

        if duration_s <= 0.0 or elapsed >= duration_s:
            break
        if not saw_target and not saw_chunk:
            time.sleep(max(float(config.transport.read_timeout), 1.0e-3))

    return float(current_velocity_cmd)


def run_speed_hold_phase(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    capture_buffer: CaptureBuffer,
    target_motor_id: int,
    group_index: int,
    round_index: int,
) -> None:
    log_stage_transition("speed-hold", target_motor_id=target_motor_id)
    rerun_recorder.log_phase_event(motor_id=int(target_motor_id), phase_name="speed-hold", detail="start")
    target_index = config.motor_index(target_motor_id)
    kd_speed = float(config.mit_velocity.kd_speed[target_index])
    ramp_acceleration = float(config.mit_velocity.ramp_acceleration)
    hold_duration = float(config.mit_velocity.steady_hold_duration)
    limits = identification_limits_for_motor(config, target_motor_id=int(target_motor_id))
    validate_abs_less_than(
        config.identification.steady_speed_points,
        limit_abs=float(limits.identification_speed_abs),
        name="identification.steady_speed_points",
    )
    holdout_speed = max(float(item) for item in config.identification.steady_speed_points)
    current_velocity = 0.0

    speed_points: list[float] = [float(point) for point in config.identification.steady_speed_points]
    speed_points.extend([-float(point) for point in config.identification.steady_speed_points])
    for target_velocity in speed_points:
        bucket = "valid" if np.isclose(abs(float(target_velocity)), holdout_speed) else "train"
        ramp_duration = abs(float(target_velocity) - float(current_velocity)) / ramp_acceleration
        current_velocity = run_velocity_segment(
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
        current_velocity = run_velocity_segment(
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

    current_velocity = run_velocity_segment(
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
    wait_for_stationary(
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


__all__ = ["run_speed_hold_phase", "run_velocity_segment"]
