from __future__ import annotations

import time

import numpy as np

from friction_identification_core.capture import CaptureBuffer, log_stage_transition, poll_feedback_frames, record_target_frame, send_command
from friction_identification_core.core import AbortEvent
from friction_identification_core.io import CommandTransport, FeedbackFrameParser
from friction_identification_core.limits import identification_limits_for_motor, validate_abs_less_than
from friction_identification_core.runtime_config import Config
from friction_identification_core.safety import RuntimeAbortError, build_abort_event, perform_hard_abort, wait_for_stationary
from friction_identification_core.visualization import RerunRecorder


def _low_speed_tracking_ok(commanded_velocity: float, measured_velocity: float, config: Config) -> bool:
    command_abs = abs(float(commanded_velocity))
    if command_abs < 1.0e-6:
        return True
    measured_abs = abs(float(measured_velocity))
    direction_ok = measured_abs < 1.0e-6 or np.sign(float(measured_velocity)) == np.sign(float(commanded_velocity))
    tracking_ratio = measured_abs / max(command_abs, 1.0e-9)
    return bool(direction_ok and tracking_ratio >= float(config.identification.min_tracking_ratio))


def _low_speed_stiction_evidence(commanded_velocity: float, measured_velocity: float, tracking_ok: bool, config: Config) -> bool:
    if abs(float(commanded_velocity)) < 1.0e-6:
        return False
    if bool(tracking_ok):
        return False
    return bool(abs(float(measured_velocity)) < max(float(config.safety.moving_velocity_threshold), abs(float(commanded_velocity))))


def _low_speed_settle_timeout_s(config: Config) -> float:
    return max(
        float(config.transport.sync_timeout),
        float(config.safety.moving_hold_ms) / 1000.0 + max(0.05, 4.0 * float(config.transport.read_timeout)),
    )


def _run_low_speed_velocity_segment(
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
    start_velocity: float,
    end_velocity: float,
    duration_s: float,
    kd_speed: float,
) -> float:
    limits = identification_limits_for_motor(config, target_motor_id=int(target_motor_id))
    validate_abs_less_than(
        (float(start_velocity), float(end_velocity)),
        limit_abs=float(limits.identification_speed_abs),
        name=f"low_speed.{phase_name}.velocity_cmd",
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
                semantic_mode="mit_velocity",
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
                stage="low-speed",
            )
            if int(frame.motor_id) != int(target_motor_id):
                continue
            saw_target = True
            abort_event = build_abort_event(
                config=config,
                stage="low-speed",
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
                    semantic_mode="mit_velocity",
                )
                raise RuntimeAbortError(abort_event)
            tracking_ok = _low_speed_tracking_ok(float(current_velocity_cmd), float(frame.velocity), config)
            stiction_evidence = _low_speed_stiction_evidence(
                float(current_velocity_cmd),
                float(frame.velocity),
                tracking_ok,
                config,
            )
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
                stage="low-speed",
                kd_cmd=float(kd_speed),
                tracking_ok=tracking_ok,
                stiction_evidence=stiction_evidence,
                used_for_fit=bool(tracking_ok and not stiction_evidence and np.isfinite(float(frame.torque))),
            )

        if duration_s <= 0.0 or elapsed >= duration_s:
            break
        if not saw_target and not saw_chunk:
            time.sleep(max(float(config.transport.read_timeout), 1.0e-3))

    return float(current_velocity_cmd)


def _run_low_speed_micro_motion(
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
    duration_s = float(config.low_speed.micro_motion_record_duration)
    if not bool(config.low_speed.micro_motion_enabled) or duration_s <= 0.0:
        return

    target_index = config.motor_index(target_motor_id)
    send_interval_s = max(float(config.transport.read_timeout), 5.0e-3)
    frequency = float(config.low_speed.micro_motion_frequency_hz)
    velocity_limit = float(config.low_speed.micro_motion_velocity_limit)
    micro_motion_kd = float(config.low_speed.micro_motion_kd)
    limits = identification_limits_for_motor(config, target_motor_id=int(target_motor_id))
    validate_abs_less_than(
        (velocity_limit,),
        limit_abs=float(limits.identification_speed_abs),
        name="low_speed.micro_motion_velocity_limit",
    )
    omega = 2.0 * np.pi * frequency
    start_monotonic = time.monotonic()
    last_send = 0.0

    while True:
        now = time.monotonic()
        elapsed = now - start_monotonic
        if elapsed >= duration_s:
            break
        velocity_cmd = velocity_limit * np.sin(omega * elapsed)
        acceleration_cmd = velocity_limit * omega * np.cos(omega * elapsed)
        bucket = "train" if (elapsed / max(duration_s, 1.0e-9)) < float(config.low_speed.train_valid_split) else "valid"
        phase_name = f"low_speed_micro_{bucket}"

        if (now - last_send) >= send_interval_s:
            send_command(
                config=config,
                transport=transport,
                rerun_recorder=rerun_recorder,
                target_motor_id=int(target_motor_id),
                target_index=target_index,
                semantic_mode="mit_velocity",
                command_value=float(velocity_cmd),
                kd_speed=float(micro_motion_kd),
                position_cmd=0.0,
                velocity_cmd=float(velocity_cmd),
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
                stage="low-speed",
            )
            if int(frame.motor_id) != int(target_motor_id):
                continue
            saw_target = True
            abort_event = build_abort_event(
                config=config,
                stage="low-speed",
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
                    semantic_mode="mit_velocity",
                )
                raise RuntimeAbortError(abort_event)
            tracking_ok = _low_speed_tracking_ok(float(velocity_cmd), float(frame.velocity), config)
            stiction_evidence = _low_speed_stiction_evidence(float(velocity_cmd), float(frame.velocity), tracking_ok, config)
            record_target_frame(
                config=config,
                rerun_recorder=rerun_recorder,
                capture_buffer=capture_buffer,
                group_index=group_index,
                round_index=round_index,
                target_motor_id=target_motor_id,
                frame=frame,
                command_raw=float(velocity_cmd),
                command=float(velocity_cmd),
                position_cmd=0.0,
                velocity_cmd=float(velocity_cmd),
                acceleration_cmd=float(acceleration_cmd),
                phase_name=phase_name,
                stage="low-speed",
                kp_cmd=0.0,
                kd_cmd=float(micro_motion_kd),
                torque_ff_cmd=0.0,
                tracking_ok=tracking_ok,
                stiction_evidence=stiction_evidence,
                used_for_fit=bool(tracking_ok and not stiction_evidence and np.isfinite(float(frame.torque))),
            )

        if not saw_target and not saw_chunk:
            time.sleep(max(float(config.transport.read_timeout), 1.0e-3))


def run_low_speed_characterization_phase(
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
    if not bool(config.low_speed.enabled):
        return
    log_stage_transition("low-speed", target_motor_id=target_motor_id)
    rerun_recorder.log_phase_event(motor_id=int(target_motor_id), phase_name="low-speed", detail="start")
    target_index = config.motor_index(target_motor_id)
    kd_speed = float(config.mit_velocity.kd_speed[target_index])
    ramp_acceleration = float(config.low_speed.ramp_acceleration)
    hold_duration = float(config.low_speed.hold_duration)
    speed_points = [float(point) for point in config.low_speed.speed_points]
    limits = identification_limits_for_motor(config, target_motor_id=int(target_motor_id))
    validate_abs_less_than(
        speed_points,
        limit_abs=float(limits.identification_speed_abs),
        name="low_speed.speed_points",
    )
    current_velocity = 0.0

    for index, speed in enumerate(speed_points):
        bucket = "train" if (index / max(len(speed_points), 1)) < float(config.low_speed.train_valid_split) else "valid"
        for target_velocity in (float(speed), -float(speed)):
            ramp_duration = abs(float(target_velocity) - float(current_velocity)) / ramp_acceleration
            current_velocity = _run_low_speed_velocity_segment(
                config=config,
                transport=transport,
                parser=parser,
                rerun_recorder=rerun_recorder,
                capture_buffer=capture_buffer,
                target_motor_id=target_motor_id,
                group_index=group_index,
                round_index=round_index,
                phase_name=f"low_speed_ramp_{bucket}_{float(target_velocity):+0.2f}",
                start_velocity=float(current_velocity),
                end_velocity=float(target_velocity),
                duration_s=float(ramp_duration),
                kd_speed=kd_speed,
            )
            current_velocity = _run_low_speed_velocity_segment(
                config=config,
                transport=transport,
                parser=parser,
                rerun_recorder=rerun_recorder,
                capture_buffer=capture_buffer,
                target_motor_id=target_motor_id,
                group_index=group_index,
                round_index=round_index,
                phase_name=f"low_speed_hold_{bucket}_{float(target_velocity):+0.2f}",
                start_velocity=float(current_velocity),
                end_velocity=float(target_velocity),
                duration_s=hold_duration,
                kd_speed=kd_speed,
            )

    settle_timeout_s = _low_speed_settle_timeout_s(config)
    if bool(config.low_speed.micro_motion_enabled) and float(config.low_speed.micro_motion_record_duration) > 0.0:
        if abs(float(current_velocity)) > 1.0e-9:
            current_velocity = _run_low_speed_velocity_segment(
                config=config,
                transport=transport,
                parser=parser,
                rerun_recorder=rerun_recorder,
                capture_buffer=capture_buffer,
                target_motor_id=target_motor_id,
                group_index=group_index,
                round_index=round_index,
                phase_name="low_speed_ramp_to_zero_before_micro",
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
            phase_name="low_speed_pre_micro_settle",
            stage="low-speed",
            semantic_mode="mit_velocity",
            capture_buffer=capture_buffer,
            timeout_s=settle_timeout_s,
        )
        _run_low_speed_micro_motion(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=target_motor_id,
            group_index=group_index,
            round_index=round_index,
        )

    wait_for_stationary(
        config=config,
        transport=transport,
        parser=parser,
        rerun_recorder=rerun_recorder,
        target_motor_id=target_motor_id,
        group_index=group_index,
        round_index=round_index,
        phase_name="low_speed_settle",
        stage="low-speed",
        semantic_mode="mit_velocity",
        capture_buffer=capture_buffer,
        timeout_s=settle_timeout_s,
    )


__all__ = ["run_low_speed_characterization_phase"]
