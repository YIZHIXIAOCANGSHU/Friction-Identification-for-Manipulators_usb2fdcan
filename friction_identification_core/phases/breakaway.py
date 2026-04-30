from __future__ import annotations

import time

import numpy as np

from friction_identification_core.capture import CaptureBuffer, log_stage_transition, poll_feedback_frames, record_target_frame, send_command
from friction_identification_core.core import BreakawayIdentificationResult
from friction_identification_core.io import CommandTransport, FeedbackFrameParser
from friction_identification_core.limits import identification_limits_for_motor
from friction_identification_core.runtime_config import Config
from friction_identification_core.safety import RuntimeAbortError, build_abort_event, perform_hard_abort, wait_for_stationary
from friction_identification_core.visualization import RerunRecorder


def breakaway_torque_scan_values(*, torque_step: float, scan_limit: float) -> np.ndarray:
    step = float(torque_step)
    limit = float(scan_limit)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("breakaway.torque_step must be > 0.")
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("breakaway.scan_max_torque must be > 0.")
    values: list[float] = []
    current = step
    while current < limit:
        values.append(float(current))
        current += step
    if not values or abs(float(values[-1]) - limit) > 1.0e-9:
        values.append(limit)
    return np.asarray(values, dtype=np.float64)


def scan_breakaway_direction(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    capture_buffer: CaptureBuffer,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    direction: int,
) -> float:
    direction_label = "pos" if int(direction) > 0 else "neg"
    target_index = config.motor_index(target_motor_id)
    scan_limit = float(config.breakaway.scan_max_torque[target_index])
    limits = identification_limits_for_motor(config, target_motor_id=int(target_motor_id))
    if scan_limit > float(limits.motor_tmax) + 1.0e-9:
        raise ValueError(
            "breakaway.scan_max_torque must be <= motor torque limit for "
            f"motor_id={int(target_motor_id)} ({float(limits.motor_tmax):.6f} Nm)."
        )
    send_interval_s = max(float(config.transport.read_timeout), 5.0e-3)
    moving_hold_s = float(config.safety.moving_hold_ms) / 1000.0
    torque_step = float(config.breakaway.torque_step)
    hold_duration = float(config.breakaway.hold_duration)
    torque_values = breakaway_torque_scan_values(torque_step=torque_step, scan_limit=scan_limit)

    wait_for_stationary(
        config=config,
        transport=transport,
        parser=parser,
        rerun_recorder=rerun_recorder,
        target_motor_id=target_motor_id,
        group_index=group_index,
        round_index=round_index,
        phase_name=f"breakaway_{direction_label}_settle",
        stage="breakaway",
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
                send_command(
                    config=config,
                    transport=transport,
                    rerun_recorder=rerun_recorder,
                    target_motor_id=int(target_motor_id),
                    target_index=target_index,
                    semantic_mode="mit_torque",
                    command_value=float(command_value),
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
                    stage="breakaway",
                )
                if int(frame.motor_id) != int(target_motor_id):
                    continue
                saw_target = True
                abort_event = build_abort_event(
                    config=config,
                    stage="breakaway",
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
                record_target_frame(
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
                    torque_ff_cmd=float(command_value),
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


def run_breakaway_phase(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    capture_buffer: CaptureBuffer,
    target_motor_id: int,
    group_index: int,
    round_index: int,
) -> BreakawayIdentificationResult:
    log_stage_transition("breakaway", target_motor_id=target_motor_id)
    rerun_recorder.log_phase_event(motor_id=int(target_motor_id), phase_name="breakaway", detail="start")
    positive = scan_breakaway_direction(
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
    negative = scan_breakaway_direction(
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
    scan_limit = float(config.breakaway.scan_max_torque[config.motor_index(target_motor_id)])
    torque_step = float(config.breakaway.torque_step)
    limit_tolerance = max(0.5 * torque_step, 1.0e-9)
    positive_limit_reached = bool(np.isclose(abs(float(positive)), scan_limit, atol=limit_tolerance, rtol=0.0))
    negative_limit_reached = bool(np.isclose(abs(float(negative)), scan_limit, atol=limit_tolerance, rtol=0.0))
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
            "scan_max_torque": scan_limit,
            "torque_step": torque_step,
            "hold_duration": float(config.breakaway.hold_duration),
            "positive_scan_limit_reached": positive_limit_reached,
            "negative_scan_limit_reached": negative_limit_reached,
            "both_scan_limits_reached": bool(positive_limit_reached and negative_limit_reached),
        },
    )


__all__ = ["breakaway_torque_scan_values", "run_breakaway_phase", "scan_breakaway_direction"]
