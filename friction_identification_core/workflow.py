from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.core import (
    AbortEvent,
    MotorCompensationParameters,
    ReferenceSample,
    ReferenceTrajectory,
    RoundCapture,
    RunResult,
    SingleMotorController,
    build_reference_trajectory,
    runtime_abort_from_frame as core_runtime_abort_from_frame,
    safety_margin_text as core_safety_margin_text,
    theoretical_velocity_for_phase as core_theoretical_velocity_for_phase,
    velocity_abort_limit_for_phase as core_velocity_abort_limit_for_phase,
    zeroing_lock_state_update as core_zeroing_lock_state_update,
    zeroing_theoretical_velocity_from_position as core_zeroing_theoretical_velocity_from_position,
)
from friction_identification_core.identification import identify_motor_friction, identify_motor_friction_lugre
from friction_identification_core.io import (
    MotorSequenceChecker,
    SerialFrameParser,
    SerialTransport,
    SingleMotorCommandAdapter,
    open_serial_transport,
)
from friction_identification_core.results import (
    ResultStore,
    RoundArtifact,
    SummaryPaths,
    load_compensation_parameters as resolve_compensation_parameters,
    log_info,
    utc_now_iso8601,
)
from friction_identification_core.visualization import RerunRecorder


class _RuntimeAbortError(ValueError):
    def __init__(self, event: AbortEvent) -> None:
        self.event = event
        super().__init__(event.error_message())


def _prebuild_references(config: Config) -> dict[int, ReferenceTrajectory]:
    references: dict[int, ReferenceTrajectory] = {}
    for motor_id in config.enabled_motor_ids:
        motor_index = config.motor_index(motor_id)
        references[int(motor_id)] = build_reference_trajectory(
            config.excitation,
            max_velocity=float(config.control.max_velocity[motor_index]),
        )
    return references


def _expected_velocity_vector(config: Config, *, target_index: int, target_velocity: float) -> np.ndarray:
    expected = np.zeros(config.motor_count, dtype=np.float64)
    expected[target_index] = float(target_velocity)
    return expected


def _expected_position_vector(config: Config, *, target_index: int, target_position: float) -> np.ndarray:
    expected = np.zeros(config.motor_count, dtype=np.float64)
    expected[target_index] = float(target_position)
    return expected


def _sent_command_vector(config: Config, *, target_index: int, target_command: float) -> np.ndarray:
    sent_commands = np.zeros(config.motor_count, dtype=np.float64)
    sent_commands[target_index] = float(target_command)
    return sent_commands


def _capture_compensation_metrics(capture: RoundCapture) -> dict[str, float]:
    velocity = np.asarray(capture.velocity, dtype=np.float64)
    velocity_cmd = np.asarray(capture.velocity_cmd, dtype=np.float64)
    error = velocity - velocity_cmd
    finite = np.isfinite(error)
    if not np.any(finite):
        return {
            "tracking_velocity_rmse": float("nan"),
            "tracking_velocity_mae": float("nan"),
            "tracking_velocity_max_abs": float("nan"),
        }
    error = error[finite]
    return {
        "tracking_velocity_rmse": float(np.sqrt(np.mean(error**2))),
        "tracking_velocity_mae": float(np.mean(np.abs(error))),
        "tracking_velocity_max_abs": float(np.max(np.abs(error))),
    }


def _send_zero_command(
    *,
    transport: SerialTransport,
    command_adapter: SingleMotorCommandAdapter,
    target_motor_id: int,
    rerun_recorder: RerunRecorder,
    config: Config,
    target_index: int,
) -> None:
    _send_target_command(
        transport=transport,
        command_adapter=command_adapter,
        target_motor_id=int(target_motor_id),
        target_command=0.0,
        rerun_recorder=rerun_recorder,
        config=config,
        target_index=target_index,
    )


def _send_target_command(
    *,
    transport: SerialTransport,
    command_adapter: SingleMotorCommandAdapter,
    target_motor_id: int,
    target_command: float,
    rerun_recorder: RerunRecorder,
    config: Config,
    target_index: int,
) -> None:
    packet = command_adapter.pack(int(target_motor_id), float(target_command))
    transport.write(packet)
    rerun_recorder.log_live_command_packet(
        sent_commands=_sent_command_vector(config, target_index=target_index, target_command=float(target_command)),
        expected_positions=_expected_position_vector(config, target_index=target_index, target_position=0.0),
        expected_velocities=_expected_velocity_vector(config, target_index=target_index, target_velocity=0.0),
        raw_packet=packet,
    )


def _take_control_with_zero_command(
    *,
    config: Config,
    transport: SerialTransport,
    parser: SerialFrameParser,
    command_adapter: SingleMotorCommandAdapter,
    rerun_recorder: RerunRecorder,
) -> None:
    target_motor_id = int(config.enabled_motor_ids[0])
    target_index = config.motor_index(target_motor_id)
    _send_zero_command(
        transport=transport,
        command_adapter=command_adapter,
        target_motor_id=target_motor_id,
        rerun_recorder=rerun_recorder,
        config=config,
        target_index=target_index,
    )
    if config.serial.flush_input_before_round:
        # Drop frames that may still reflect retained commands before this process took over.
        transport.reset_input_buffer()
        parser.reset()


def _phase_theoretical_velocity(
    reference: ReferenceTrajectory,
    *,
    phase_name: str,
    feedback_position: float,
    reference_index: int,
    zero_target_velocity_threshold: float,
) -> tuple[float, int]:
    return core_theoretical_velocity_for_phase(
        reference,
        phase_name=phase_name,
        feedback_position=feedback_position,
        reference_index=reference_index,
        zero_target_velocity_threshold=zero_target_velocity_threshold,
    )


def _zeroing_theoretical_velocity_from_position(
    *,
    filtered_position: float,
    zeroing_position_gain: float,
    zeroing_velocity_gain: float,
    zeroing_hard_velocity_limit: float,
) -> float:
    return core_zeroing_theoretical_velocity_from_position(
        filtered_position=filtered_position,
        zeroing_position_gain=zeroing_position_gain,
        zeroing_velocity_gain=zeroing_velocity_gain,
        zeroing_hard_velocity_limit=zeroing_hard_velocity_limit,
    )


def _safety_margin_text(
    *,
    velocity_limit: float,
    observed_velocity: float,
    torque_limit: float,
    feedback_torque: float,
    position_limit: float,
    feedback_position: float,
) -> str:
    return core_safety_margin_text(
        velocity_limit=velocity_limit,
        observed_velocity=observed_velocity,
        torque_limit=torque_limit,
        feedback_torque=feedback_torque,
        position_limit=position_limit,
        feedback_position=feedback_position,
    )


def _runtime_abort_from_frame(
    *,
    frame,
    stage: str,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    phase_name: str,
    velocity_limit: float,
    torque_limit: float,
    position_limit: float,
) -> AbortEvent | None:
    return core_runtime_abort_from_frame(
        frame=frame,
        stage=stage,
        target_motor_id=target_motor_id,
        group_index=group_index,
        round_index=round_index,
        phase_name=phase_name,
        velocity_limit=velocity_limit,
        torque_limit=torque_limit,
        position_limit=position_limit,
    )


def _load_compensation_parameters(
    config: Config,
    *,
    parameters_path: Path | None = None,
) -> tuple[Path, str, dict[int, MotorCompensationParameters]]:
    return resolve_compensation_parameters(
        config,
        parameters_path=parameters_path,
    )


def _perform_zeroing(
    *,
    config: Config,
    transport: SerialTransport,
    parser: SerialFrameParser,
    command_adapter: SingleMotorCommandAdapter,
    controller: SingleMotorController,
    rerun_recorder: RerunRecorder,
) -> None:
    for zeroing_index, target_motor_id in enumerate(config.enabled_motor_ids, start=1):
        if config.serial.flush_input_before_round:
            transport.reset_input_buffer()
            parser.reset()

        motor_index = config.motor_index(target_motor_id)
        max_torque = float(config.control.max_torque[motor_index])
        zeroing_position_gain = float(config.control.zeroing_position_gain[motor_index])
        zeroing_velocity_gain = float(config.control.zeroing_velocity_gain[motor_index])
        zeroing_hard_velocity_limit = float(config.control.zeroing_hard_velocity_limit[motor_index])
        zeroing_velocity_limit = float(config.control.zeroing_velocity_limit[motor_index])
        position_tolerance = float(config.control.zeroing_position_tolerance[motor_index])
        velocity_tolerance = float(config.control.zeroing_velocity_tolerance[motor_index])
        # Zeroing may need to recover from a pose outside the excitation envelope,
        # so do not reuse excitation.position_limit as a zeroing abort guard.
        position_abort_limit = float("nan")
        success_count = 0
        recent_position: deque[float] = deque(maxlen=5)
        recent_velocity: deque[float] = deque(maxlen=5)
        last_raw_position = float("nan")
        last_raw_velocity = float("nan")
        last_filtered_position = float("nan")
        last_filtered_velocity = float("nan")
        last_zeroing_command = 0.0
        zeroing_velocity_violation_streak = 0
        start_monotonic = time.monotonic()
        feedback_request_interval_s = max(float(config.serial.read_timeout), 5.0e-3)
        last_feedback_request_monotonic = 0.0

        def _request_feedback() -> None:
            nonlocal last_feedback_request_monotonic
            _send_target_command(
                transport=transport,
                command_adapter=command_adapter,
                target_motor_id=int(target_motor_id),
                target_command=float(last_zeroing_command),
                rerun_recorder=rerun_recorder,
                config=config,
                target_index=motor_index,
            )
            last_feedback_request_monotonic = time.monotonic()

        rerun_recorder.log_zeroing_event(
            event="zeroing_start",
            motor_id=int(target_motor_id),
            detail=f"index={zeroing_index}",
        )
        _request_feedback()

        while True:
            elapsed = time.monotonic() - start_monotonic
            if elapsed > float(config.control.zeroing_timeout):
                detail_parts = [f"elapsed={elapsed:.3f}"]
                if np.isfinite(last_raw_position):
                    detail_parts.append(f"raw_position={last_raw_position:.6f}")
                if np.isfinite(last_raw_velocity):
                    detail_parts.append(f"raw_velocity={last_raw_velocity:.6f}")
                if np.isfinite(last_filtered_position):
                    detail_parts.append(f"filtered_position={last_filtered_position:.6f}")
                if np.isfinite(last_filtered_velocity):
                    detail_parts.append(f"filtered_velocity={last_filtered_velocity:.6f}")
                detail_parts.append(
                    f"success_count={int(success_count)}/{int(config.control.zeroing_required_frames)}"
                )
                detail = ", ".join(detail_parts)
                _send_zero_command(
                    transport=transport,
                    command_adapter=command_adapter,
                    target_motor_id=int(target_motor_id),
                    rerun_recorder=rerun_recorder,
                    config=config,
                    target_index=motor_index,
                )
                rerun_recorder.log_zeroing_event(
                    event="zeroing_timeout",
                    motor_id=int(target_motor_id),
                    detail=detail,
                )
                raise _RuntimeAbortError(
                    AbortEvent(
                        reason="zeroing_timeout",
                        stage="zeroing",
                        motor_id=int(target_motor_id),
                        group_index=0,
                        round_index=0,
                        phase_name="zeroing",
                        detail=detail,
                    )
                )

            chunk = transport.read(config.serial.read_chunk_size)
            if chunk:
                parser.feed(chunk)
            saw_frame = False
            saw_target_frame = False
            while True:
                frame = parser.pop_frame()
                if frame is None:
                    break
                saw_frame = True
                rerun_recorder.log_live_feedback_frame(
                    group_index=0,
                    round_index=0,
                    active_motor_id=int(target_motor_id),
                    motor_id=int(frame.motor_id),
                    state=int(frame.state),
                    position=float(frame.position),
                    velocity=float(frame.velocity),
                    feedback_torque=float(frame.torque),
                    mos_temperature=float(frame.mos_temperature),
                    phase_name="zeroing",
                    stage="zeroing",
                )
                if int(frame.motor_id) != int(target_motor_id):
                    continue
                saw_target_frame = True

                last_raw_position = float(frame.position)
                last_raw_velocity = float(frame.velocity)
                recent_position.append(float(frame.position))
                recent_velocity.append(float(frame.velocity))
                filtered_position = float(np.median(np.asarray(recent_position, dtype=np.float64)))
                filtered_velocity = float(np.median(np.asarray(recent_velocity, dtype=np.float64)))
                last_filtered_position = filtered_position
                last_filtered_velocity = filtered_velocity
                zeroing_theoretical_velocity = _zeroing_theoretical_velocity_from_position(
                    filtered_position=filtered_position,
                    zeroing_position_gain=zeroing_position_gain,
                    zeroing_velocity_gain=zeroing_velocity_gain,
                    zeroing_hard_velocity_limit=zeroing_hard_velocity_limit,
                )
                # Keep a dedicated zeroing abort guard during the return motion and
                # only use zeroing_velocity_limit to decide when the final near-zero
                # lock criteria are allowed to engage.
                active_velocity_limit = zeroing_hard_velocity_limit
                if abs(float(frame.velocity)) > float(active_velocity_limit):
                    zeroing_velocity_violation_streak += 1
                else:
                    zeroing_velocity_violation_streak = 0
                abort_event = _runtime_abort_from_frame(
                    frame=frame,
                    stage="zeroing",
                    target_motor_id=int(target_motor_id),
                    group_index=0,
                    round_index=0,
                    phase_name="zeroing",
                    # Zeroing feedback can contain isolated velocity spikes even when
                    # the motor is stationary; require two consecutive violations
                    # before treating it as a true overspeed event.
                    velocity_limit=(
                        float(active_velocity_limit)
                        if zeroing_velocity_violation_streak >= 2
                        else float("inf")
                    ),
                    torque_limit=float(max_torque),
                    position_limit=float(position_abort_limit),
                )
                if abort_event is not None:
                    _send_zero_command(
                        transport=transport,
                        command_adapter=command_adapter,
                        target_motor_id=int(target_motor_id),
                        rerun_recorder=rerun_recorder,
                        config=config,
                        target_index=motor_index,
                    )
                    rerun_recorder.log_zeroing_event(
                        event="zeroing_abort",
                        motor_id=int(target_motor_id),
                        detail=abort_event.reason,
                    )
                    raise _RuntimeAbortError(abort_event)

                zero_reference = ReferenceSample(0.0, 0.0, 0.0, "zeroing")
                command_raw, command = controller.update(
                    int(target_motor_id),
                    zero_reference,
                    frame,
                    position_gain=zeroing_position_gain,
                    velocity_gain=zeroing_velocity_gain,
                )
                last_zeroing_command = float(command)
                _send_target_command(
                    transport=transport,
                    command_adapter=command_adapter,
                    target_motor_id=int(target_motor_id),
                    target_command=float(command),
                    rerun_recorder=rerun_recorder,
                    config=config,
                    target_index=motor_index,
                )

                lock_state = core_zeroing_lock_state_update(
                    success_count=int(success_count),
                    filtered_position=filtered_position,
                    filtered_velocity=filtered_velocity,
                    zeroing_theoretical_velocity=zeroing_theoretical_velocity,
                    position_tolerance=position_tolerance,
                    velocity_tolerance=velocity_tolerance,
                    zeroing_velocity_limit=zeroing_velocity_limit,
                )
                success_count = int(lock_state.success_count)

                rerun_recorder.log_zeroing_sample(
                    motor_id=int(target_motor_id),
                    raw_position=float(frame.position),
                    raw_velocity=float(frame.velocity),
                    filtered_position=filtered_position,
                    filtered_velocity=filtered_velocity,
                    position_error=float(-filtered_position),
                    velocity_error=float(-filtered_velocity),
                    success_count=int(success_count),
                    required_frames=int(config.control.zeroing_required_frames),
                    inside_entry_band=bool(lock_state.inside_entry_band),
                    inside_exit_band=bool(lock_state.inside_exit_band),
                    command_raw=float(command_raw),
                    command=float(command),
                    feedback_torque=float(frame.torque),
                    torque_limit=float(max_torque),
                    velocity_limit=float(active_velocity_limit),
                    position_limit=float(position_abort_limit),
                )
                if success_count >= int(config.control.zeroing_required_frames):
                    _send_zero_command(
                        transport=transport,
                        command_adapter=command_adapter,
                        target_motor_id=int(target_motor_id),
                        rerun_recorder=rerun_recorder,
                        config=config,
                        target_index=motor_index,
                    )
                    rerun_recorder.log_zeroing_event(
                        event="zeroing_locked",
                        motor_id=int(target_motor_id),
                        detail=f"elapsed={elapsed:.3f}",
                    )
                    break
            if success_count >= int(config.control.zeroing_required_frames):
                break
            if not saw_target_frame and (time.monotonic() - last_feedback_request_monotonic) >= feedback_request_interval_s:
                _request_feedback()
            if not saw_frame and not chunk:
                time.sleep(max(float(config.serial.read_timeout), 1.0e-3))


def _capture_round(
    *,
    config: Config,
    transport: SerialTransport,
    parser: SerialFrameParser,
    command_adapter: SingleMotorCommandAdapter,
    controller: SingleMotorController,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    rerun_recorder: RerunRecorder,
    mode: str,
    reference: ReferenceTrajectory,
    compensation: MotorCompensationParameters | None = None,
) -> RoundCapture:
    motor_name = config.motors.name_for(target_motor_id)
    target_index = config.motor_index(target_motor_id)
    target_max_velocity = float(config.control.max_velocity[target_index])
    target_max_torque = float(config.control.max_torque[target_index])
    position_limit = float(config.excitation.position_limit)
    planned_duration_s = float(reference.duration_s)

    if config.serial.flush_input_before_round:
        transport.reset_input_buffer()
        parser.reset()

    time_log: list[float] = []
    motor_id_log: list[int] = []
    position_log: list[float] = []
    velocity_log: list[float] = []
    torque_log: list[float] = []
    command_raw_log: list[float] = []
    command_log: list[float] = []
    position_cmd_log: list[float] = []
    velocity_cmd_log: list[float] = []
    acceleration_cmd_log: list[float] = []
    phase_log: list[str] = []
    state_log: list[int] = []
    mos_temperature_log: list[float] = []
    id_match_log: list[bool] = []
    observed_frame_count = 0
    target_frame_count = 0
    target_frame_goal = max(int(reference.time.size), 1)
    sequence_checker = MotorSequenceChecker(config.motor_ids)
    sync_required_target_frames = max(int(config.serial.sync_cycles_required), 1)
    sync_wait_duration_s = 0.0
    round_started_at = utc_now_iso8601()
    sync_started_monotonic = time.monotonic()
    capture_started_monotonic: float | None = None
    capture_started_at: str | None = None
    target_sync_frame_count = 0
    feedback_request_interval_s = max(float(config.serial.read_timeout), 5.0e-3)
    last_feedback_request_monotonic = 0.0

    def _request_sync_feedback() -> None:
        nonlocal last_feedback_request_monotonic
        _send_zero_command(
            transport=transport,
            command_adapter=command_adapter,
            target_motor_id=int(target_motor_id),
            rerun_recorder=rerun_recorder,
            config=config,
            target_index=target_index,
        )
        last_feedback_request_monotonic = time.monotonic()

    rerun_recorder.log_round_timing(
        group_index=int(group_index),
        round_index=int(round_index),
        active_motor_id=int(target_motor_id),
        planned_duration_s=planned_duration_s,
        actual_capture_duration_s=0.0,
        sync_wait_duration_s=0.0,
        round_total_duration_s=0.0,
    )
    _request_sync_feedback()

    try:
        while True:
            now = time.monotonic()
            if capture_started_monotonic is not None:
                elapsed_s = now - capture_started_monotonic
                if elapsed_s >= planned_duration_s:
                    break
            elif (now - sync_started_monotonic) >= float(config.serial.sync_timeout):
                raise _RuntimeAbortError(
                    AbortEvent(
                        reason="sync_timeout",
                        stage=str(mode),
                        motor_id=int(target_motor_id),
                        group_index=int(group_index),
                        round_index=int(round_index),
                        phase_name="sync_wait",
                    )
                )

            chunk = transport.read(config.serial.read_chunk_size)
            if chunk:
                parser.feed(chunk)
            saw_frame = False
            saw_target_frame = False
            while True:
                frame = parser.pop_frame()
                if frame is None:
                    break
                saw_frame = True
                observed_frame_count += 1
                sequence_checker.observe(int(frame.motor_id))
                if capture_started_monotonic is None:
                    feedback_phase_name = "sync_wait"
                else:
                    feedback_phase_name = (
                        str(reference.sample(time.monotonic() - capture_started_monotonic).phase_name)
                        if int(frame.motor_id) == int(target_motor_id)
                        else "idle"
                    )
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
                    phase_name=str(feedback_phase_name),
                    stage=str(mode),
                )

                if capture_started_monotonic is None:
                    if int(frame.motor_id) == int(target_motor_id):
                        saw_target_frame = True
                        target_sync_frame_count += 1
                        _send_zero_command(
                            transport=transport,
                            command_adapter=command_adapter,
                            target_motor_id=int(target_motor_id),
                            rerun_recorder=rerun_recorder,
                            config=config,
                            target_index=target_index,
                        )
                    if target_sync_frame_count >= sync_required_target_frames:
                        sync_wait_duration_s = time.monotonic() - sync_started_monotonic
                        capture_started_monotonic = time.monotonic()
                        capture_started_at = utc_now_iso8601()
                    continue

                elapsed_s = time.monotonic() - capture_started_monotonic
                reference_index = reference.index_at(elapsed_s)
                reference_sample = reference.sample(elapsed_s)
                phase_name = str(reference_sample.phase_name) if int(frame.motor_id) == int(target_motor_id) else "idle"
                expected_velocity = float(reference_sample.velocity_cmd) if int(frame.motor_id) == int(target_motor_id) else 0.0
                expected_position = float(reference_sample.position_cmd) if int(frame.motor_id) == int(target_motor_id) else 0.0
                expected_acceleration = float(reference_sample.acceleration_cmd) if int(frame.motor_id) == int(target_motor_id) else 0.0
                command_raw = 0.0
                command = 0.0
                velocity_limit = float(config.control.low_speed_abort_limit[target_index])

                if int(frame.motor_id) == int(target_motor_id):
                    v_theory, _matched_index = _phase_theoretical_velocity(
                        reference,
                        phase_name=str(reference_sample.phase_name),
                        feedback_position=float(frame.position),
                        reference_index=reference_index,
                        zero_target_velocity_threshold=float(config.control.zero_target_velocity_threshold[target_index]),
                    )
                    phase_mask = np.asarray(reference.phase_name).astype(str) == str(reference_sample.phase_name)
                    phase_peak_velocity = (
                        float(np.max(np.abs(np.asarray(reference.velocity_cmd[phase_mask], dtype=np.float64))))
                        if np.any(phase_mask)
                        else abs(float(v_theory))
                    )
                    velocity_limit = core_velocity_abort_limit_for_phase(
                        phase_name=str(reference_sample.phase_name),
                        theoretical_velocity=max(abs(float(v_theory)), float(phase_peak_velocity)),
                        low_speed_abort_limit=float(config.control.low_speed_abort_limit[target_index]),
                        speed_abort_ratio=float(config.control.speed_abort_ratio[target_index]),
                    )
                    abort_event = _runtime_abort_from_frame(
                        frame=frame,
                        stage=str(mode),
                        target_motor_id=int(target_motor_id),
                        group_index=int(group_index),
                        round_index=int(round_index),
                        phase_name=str(reference_sample.phase_name),
                        velocity_limit=float(velocity_limit),
                        torque_limit=float(target_max_torque),
                        position_limit=float(position_limit),
                    )
                    if abort_event is not None:
                        raise _RuntimeAbortError(abort_event)

                    command_raw, limited_command = controller.update(
                        int(target_motor_id),
                        reference_sample,
                        frame,
                        compensation=compensation,
                    )
                    command = command_adapter.limit_command(int(target_motor_id), limited_command)
                    packet = command_adapter.pack(int(target_motor_id), command)
                    transport.write(packet)
                    rerun_recorder.log_live_command_packet(
                        sent_commands=_sent_command_vector(config, target_index=target_index, target_command=float(command)),
                        expected_positions=_expected_position_vector(
                            config,
                            target_index=target_index,
                            target_position=float(reference_sample.position_cmd),
                        ),
                        expected_velocities=_expected_velocity_vector(config, target_index=target_index, target_velocity=float(reference_sample.velocity_cmd)),
                        raw_packet=packet,
                    )
                    target_frame_count += 1

                    time_log.append(float(elapsed_s))
                    motor_id_log.append(int(frame.motor_id))
                    position_log.append(float(frame.position))
                    velocity_log.append(float(frame.velocity))
                    torque_log.append(float(frame.torque))
                    command_raw_log.append(float(command_raw))
                    command_log.append(float(command))
                    position_cmd_log.append(float(reference_sample.position_cmd))
                    velocity_cmd_log.append(float(reference_sample.velocity_cmd))
                    acceleration_cmd_log.append(float(reference_sample.acceleration_cmd))
                    phase_log.append(str(reference_sample.phase_name))
                    state_log.append(int(frame.state))
                    mos_temperature_log.append(float(frame.mos_temperature))
                    id_match_log.append(True)

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
                    reference_position=float(expected_position),
                    reference_velocity=float(expected_velocity),
                    reference_acceleration=float(expected_acceleration),
                    velocity_limit=float(velocity_limit),
                    torque_limit=float(target_max_torque),
                    position_limit=float(position_limit),
                    phase_name=str(phase_name),
                    stage=str(mode),
                    safety_margin_text=_safety_margin_text(
                        velocity_limit=float(velocity_limit),
                        observed_velocity=float(frame.velocity),
                        torque_limit=float(target_max_torque),
                        feedback_torque=float(frame.torque),
                        position_limit=float(position_limit),
                        feedback_position=float(frame.position),
                    ),
                )
                rerun_recorder.log_round_timing(
                    group_index=int(group_index),
                    round_index=int(round_index),
                    active_motor_id=int(target_motor_id),
                    planned_duration_s=planned_duration_s,
                    actual_capture_duration_s=float(elapsed_s),
                    sync_wait_duration_s=float(sync_wait_duration_s),
                    round_total_duration_s=float(time.monotonic() - sync_started_monotonic),
                )
            if capture_started_monotonic is None and not saw_target_frame:
                if (time.monotonic() - last_feedback_request_monotonic) >= feedback_request_interval_s:
                    _request_sync_feedback()
            if not saw_frame and not chunk:
                time.sleep(max(float(config.serial.read_timeout), 1.0e-3))
    except KeyboardInterrupt as exc:
        raise _RuntimeAbortError(
            AbortEvent(
                reason="interrupted",
                stage=str(mode),
                motor_id=int(target_motor_id),
                group_index=int(group_index),
                round_index=int(round_index),
                phase_name="interrupted",
            )
        ) from exc
    finally:
        _send_zero_command(
            transport=transport,
            command_adapter=command_adapter,
            target_motor_id=int(target_motor_id),
            rerun_recorder=rerun_recorder,
            config=config,
            target_index=target_index,
        )

    actual_capture_duration_s = min(time.monotonic() - (capture_started_monotonic or time.monotonic()), planned_duration_s)
    round_total_duration_s = time.monotonic() - sync_started_monotonic
    rerun_recorder.log_round_stop(
        group_index=int(group_index),
        round_index=int(round_index),
        motor_id=int(target_motor_id),
        phase_name="completed",
        stage=str(mode),
    )

    target_frame_ratio = min(float(target_frame_count / target_frame_goal), 1.0) if target_frame_goal else 0.0
    metadata = {
        "mode": str(mode),
        "group_index": int(group_index),
        "round_index": int(round_index),
        "target_motor_id": int(target_motor_id),
        "enabled_motor_ids": list(config.enabled_motor_ids),
        "excitation_config": asdict(config.excitation),
        "start_time": capture_started_at or round_started_at,
        "round_start_time": round_started_at,
        "stop_reason": "completed",
        "synced_before_capture": True,
        "sync_wait_duration_s": float(sync_wait_duration_s),
        "sync_timeout": float(config.serial.sync_timeout),
        "sync_required_target_frames": int(sync_required_target_frames),
        "target_sync_frame_count": int(target_sync_frame_count),
        "observed_frame_count": int(observed_frame_count),
        "sequence_error_count": int(sequence_checker.error_count),
        "sequence_error_ratio": (float(sequence_checker.error_count) / float(observed_frame_count)) if observed_frame_count else 0.0,
        "target_frame_goal": int(target_frame_goal),
        "target_frame_count": int(target_frame_count),
        "target_frame_ratio": float(target_frame_ratio),
        "target_max_velocity": float(target_max_velocity),
        "target_max_torque": float(target_max_torque),
        "position_limit": float(position_limit),
        "planned_duration_s": float(planned_duration_s),
        "actual_capture_duration_s": float(actual_capture_duration_s),
        "round_total_duration_s": float(round_total_duration_s),
    }
    return RoundCapture(
        group_index=int(group_index),
        round_index=int(round_index),
        target_motor_id=int(target_motor_id),
        motor_name=motor_name,
        time=np.asarray(time_log, dtype=np.float64),
        motor_id=np.asarray(motor_id_log, dtype=np.int64),
        position=np.asarray(position_log, dtype=np.float64),
        velocity=np.asarray(velocity_log, dtype=np.float64),
        torque_feedback=np.asarray(torque_log, dtype=np.float64),
        command_raw=np.asarray(command_raw_log, dtype=np.float64),
        command=np.asarray(command_log, dtype=np.float64),
        position_cmd=np.asarray(position_cmd_log, dtype=np.float64),
        velocity_cmd=np.asarray(velocity_cmd_log, dtype=np.float64),
        acceleration_cmd=np.asarray(acceleration_cmd_log, dtype=np.float64),
        phase_name=np.asarray(phase_log),
        state=np.asarray(state_log, dtype=np.uint8),
        mos_temperature=np.asarray(mos_temperature_log, dtype=np.float64),
        id_match_ok=np.asarray(id_match_log, dtype=bool),
        metadata=metadata,
    )


def _step_phase_name(step_index: int) -> str:
    return f"step_{int(step_index):03d}"


def _max_step_index_for_motor(
    *,
    initial_torque: float,
    torque_step: float,
    max_torque: float,
) -> int:
    if float(initial_torque) >= float(max_torque):
        return 0
    return max(int(np.ceil((float(max_torque) - float(initial_torque)) / float(torque_step))), 0)


def _capture_step_torque_round(
    *,
    config: Config,
    transport: SerialTransport,
    parser: SerialFrameParser,
    command_adapter: SingleMotorCommandAdapter,
    target_motor_id: int,
    round_index: int,
    rerun_recorder: RerunRecorder,
) -> RoundCapture:
    motor_name = config.motors.name_for(target_motor_id)
    target_index = config.motor_index(target_motor_id)
    target_max_torque = float(config.control.max_torque[target_index])
    velocity_limit = float(config.step_torque.velocity_limit)
    initial_torque = float(config.step_torque.initial_torque)
    torque_step = float(config.step_torque.torque_step)
    hold_duration = float(config.step_torque.hold_duration)
    max_step_index = _max_step_index_for_motor(
        initial_torque=initial_torque,
        torque_step=torque_step,
        max_torque=target_max_torque,
    )
    planned_duration_s = float(max_step_index + 1) * hold_duration

    if config.serial.flush_input_before_round:
        transport.reset_input_buffer()
        parser.reset()

    time_log: list[float] = []
    motor_id_log: list[int] = []
    position_log: list[float] = []
    velocity_log: list[float] = []
    torque_log: list[float] = []
    command_raw_log: list[float] = []
    command_log: list[float] = []
    position_cmd_log: list[float] = []
    velocity_cmd_log: list[float] = []
    acceleration_cmd_log: list[float] = []
    phase_log: list[str] = []
    state_log: list[int] = []
    mos_temperature_log: list[float] = []
    id_match_log: list[bool] = []

    observed_frame_count = 0
    target_frame_count = 0
    sequence_checker = MotorSequenceChecker(config.motor_ids)
    sync_required_target_frames = max(int(config.serial.sync_cycles_required), 1)
    sync_wait_duration_s = 0.0
    round_started_at = utc_now_iso8601()
    sync_started_monotonic = time.monotonic()
    capture_started_monotonic: float | None = None
    capture_started_at: str | None = None
    target_sync_frame_count = 0
    feedback_request_interval_s = max(float(config.serial.read_timeout), 5.0e-3)
    feedback_log_interval_s = max(min(hold_duration / 4.0, 0.25), 0.10)
    last_feedback_request_monotonic = 0.0
    last_feedback_log_monotonic = 0.0
    last_logged_step_index = -1
    stop_reason = "completed"
    velocity_limit_reached = False
    velocity_limit_trigger_command = float("nan")
    velocity_limit_trigger_feedback_torque = float("nan")
    velocity_limit_trigger_velocity = float("nan")

    def _step_command(step_index: int) -> float:
        return min(target_max_torque, initial_torque + float(step_index) * torque_step)

    def _request_sync_feedback() -> None:
        nonlocal last_feedback_request_monotonic
        _send_zero_command(
            transport=transport,
            command_adapter=command_adapter,
            target_motor_id=int(target_motor_id),
            rerun_recorder=rerun_recorder,
            config=config,
            target_index=target_index,
        )
        last_feedback_request_monotonic = time.monotonic()

    rerun_recorder.log_round_timing(
        group_index=1,
        round_index=int(round_index),
        active_motor_id=int(target_motor_id),
        planned_duration_s=planned_duration_s,
        actual_capture_duration_s=0.0,
        sync_wait_duration_s=0.0,
        round_total_duration_s=0.0,
    )
    _request_sync_feedback()

    try:
        while True:
            now = time.monotonic()
            if capture_started_monotonic is not None:
                elapsed_s = now - capture_started_monotonic
                if elapsed_s >= planned_duration_s:
                    stop_reason = "max_torque_reached"
                    break
            elif (now - sync_started_monotonic) >= float(config.serial.sync_timeout):
                raise _RuntimeAbortError(
                    AbortEvent(
                        reason="sync_timeout",
                        stage="step_torque",
                        motor_id=int(target_motor_id),
                        group_index=1,
                        round_index=int(round_index),
                        phase_name="sync_wait",
                    )
                )

            chunk = transport.read(config.serial.read_chunk_size)
            if chunk:
                parser.feed(chunk)
            saw_frame = False
            saw_target_frame = False
            round_finished = False

            while True:
                frame = parser.pop_frame()
                if frame is None:
                    break

                saw_frame = True
                observed_frame_count += 1
                sequence_checker.observe(int(frame.motor_id))
                if capture_started_monotonic is None:
                    feedback_phase_name = "sync_wait"
                else:
                    elapsed_for_feedback = time.monotonic() - capture_started_monotonic
                    feedback_step_index = min(int(elapsed_for_feedback // hold_duration), max_step_index)
                    feedback_phase_name = (
                        _step_phase_name(feedback_step_index)
                        if int(frame.motor_id) == int(target_motor_id)
                        else "idle"
                    )
                rerun_recorder.log_live_feedback_frame(
                    group_index=1,
                    round_index=int(round_index),
                    active_motor_id=int(target_motor_id),
                    motor_id=int(frame.motor_id),
                    state=int(frame.state),
                    position=float(frame.position),
                    velocity=float(frame.velocity),
                    feedback_torque=float(frame.torque),
                    mos_temperature=float(frame.mos_temperature),
                    phase_name=str(feedback_phase_name),
                    stage="step_torque",
                )

                if capture_started_monotonic is None:
                    if int(frame.motor_id) == int(target_motor_id):
                        saw_target_frame = True
                        target_sync_frame_count += 1
                        _send_zero_command(
                            transport=transport,
                            command_adapter=command_adapter,
                            target_motor_id=int(target_motor_id),
                            rerun_recorder=rerun_recorder,
                            config=config,
                            target_index=target_index,
                        )
                    if target_sync_frame_count >= sync_required_target_frames:
                        sync_wait_duration_s = time.monotonic() - sync_started_monotonic
                        capture_started_monotonic = time.monotonic()
                        capture_started_at = utc_now_iso8601()
                    continue

                if int(frame.motor_id) != int(target_motor_id):
                    continue

                saw_target_frame = True
                elapsed_s = time.monotonic() - capture_started_monotonic
                step_index = min(int(elapsed_s // hold_duration), max_step_index)
                step_name = _step_phase_name(step_index)
                target_command = float(_step_command(step_index))
                overspeed_event = _runtime_abort_from_frame(
                    frame=frame,
                    stage="step_torque",
                    target_motor_id=int(target_motor_id),
                    group_index=1,
                    round_index=int(round_index),
                    phase_name=step_name,
                    velocity_limit=velocity_limit,
                    torque_limit=float("inf"),
                    position_limit=float("nan"),
                )

                time_log.append(float(elapsed_s))
                motor_id_log.append(int(frame.motor_id))
                position_log.append(float(frame.position))
                velocity_log.append(float(frame.velocity))
                torque_log.append(float(frame.torque))
                command_raw_log.append(float(target_command))
                command_log.append(float(target_command))
                position_cmd_log.append(0.0)
                velocity_cmd_log.append(0.0)
                acceleration_cmd_log.append(0.0)
                phase_log.append(step_name)
                state_log.append(int(frame.state))
                mos_temperature_log.append(float(frame.mos_temperature))
                id_match_log.append(True)
                target_frame_count += 1

                rerun_recorder.log_live_motor_sample(
                    group_index=1,
                    round_index=int(round_index),
                    active_motor_id=int(target_motor_id),
                    motor_id=int(frame.motor_id),
                    position=float(frame.position),
                    velocity=float(frame.velocity),
                    feedback_torque=float(frame.torque),
                    command_raw=float(target_command),
                    command=float(target_command),
                    reference_position=0.0,
                    reference_velocity=0.0,
                    reference_acceleration=0.0,
                    velocity_limit=float(velocity_limit),
                    torque_limit=float(target_max_torque),
                    position_limit=float("nan"),
                    phase_name=step_name,
                    stage="step_torque",
                    safety_margin_text=(
                        f"velocity_margin={velocity_limit - abs(float(frame.velocity)):+.6f}, "
                        f"torque_margin={target_max_torque - abs(float(target_command)):+.6f}"
                    ),
                )
                rerun_recorder.log_round_timing(
                    group_index=1,
                    round_index=int(round_index),
                    active_motor_id=int(target_motor_id),
                    planned_duration_s=planned_duration_s,
                    actual_capture_duration_s=float(elapsed_s),
                    sync_wait_duration_s=float(sync_wait_duration_s),
                    round_total_duration_s=float(time.monotonic() - sync_started_monotonic),
                )

                now_monotonic = time.monotonic()
                if step_index != last_logged_step_index:
                    log_info(
                        f"motor_id={target_motor_id} step={step_index} "
                        f"command={target_command:.3f}Nm"
                    )
                    last_logged_step_index = step_index
                    last_feedback_log_monotonic = 0.0
                if (now_monotonic - last_feedback_log_monotonic) >= feedback_log_interval_s:
                    log_info(
                        f"motor_id={target_motor_id} feedback "
                        f"position={float(frame.position):+.4f}rad "
                        f"velocity={float(frame.velocity):+.4f}rad/s "
                        f"torque={float(frame.torque):+.4f}Nm "
                        f"command={target_command:+.4f}Nm"
                    )
                    last_feedback_log_monotonic = now_monotonic

                if overspeed_event is not None:
                    stop_reason = str(overspeed_event.reason)
                    velocity_limit_reached = True
                    velocity_limit_trigger_command = float(target_command)
                    velocity_limit_trigger_feedback_torque = float(frame.torque)
                    velocity_limit_trigger_velocity = float(frame.velocity)
                    log_info(
                        f"motor_id={target_motor_id} overspeed: "
                        f"velocity={float(frame.velocity):+.4f}rad/s "
                        f"limit={velocity_limit:.4f}rad/s, switching to next motor."
                    )
                    round_finished = True
                    break

                packet = command_adapter.pack(int(target_motor_id), float(target_command))
                transport.write(packet)
                rerun_recorder.log_live_command_packet(
                    sent_commands=_sent_command_vector(config, target_index=target_index, target_command=float(target_command)),
                    expected_positions=_expected_position_vector(
                        config,
                        target_index=target_index,
                        target_position=0.0,
                    ),
                    expected_velocities=_expected_velocity_vector(
                        config,
                        target_index=target_index,
                        target_velocity=0.0,
                    ),
                    raw_packet=packet,
                )

            if round_finished:
                break
            if capture_started_monotonic is None and not saw_target_frame:
                if (time.monotonic() - last_feedback_request_monotonic) >= feedback_request_interval_s:
                    _request_sync_feedback()
            if not saw_frame and not chunk:
                time.sleep(max(float(config.serial.read_timeout), 1.0e-3))
    except KeyboardInterrupt as exc:
        raise _RuntimeAbortError(
            AbortEvent(
                reason="interrupted",
                stage="step_torque",
                motor_id=int(target_motor_id),
                group_index=1,
                round_index=int(round_index),
                phase_name="interrupted",
            )
        ) from exc
    finally:
        _send_zero_command(
            transport=transport,
            command_adapter=command_adapter,
            target_motor_id=int(target_motor_id),
            rerun_recorder=rerun_recorder,
            config=config,
            target_index=target_index,
        )

    actual_capture_duration_s = (
        0.0 if capture_started_monotonic is None else time.monotonic() - capture_started_monotonic
    )
    round_total_duration_s = time.monotonic() - sync_started_monotonic
    target_frame_ratio = float(target_frame_count > 0)
    metadata = {
        "mode": "step_torque",
        "group_index": 1,
        "round_index": int(round_index),
        "target_motor_id": int(target_motor_id),
        "enabled_motor_ids": list(config.enabled_motor_ids),
        "start_time": capture_started_at or round_started_at,
        "round_start_time": round_started_at,
        "stop_reason": str(stop_reason),
        "synced_before_capture": bool(capture_started_monotonic is not None),
        "sync_wait_duration_s": float(sync_wait_duration_s),
        "sync_timeout": float(config.serial.sync_timeout),
        "sync_required_target_frames": int(sync_required_target_frames),
        "target_sync_frame_count": int(target_sync_frame_count),
        "observed_frame_count": int(observed_frame_count),
        "sequence_error_count": int(sequence_checker.error_count),
        "sequence_error_ratio": (
            float(sequence_checker.error_count) / float(observed_frame_count)
            if observed_frame_count
            else 0.0
        ),
        "target_frame_count": int(target_frame_count),
        "target_frame_ratio": float(target_frame_ratio),
        "target_max_torque": float(target_max_torque),
        "velocity_limit": float(velocity_limit),
        "velocity_limit_reached": bool(velocity_limit_reached),
        "velocity_limit_trigger_command": float(velocity_limit_trigger_command),
        "velocity_limit_trigger_feedback_torque": float(velocity_limit_trigger_feedback_torque),
        "velocity_limit_trigger_velocity": float(velocity_limit_trigger_velocity),
        "initial_torque": float(initial_torque),
        "torque_step": float(torque_step),
        "hold_duration": float(hold_duration),
        "max_step_index": int(max_step_index),
        "planned_duration_s": float(planned_duration_s),
        "actual_capture_duration_s": float(actual_capture_duration_s),
        "round_total_duration_s": float(round_total_duration_s),
    }
    return RoundCapture(
        group_index=1,
        round_index=int(round_index),
        target_motor_id=int(target_motor_id),
        motor_name=motor_name,
        time=np.asarray(time_log, dtype=np.float64),
        motor_id=np.asarray(motor_id_log, dtype=np.int64),
        position=np.asarray(position_log, dtype=np.float64),
        velocity=np.asarray(velocity_log, dtype=np.float64),
        torque_feedback=np.asarray(torque_log, dtype=np.float64),
        command_raw=np.asarray(command_raw_log, dtype=np.float64),
        command=np.asarray(command_log, dtype=np.float64),
        position_cmd=np.asarray(position_cmd_log, dtype=np.float64),
        velocity_cmd=np.asarray(velocity_cmd_log, dtype=np.float64),
        acceleration_cmd=np.asarray(acceleration_cmd_log, dtype=np.float64),
        phase_name=np.asarray(phase_log),
        state=np.asarray(state_log, dtype=np.uint8),
        mos_temperature=np.asarray(mos_temperature_log, dtype=np.float64),
        id_match_ok=np.asarray(id_match_log, dtype=bool),
        metadata=metadata,
    )


def run_step_torque(
    config: Config,
    *,
    transport_factory: Callable[[], SerialTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    store = ResultStore(config, mode="step_torque")
    parser = SerialFrameParser(max_motor_id=max(config.motor_ids))
    command_adapter = SingleMotorCommandAdapter(
        motor_count=max(config.motor_ids),
        torque_limits=config.control.max_torque,
    )
    rerun_recorder = RerunRecorder(
        store.rerun_recording_path,
        motor_ids=config.motor_ids,
        motor_names={motor_id: config.motors.name_for(motor_id) for motor_id in config.motor_ids},
        mode="step_torque",
        show_viewer=show_rerun_viewer,
    )
    artifacts: list[RoundArtifact] = []

    transport = transport_factory() if transport_factory is not None else open_serial_transport(config.serial)
    try:
        _take_control_with_zero_command(
            config=config,
            transport=transport,
            parser=parser,
            command_adapter=command_adapter,
            rerun_recorder=rerun_recorder,
        )
        total_rounds = len(config.enabled_motor_ids)
        for round_index, target_motor_id in enumerate(config.enabled_motor_ids, start=1):
            log_info(
                "Starting step torque round "
                f"{round_index}/{total_rounds}: "
                f"motor_id={target_motor_id}"
            )
            capture = _capture_step_torque_round(
                config=config,
                transport=transport,
                parser=parser,
                command_adapter=command_adapter,
                target_motor_id=int(target_motor_id),
                round_index=int(round_index),
                rerun_recorder=rerun_recorder,
            )
            if bool(capture.metadata.get("velocity_limit_reached", False)):
                log_info(
                    f"motor_id={target_motor_id} finished: "
                    f"speed limit reached at command="
                    f"{float(capture.metadata['velocity_limit_trigger_command']):.4f}Nm, "
                    f"feedback_torque="
                    f"{float(capture.metadata['velocity_limit_trigger_feedback_torque']):+.4f}Nm, "
                    f"velocity="
                    f"{float(capture.metadata['velocity_limit_trigger_velocity']):+.4f}rad/s"
                )
            else:
                log_info(
                    f"motor_id={target_motor_id} finished: "
                    f"speed limit not reached, max torque limit="
                    f"{float(capture.metadata['target_max_torque']):.4f}Nm"
                )
            capture_path = store.save_capture(capture)
            artifacts.append(
                RoundArtifact(
                    capture=capture,
                    identification=None,
                    dynamic_identification=None,
                    capture_path=capture_path,
                    identification_path=None,
                    dynamic_identification_path=None,
                )
            )
        store.finalize()
        return RunResult(
            artifacts=tuple(artifacts),
            summary_paths=None,
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


def run_identify(
    config: Config,
    *,
    transport_factory: Callable[[], SerialTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    references = _prebuild_references(config)
    store = ResultStore(config, mode="identify")
    parser = SerialFrameParser(max_motor_id=max(config.motor_ids))
    command_adapter = SingleMotorCommandAdapter(
        motor_count=max(config.motor_ids),
        torque_limits=config.control.max_torque,
    )
    controller = SingleMotorController(config)
    rerun_recorder = RerunRecorder(
        store.rerun_recording_path,
        motor_ids=config.motor_ids,
        motor_names={motor_id: config.motors.name_for(motor_id) for motor_id in config.motor_ids},
        mode="identify",
        show_viewer=show_rerun_viewer,
    )
    artifacts: list[RoundArtifact] = []

    transport = transport_factory() if transport_factory is not None else open_serial_transport(config.serial)
    try:
        _take_control_with_zero_command(
            config=config,
            transport=transport,
            parser=parser,
            command_adapter=command_adapter,
            rerun_recorder=rerun_recorder,
        )
        _perform_zeroing(
            config=config,
            transport=transport,
            parser=parser,
            command_adapter=command_adapter,
            controller=controller,
            rerun_recorder=rerun_recorder,
        )
        total_rounds = int(config.group_count) * len(config.enabled_motor_ids)
        current_round = 0
        for group_index in range(1, int(config.group_count) + 1):
            for target_motor_id in config.enabled_motor_ids:
                current_round += 1
                log_info(
                    "Starting identify round "
                    f"{current_round}/{total_rounds}: "
                    f"group={group_index}, motor_id={target_motor_id}"
                )
                capture = _capture_round(
                    config=config,
                    transport=transport,
                    parser=parser,
                    command_adapter=command_adapter,
                    controller=controller,
                    target_motor_id=int(target_motor_id),
                    group_index=int(group_index),
                    round_index=int(current_round),
                    rerun_recorder=rerun_recorder,
                    mode="identify",
                    reference=references[int(target_motor_id)],
                )
                identification = identify_motor_friction(
                    config.identification,
                    capture,
                    max_torque=float(config.control.max_torque[config.motor_index(target_motor_id)]),
                    max_velocity=float(config.control.max_velocity[config.motor_index(target_motor_id)]),
                )
                dynamic_identification = identify_motor_friction_lugre(
                    config.identification,
                    capture,
                    identification,
                )
                capture_path = store.save_capture(capture)
                identification_path = store.save_identification(capture, identification)
                dynamic_identification_path = store.save_dynamic_identification(capture, dynamic_identification)
                rerun_recorder.log_identification(capture, identification, dynamic_identification)
                artifacts.append(
                    RoundArtifact(
                        capture=capture,
                        identification=identification,
                        dynamic_identification=dynamic_identification,
                        capture_path=capture_path,
                        identification_path=identification_path,
                        dynamic_identification_path=dynamic_identification_path,
                    )
                )
        summary_paths = store.save_summary(artifacts)
        rerun_recorder.log_summary(
            summary_path=summary_paths.run_summary_path,
            report_path=summary_paths.run_summary_report_path,
            dynamic_summary_path=summary_paths.dynamic_run_summary_path,
            dynamic_report_path=summary_paths.dynamic_run_summary_report_path,
        )
        return RunResult(
            artifacts=tuple(artifacts),
            summary_paths=summary_paths,
            manifest_path=store.manifest_path,
        )
    except _RuntimeAbortError as exc:
        rerun_recorder.log_abort_event(exc.event.to_payload())
        store.record_abort_event(exc.event.to_payload())
        if artifacts:
            summary_paths = store.save_summary(artifacts)
            rerun_recorder.log_summary(
                summary_path=summary_paths.run_summary_path,
                report_path=summary_paths.run_summary_report_path,
                dynamic_summary_path=summary_paths.dynamic_run_summary_path,
                dynamic_report_path=summary_paths.dynamic_run_summary_report_path,
            )
        else:
            store.finalize()
        raise
    finally:
        rerun_recorder.close()
        transport.close()


def run_compensate(
    config: Config,
    *,
    transport_factory: Callable[[], SerialTransport] | None = None,
    show_rerun_viewer: bool = False,
    parameters_path: Path | None = None,
) -> RunResult:
    references = _prebuild_references(config)
    resolved_parameters_path, parameters_source, parameters_by_motor = _load_compensation_parameters(
        config,
        parameters_path=parameters_path,
    )
    log_info("Compensation parameters source: " f"{parameters_source} ({resolved_parameters_path})")
    store = ResultStore(config, mode="compensate")
    parser = SerialFrameParser(max_motor_id=max(config.motor_ids))
    command_adapter = SingleMotorCommandAdapter(
        motor_count=max(config.motor_ids),
        torque_limits=config.control.max_torque,
    )
    controller = SingleMotorController(config)
    rerun_recorder = RerunRecorder(
        store.rerun_recording_path,
        motor_ids=config.motor_ids,
        motor_names={motor_id: config.motors.name_for(motor_id) for motor_id in config.motor_ids},
        mode="compensate",
        show_viewer=show_rerun_viewer,
    )
    artifacts: list[RoundArtifact] = []

    for motor_id, parameters in parameters_by_motor.items():
        rerun_recorder.log_compensation_reference(
            motor_id=int(motor_id),
            parameters=parameters,
            parameters_path=resolved_parameters_path,
        )

    transport = transport_factory() if transport_factory is not None else open_serial_transport(config.serial)
    try:
        total_rounds = int(config.group_count) * len(config.enabled_motor_ids)
        current_round = 0
        for group_index in range(1, int(config.group_count) + 1):
            for target_motor_id in config.enabled_motor_ids:
                current_round += 1
                log_info(
                    "Starting compensate round "
                    f"{current_round}/{total_rounds}: "
                    f"group={group_index}, motor_id={target_motor_id}"
                )
                capture = _capture_round(
                    config=config,
                    transport=transport,
                    parser=parser,
                    command_adapter=command_adapter,
                    controller=controller,
                    target_motor_id=int(target_motor_id),
                    group_index=int(group_index),
                    round_index=int(current_round),
                    rerun_recorder=rerun_recorder,
                    mode="compensate",
                    reference=references[int(target_motor_id)],
                    compensation=parameters_by_motor[int(target_motor_id)],
                )
                capture = replace(
                    capture,
                    metadata={
                        **capture.metadata,
                        **_capture_compensation_metrics(capture),
                        "compensation_parameters_path": str(resolved_parameters_path),
                        "compensation_parameters": asdict(parameters_by_motor[int(target_motor_id)]),
                    },
                )
                capture_path = store.save_capture(capture)
                artifacts.append(
                    RoundArtifact(
                        capture=capture,
                        identification=None,
                        dynamic_identification=None,
                        capture_path=capture_path,
                        identification_path=None,
                        dynamic_identification_path=None,
                    )
                )
        store.finalize(compensation_parameters_path=resolved_parameters_path)
        return RunResult(
            artifacts=tuple(artifacts),
            summary_paths=None,
            manifest_path=store.manifest_path,
        )
    except _RuntimeAbortError as exc:
        rerun_recorder.log_abort_event(exc.event.to_payload())
        store.record_abort_event(exc.event.to_payload())
        store.finalize(compensation_parameters_path=resolved_parameters_path)
        raise
    finally:
        rerun_recorder.close()
        transport.close()


def _removed_compensation_mode(*_args, **_kwargs) -> RunResult:
    raise ValueError("Compensation mode has been removed. Use step mode instead.")


perform_zeroing = _perform_zeroing
capture_round = _capture_round
capture_step_torque_round = _capture_step_torque_round
run_step_torque_scan = run_step_torque
run_identify = run_step_torque
run_sequential_identification = run_step_torque
run_compensate = _removed_compensation_mode
run_compensation_validation = _removed_compensation_mode
