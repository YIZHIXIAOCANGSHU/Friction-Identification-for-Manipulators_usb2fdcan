from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from friction_identification_core.capture import CaptureBuffer, log_stage_transition, poll_feedback_frames, record_target_frame, send_command
from friction_identification_core.core import AbortEvent
from friction_identification_core.io import CommandTransport, FeedbackFrameParser
from friction_identification_core.limits import identification_limits_for_motor, validate_abs_less_or_equal, validate_abs_less_than
from friction_identification_core.runtime_config import Config
from friction_identification_core.safety import RuntimeAbortError, build_abort_event, perform_hard_abort, wait_for_stationary
from friction_identification_core.visualization import RerunRecorder


@dataclass(frozen=True)
class DynamicMitTrajectory:
    time_s: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    @property
    def duration_s(self) -> float:
        if self.time_s.size == 0:
            return 0.0
        return float(self.time_s[-1])

    def sample(self, elapsed_s: float) -> tuple[float, float, float]:
        t = float(np.clip(float(elapsed_s), 0.0, self.duration_s))
        return (
            float(np.interp(t, self.time_s, self.position)),
            float(np.interp(t, self.time_s, self.velocity)),
            float(np.interp(t, self.time_s, self.acceleration)),
        )


def _dynamic_mit_limit_error_detail(config: Config, *, max_velocity: float, identification_speed_abs: float) -> str:
    trajectory_type = str(config.dynamic_mit.trajectory_type)
    if trajectory_type == "sine":
        knobs = "position_amplitude or frequency_hz"
    elif trajectory_type == "chirp":
        knobs = "position_amplitude or frequency_range_hz"
    else:
        knobs = "velocity_limit or cycle_count"
    return (
        f"generated dynamic_mit trajectory max_abs_velocity={float(max_velocity):.6f} rad/s, "
        f"dynamic_mit.velocity_limit={float(config.dynamic_mit.velocity_limit):.6f} rad/s, "
        f"identification_speed_abs={float(identification_speed_abs):.6f} rad/s; "
        f"lower {knobs}."
    )


def build_dynamic_mit_trajectory(
    config: Config,
    *,
    sample_period_s: float = 0.002,
    target_motor_id: int | None = None,
) -> DynamicMitTrajectory:
    duration = float(config.dynamic_mit.record_duration)
    sample_count = max(int(np.ceil(duration / max(float(sample_period_s), 1.0e-4))) + 1, 3)
    time_s = np.linspace(0.0, duration, sample_count, dtype=np.float64)
    amplitude = float(config.dynamic_mit.position_amplitude)
    velocity_limit = float(config.dynamic_mit.velocity_limit)
    if target_motor_id is None:
        target_motor_id = int(config.enabled_motor_ids[0] if config.enabled_motor_ids else config.motor_ids[0])
    limits = identification_limits_for_motor(config, target_motor_id=int(target_motor_id))
    trajectory_type = str(config.dynamic_mit.trajectory_type)

    if trajectory_type == "sine":
        frequency = float(config.dynamic_mit.frequency_hz)
        omega = 2.0 * np.pi * frequency
        position = 0.5 * amplitude * (1.0 - np.cos(omega * time_s))
        velocity = 0.5 * amplitude * omega * np.sin(omega * time_s)
        acceleration = 0.5 * amplitude * omega * omega * np.cos(omega * time_s)
    elif trajectory_type == "chirp":
        f0, f1 = config.dynamic_mit.frequency_range_hz
        sweep = (float(f1) - float(f0)) / max(duration, 1.0e-9)
        phase = 2.0 * np.pi * (float(f0) * time_s + 0.5 * sweep * time_s**2)
        position = amplitude * np.sin(phase)
        velocity = np.gradient(position, time_s, edge_order=1)
        acceleration = np.gradient(velocity, time_s, edge_order=1)
    elif trajectory_type == "trapezoid_velocity":
        cycles = max(int(config.dynamic_mit.cycle_count), 1)
        phase = (time_s / max(duration, 1.0e-9) * cycles) % 1.0
        velocity = np.zeros_like(time_s)
        for index, item in enumerate(phase):
            if item < 1.0 / 6.0:
                velocity[index] = velocity_limit * (item / (1.0 / 6.0))
            elif item < 2.0 / 6.0:
                velocity[index] = velocity_limit
            elif item < 4.0 / 6.0:
                velocity[index] = velocity_limit - 2.0 * velocity_limit * ((item - 2.0 / 6.0) / (2.0 / 6.0))
            elif item < 5.0 / 6.0:
                velocity[index] = -velocity_limit
            else:
                velocity[index] = -velocity_limit + velocity_limit * ((item - 5.0 / 6.0) / (1.0 / 6.0))
        acceleration = np.gradient(velocity, time_s, edge_order=1)
        position = np.cumsum(velocity) * float(time_s[1] - time_s[0])
        position -= float(position[0])
        max_position = float(np.nanmax(np.abs(position))) if position.size else 0.0
        if max_position > amplitude > 0.0:
            position *= amplitude / max_position
    else:  # pragma: no cover - validated by config loading
        raise ValueError(f"Unsupported dynamic MIT trajectory: {trajectory_type}")

    max_velocity = float(np.nanmax(np.abs(velocity))) if velocity.size else 0.0
    try:
        validate_abs_less_or_equal(
            velocity,
            limit_abs=velocity_limit,
            name="dynamic_mit.trajectory.velocity_cmd",
        )
        validate_abs_less_than(
            velocity,
            limit_abs=float(limits.identification_speed_abs),
            name="dynamic_mit.trajectory.velocity_cmd",
        )
    except ValueError as exc:
        raise ValueError(
            _dynamic_mit_limit_error_detail(
                config,
                max_velocity=max_velocity,
                identification_speed_abs=float(limits.identification_speed_abs),
            )
        ) from exc
    return DynamicMitTrajectory(
        time_s=np.asarray(time_s, dtype=np.float64),
        position=np.asarray(position, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        acceleration=np.asarray(acceleration, dtype=np.float64),
    )


def _motor_torque_limit(transport: CommandTransport, target_motor_id: int) -> float:
    motor_limits = getattr(transport, "motor_limits", None)
    if callable(motor_limits):
        limits = motor_limits(int(target_motor_id))
        torque_limit = getattr(limits, "tmax", None) if limits is not None else None
        if torque_limit is not None and np.isfinite(float(torque_limit)) and float(torque_limit) > 0.0:
            return float(torque_limit)
    return float("inf")


def _build_dynamic_velocity_abort_event(
    *,
    config: Config,
    stage: str,
    group_index: int,
    round_index: int,
    phase_name: str,
    target_motor_id: int,
    observed_velocity: float,
) -> AbortEvent | None:
    dynamic_velocity_limit = float(config.dynamic_mit.velocity_limit)
    if abs(float(observed_velocity)) < dynamic_velocity_limit:
        return None
    return AbortEvent(
        reason="dynamic_mit_velocity_abort",
        stage=str(stage),
        motor_id=int(target_motor_id),
        group_index=int(group_index),
        round_index=int(round_index),
        phase_name=str(phase_name),
        observed_velocity=float(observed_velocity),
        velocity_limit=float(dynamic_velocity_limit),
        detail=(
            f"abs_velocity={abs(float(observed_velocity)):.6f}, "
            f"hard_speed_limit={float(config.safety.hard_speed_abort_abs):.6f}"
        ),
    )


def _latest_captured_position(capture_buffer: CaptureBuffer) -> float | None:
    if not capture_buffer.position_log:
        return None
    return float(capture_buffer.position_log[-1])


def _poll_current_target_position(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    capture_buffer: CaptureBuffer,
    target_motor_id: int,
    group_index: int,
    round_index: int,
) -> float:
    target_index = config.motor_index(target_motor_id)
    send_interval_s = max(float(config.transport.read_timeout), 5.0e-3)
    deadline = time.monotonic() + max(float(config.transport.sync_timeout), send_interval_s)
    last_send = 0.0
    total_frame_count = 0
    other_motor_ids: set[int] = set()

    while time.monotonic() < deadline:
        now = time.monotonic()
        if (now - last_send) >= send_interval_s:
            send_command(
                config=config,
                transport=transport,
                rerun_recorder=rerun_recorder,
                target_motor_id=int(target_motor_id),
                target_index=target_index,
                semantic_mode="mit_state",
                command_value=0.0,
                kd_speed=float(config.dynamic_mit.kd),
                position_cmd=0.0,
                velocity_cmd=0.0,
                kp_cmd=0.0,
                torque_ff_cmd=0.0,
            )
            last_send = now

        frames, saw_chunk = poll_feedback_frames(
            transport=transport,
            parser=parser,
            read_chunk_size=config.transport.read_chunk_size,
        )
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
                phase_name="dynamic_mit_anchor",
                stage="dynamic-mit",
            )
            if int(frame.motor_id) != int(target_motor_id):
                other_motor_ids.add(int(frame.motor_id))
                continue

            abort_event = build_abort_event(
                config=config,
                stage="dynamic-mit",
                group_index=group_index,
                round_index=round_index,
                phase_name="dynamic_mit_anchor",
                target_motor_id=target_motor_id,
                frame=frame,
            )
            if abort_event is not None:
                perform_hard_abort(
                    config=config,
                    transport=transport,
                    parser=parser,
                    target_motor_id=target_motor_id,
                    semantic_mode="mit_state",
                )
                raise RuntimeAbortError(abort_event)

            position = float(frame.position)
            velocity = float(frame.velocity)
            dynamic_abort_event = _build_dynamic_velocity_abort_event(
                config=config,
                stage="dynamic-mit",
                group_index=group_index,
                round_index=round_index,
                phase_name="dynamic_mit_anchor",
                target_motor_id=target_motor_id,
                observed_velocity=velocity,
            )
            if dynamic_abort_event is not None:
                perform_hard_abort(
                    config=config,
                    transport=transport,
                    parser=parser,
                    target_motor_id=target_motor_id,
                    semantic_mode="mit_state",
                )
                raise RuntimeAbortError(dynamic_abort_event)

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
                position_cmd=position,
                velocity_cmd=0.0,
                acceleration_cmd=0.0,
                phase_name="dynamic_mit_anchor",
                stage="dynamic-mit",
                kd_cmd=float(config.dynamic_mit.kd),
                tracking_ok=True,
                safety_ok=abs(velocity) < float(config.dynamic_mit.velocity_limit),
                state_ok=int(frame.state) in {0, 1},
                used_for_fit=False,
            )
            return position

        if not frames and not saw_chunk:
            time.sleep(max(float(config.transport.read_timeout), 1.0e-3))

    other_motor_text = ",".join(str(motor_id) for motor_id in sorted(other_motor_ids)) or "-"
    raise RuntimeAbortError(
        AbortEvent(
            reason="feedback_timeout",
            stage="dynamic-mit",
            motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
            phase_name="dynamic_mit_anchor",
            detail=(
                f"timeout_s={max(float(config.transport.sync_timeout), send_interval_s):.3f}, "
                f"target_feedback_count=0, total_frames={int(total_frame_count)}, "
                f"other_motor_ids={other_motor_text}"
            ),
        )
    )


def run_dynamic_mit_phase(
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
    log_stage_transition("dynamic-mit", target_motor_id=target_motor_id)
    rerun_recorder.log_phase_event(motor_id=int(target_motor_id), phase_name="dynamic-mit", detail="start")
    target_index = config.motor_index(target_motor_id)
    trajectory = build_dynamic_mit_trajectory(config, target_motor_id=int(target_motor_id))
    send_interval_s = max(float(config.transport.read_timeout), 5.0e-3)
    torque_limit = _motor_torque_limit(transport, target_motor_id)

    if float(config.dynamic_mit.warmup_duration) > 0.0:
        wait_for_stationary(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            target_motor_id=target_motor_id,
            group_index=group_index,
            round_index=round_index,
            phase_name="dynamic_mit_warmup",
            stage="dynamic-mit",
            semantic_mode="mit_state",
            capture_buffer=capture_buffer,
            timeout_s=float(config.dynamic_mit.warmup_duration),
        )

    position_anchor = _latest_captured_position(capture_buffer)
    if position_anchor is None:
        position_anchor = _poll_current_target_position(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=target_motor_id,
            group_index=group_index,
            round_index=round_index,
        )
    rerun_recorder.log_phase_event(
        motor_id=int(target_motor_id),
        phase_name="dynamic-mit",
        detail=f"anchor_position={float(position_anchor):+.6f}",
    )

    start_monotonic = time.monotonic()
    last_send = 0.0
    last_position_cmd = 0.0
    last_velocity_cmd = 0.0
    last_acceleration_cmd = 0.0

    while True:
        now = time.monotonic()
        elapsed = now - start_monotonic
        if elapsed > trajectory.duration_s:
            break
        position_delta_cmd, velocity_cmd, acceleration_cmd = trajectory.sample(elapsed)
        position_cmd = float(position_anchor) + float(position_delta_cmd)
        last_position_cmd = float(position_cmd)
        last_velocity_cmd = float(velocity_cmd)
        last_acceleration_cmd = float(acceleration_cmd)
        if (now - last_send) >= send_interval_s:
            send_command(
                config=config,
                transport=transport,
                rerun_recorder=rerun_recorder,
                target_motor_id=int(target_motor_id),
                target_index=target_index,
                semantic_mode="mit_state",
                command_value=0.0,
                kd_speed=float(config.dynamic_mit.kd),
                position_cmd=float(position_cmd),
                velocity_cmd=float(velocity_cmd),
                kp_cmd=float(config.dynamic_mit.kp),
                torque_ff_cmd=0.0,
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
                phase_name="dynamic_mit",
                stage="dynamic-mit",
            )
            if int(frame.motor_id) != int(target_motor_id):
                continue
            saw_target = True
            abort_event = build_abort_event(
                config=config,
                stage="dynamic-mit",
                group_index=group_index,
                round_index=round_index,
                phase_name="dynamic_mit",
                target_motor_id=target_motor_id,
                frame=frame,
            )
            if abort_event is not None:
                perform_hard_abort(
                    config=config,
                    transport=transport,
                    parser=parser,
                    target_motor_id=target_motor_id,
                    semantic_mode="mit_state",
                )
                raise RuntimeAbortError(abort_event)

            dynamic_abort_event = _build_dynamic_velocity_abort_event(
                config=config,
                stage="dynamic-mit",
                group_index=group_index,
                round_index=round_index,
                phase_name="dynamic_mit",
                target_motor_id=target_motor_id,
                observed_velocity=float(frame.velocity),
            )
            if dynamic_abort_event is not None:
                perform_hard_abort(
                    config=config,
                    transport=transport,
                    parser=parser,
                    target_motor_id=target_motor_id,
                    semantic_mode="mit_state",
                )
                raise RuntimeAbortError(dynamic_abort_event)

            position_error = float(position_cmd) - float(frame.position)
            velocity_error = float(velocity_cmd) - float(frame.velocity)
            tracking_ok = (
                abs(position_error) <= float(config.dynamic_mit.max_position_error)
                and abs(velocity_error) <= float(config.dynamic_mit.max_velocity_error)
            )
            safety_ok = abs(float(frame.velocity)) < float(config.dynamic_mit.velocity_limit)
            state_ok = int(frame.state) in {0, 1}
            saturated = bool(
                np.isfinite(torque_limit)
                and abs(float(frame.torque)) >= float(config.dynamic_mit.saturation_torque_ratio) * float(torque_limit)
            )
            used_for_fit = bool(tracking_ok and safety_ok and state_ok and not saturated and np.isfinite(float(frame.torque)))
            tau_mit_est = (
                float(config.dynamic_mit.kp) * position_error
                + float(config.dynamic_mit.kd) * velocity_error
            )
            bucket = "train" if (elapsed / max(trajectory.duration_s, 1.0e-9)) < float(config.dynamic_mit.train_valid_split) else "valid"
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
                position_cmd=float(position_cmd),
                velocity_cmd=float(velocity_cmd),
                acceleration_cmd=float(acceleration_cmd),
                phase_name=f"dynamic_mit_{bucket}",
                stage="dynamic-mit",
                kp_cmd=float(config.dynamic_mit.kp),
                kd_cmd=float(config.dynamic_mit.kd),
                torque_ff_cmd=0.0,
                tracking_ok=tracking_ok,
                safety_ok=safety_ok,
                state_ok=state_ok,
                saturated=saturated,
                used_for_fit=used_for_fit,
                tau_mit_est=float(tau_mit_est),
            )

        if not saw_target and not saw_chunk:
            time.sleep(max(float(config.transport.read_timeout), 1.0e-3))

    send_command(
        config=config,
        transport=transport,
        rerun_recorder=rerun_recorder,
        target_motor_id=int(target_motor_id),
        target_index=target_index,
        semantic_mode="mit_state",
        command_value=0.0,
        kd_speed=float(config.dynamic_mit.kd),
        position_cmd=float(last_position_cmd),
        velocity_cmd=0.0,
        kp_cmd=float(config.dynamic_mit.kp),
        torque_ff_cmd=0.0,
    )
    _ = last_velocity_cmd, last_acceleration_cmd
    wait_for_stationary(
        config=config,
        transport=transport,
        parser=parser,
        rerun_recorder=rerun_recorder,
        target_motor_id=target_motor_id,
        group_index=group_index,
        round_index=round_index,
        phase_name="dynamic_mit_settle",
        stage="dynamic-mit",
        semantic_mode="mit_state",
        capture_buffer=capture_buffer,
        timeout_s=float(config.transport.sync_timeout),
    )


__all__ = ["DynamicMitTrajectory", "build_dynamic_mit_trajectory", "run_dynamic_mit_phase"]
