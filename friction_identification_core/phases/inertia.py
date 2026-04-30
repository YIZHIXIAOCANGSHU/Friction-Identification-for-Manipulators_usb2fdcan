from __future__ import annotations

from friction_identification_core.capture import CaptureBuffer, log_stage_transition
from friction_identification_core.io import CommandTransport, FeedbackFrameParser
from friction_identification_core.limits import identification_limits_for_motor, validate_abs_less_than
from friction_identification_core.runtime_config import Config
from friction_identification_core.safety import wait_for_stationary
from friction_identification_core.visualization import RerunRecorder
from friction_identification_core.phases.speed_hold import run_velocity_segment


def run_inertia_phase(
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
    log_stage_transition("inertia", target_motor_id=target_motor_id)
    rerun_recorder.log_phase_event(motor_id=int(target_motor_id), phase_name="inertia", detail="start")
    target_index = config.motor_index(target_motor_id)
    kd_speed = float(config.mit_velocity.kd_speed[target_index])
    ramp_acceleration = float(config.mit_velocity.ramp_acceleration)
    waypoints = [float(item) for item in config.inertia.waypoints]
    if len(waypoints) < 2:
        raise ValueError("inertia.waypoints must contain at least 2 values.")
    limits = identification_limits_for_motor(config, target_motor_id=int(target_motor_id))
    validate_abs_less_than(
        waypoints,
        limit_abs=float(limits.identification_speed_abs),
        name="inertia.waypoints",
    )
    if abs(float(waypoints[0])) > 1.0e-9 or abs(float(waypoints[-1])) > 1.0e-9:
        raise ValueError("inertia.waypoints must start and end at 0.")
    current_velocity = float(waypoints[0])
    midpoint = max((len(waypoints) - 1) // 2, 1)
    for segment_index, target_velocity in enumerate(waypoints[1:], start=1):
        bucket = "train" if segment_index <= midpoint else "valid"
        current_velocity = run_velocity_segment(
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
    wait_for_stationary(
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


__all__ = ["run_inertia_phase"]
