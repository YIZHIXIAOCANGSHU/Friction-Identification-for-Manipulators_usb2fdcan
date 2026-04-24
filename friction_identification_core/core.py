from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from friction_identification_core.config import Config, ExcitationConfig

if TYPE_CHECKING:
    from friction_identification_core.results import RoundArtifact, SummaryPaths


class FeedbackLike(Protocol):
    motor_id: int
    position: float
    velocity: float
    torque: float


@dataclass(frozen=True)
class ReferenceSample:
    position_cmd: float
    velocity_cmd: float
    acceleration_cmd: float
    phase_name: str


@dataclass(frozen=True)
class ReferenceTrajectory:
    time: np.ndarray
    position_cmd: np.ndarray
    velocity_cmd: np.ndarray
    acceleration_cmd: np.ndarray
    phase_name: np.ndarray
    duration_s: float

    def index_at(self, elapsed_s: float) -> int:
        if self.time.size == 0:
            return 0
        index = int(np.searchsorted(self.time, float(elapsed_s), side="right") - 1)
        return min(max(index, 0), int(self.time.size) - 1)

    def sample(self, elapsed_s: float) -> ReferenceSample:
        if self.time.size == 0:
            return ReferenceSample(0.0, 0.0, 0.0, "empty")
        index = self.index_at(elapsed_s)
        return ReferenceSample(
            position_cmd=float(self.position_cmd[index]),
            velocity_cmd=float(self.velocity_cmd[index]),
            acceleration_cmd=float(self.acceleration_cmd[index]),
            phase_name=str(self.phase_name[index]),
        )


@dataclass(frozen=True)
class RoundCapture:
    group_index: int
    round_index: int
    target_motor_id: int
    motor_name: str
    time: np.ndarray
    motor_id: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    torque_feedback: np.ndarray
    command_raw: np.ndarray
    command: np.ndarray
    position_cmd: np.ndarray
    velocity_cmd: np.ndarray
    acceleration_cmd: np.ndarray
    phase_name: np.ndarray
    state: np.ndarray
    mos_temperature: np.ndarray
    id_match_ok: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return int(self.time.size)


@dataclass(frozen=True)
class MotorCompensationParameters:
    motor_id: int
    motor_name: str
    coulomb: float
    viscous: float
    offset: float
    velocity_scale: float

    def feedforward_torque(self, velocity: float) -> float:
        scale = max(float(self.velocity_scale), 1.0e-6)
        velocity = float(velocity)
        return float(
            float(self.coulomb) * np.tanh(velocity / scale)
            + float(self.viscous) * velocity
            + float(self.offset)
        )


@dataclass(frozen=True)
class MotorIdentificationResult:
    motor_id: int
    motor_name: str
    identified: bool
    coulomb: float
    viscous: float
    offset: float
    velocity_scale: float
    torque_pred: np.ndarray
    torque_target: np.ndarray
    sample_mask: np.ndarray
    identification_window_mask: np.ndarray
    tracking_ok_mask: np.ndarray
    saturation_ok_mask: np.ndarray
    train_mask: np.ndarray
    valid_mask: np.ndarray
    train_rmse: float
    valid_rmse: float
    train_r2: float
    valid_r2: float
    valid_sample_ratio: float
    sample_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MotorDynamicIdentificationResult:
    motor_id: int
    motor_name: str
    identified: bool
    fc: float
    fs: float
    vs: float
    sigma0: float
    sigma1: float
    sigma2: float
    offset: float
    torque_pred: np.ndarray
    torque_target: np.ndarray
    sample_mask: np.ndarray
    train_mask: np.ndarray
    valid_mask: np.ndarray
    validation_warmup_mask: np.ndarray
    train_rmse: float
    valid_rmse: float
    train_r2: float
    valid_r2: float
    valid_sample_ratio: float
    sample_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    artifacts: tuple["RoundArtifact", ...]
    summary_paths: "SummaryPaths | None"
    manifest_path: Path


@dataclass(frozen=True)
class AbortEvent:
    reason: str
    stage: str
    motor_id: int
    group_index: int
    round_index: int
    phase_name: str
    observed_velocity: float | None = None
    velocity_limit: float | None = None
    feedback_torque: float | None = None
    torque_limit: float | None = None
    feedback_position: float | None = None
    position_limit: float | None = None
    detail: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "reason": str(self.reason),
            "stage": str(self.stage),
            "motor_id": int(self.motor_id),
            "group_index": int(self.group_index),
            "round_index": int(self.round_index),
            "phase_name": str(self.phase_name),
        }
        if self.observed_velocity is not None:
            payload["observed_velocity"] = float(self.observed_velocity)
        if self.velocity_limit is not None:
            payload["velocity_limit"] = float(self.velocity_limit)
        if self.feedback_torque is not None:
            payload["feedback_torque"] = float(self.feedback_torque)
        if self.torque_limit is not None:
            payload["torque_limit"] = float(self.torque_limit)
        if self.feedback_position is not None:
            payload["feedback_position"] = float(self.feedback_position)
        if self.position_limit is not None:
            payload["position_limit"] = float(self.position_limit)
        if self.detail:
            payload["detail"] = str(self.detail)
        return payload

    def error_message(self) -> str:
        parts = [
            f"reason={self.reason}",
            f"stage={self.stage}",
            f"motor_id={self.motor_id}",
            f"group_index={self.group_index}",
            f"round_index={self.round_index}",
            f"phase_name={self.phase_name}",
        ]
        if self.observed_velocity is not None:
            parts.append(f"observed_velocity={self.observed_velocity:.6f}")
        if self.velocity_limit is not None:
            parts.append(f"velocity_limit={self.velocity_limit:.6f}")
        if self.feedback_torque is not None:
            parts.append(f"feedback_torque={self.feedback_torque:.6f}")
        if self.torque_limit is not None:
            parts.append(f"torque_limit={self.torque_limit:.6f}")
        if self.feedback_position is not None:
            parts.append(f"feedback_position={self.feedback_position:.6f}")
        if self.position_limit is not None:
            parts.append(f"position_limit={self.position_limit:.6f}")
        if self.detail:
            parts.append(f"detail={self.detail}")
        return "Runtime abort: " + ", ".join(parts)


@dataclass(frozen=True)
class ZeroingLockState:
    success_count: int
    inside_entry_band: bool
    inside_exit_band: bool


def controller_update(
    config: Config,
    motor_id: int,
    reference: ReferenceSample,
    feedback: FeedbackLike,
    *,
    compensation: MotorCompensationParameters | None = None,
    position_gain: float | None = None,
    velocity_gain: float | None = None,
) -> tuple[float, float]:
    index = config.motor_index(motor_id)
    max_torque = float(config.control.max_torque[index])
    if compensation is None:
        if position_gain is None:
            position_gain = float(config.control.position_gain[index])
        if velocity_gain is None:
            velocity_gain = float(config.control.velocity_gain[index])
        position_error = float(reference.position_cmd) - float(feedback.position)
        velocity_error = float(reference.velocity_cmd) - float(feedback.velocity)
        raw_command = float(position_gain) * position_error + float(velocity_gain) * velocity_error
    else:
        raw_command = compensation.feedforward_torque(float(feedback.velocity))
    limited_command = float(np.clip(raw_command, -max_torque, max_torque))
    return float(raw_command), limited_command


class SingleMotorController:
    def __init__(self, config: Config) -> None:
        self._config = config

    def update(
        self,
        motor_id: int,
        reference: ReferenceSample,
        feedback: FeedbackLike,
        *,
        compensation: MotorCompensationParameters | None = None,
        position_gain: float | None = None,
        velocity_gain: float | None = None,
    ) -> tuple[float, float]:
        return controller_update(
            self._config,
            motor_id,
            reference,
            feedback,
            compensation=compensation,
            position_gain=position_gain,
            velocity_gain=velocity_gain,
        )


def _schroeder_phases(harmonic_count: int) -> np.ndarray:
    indices = np.arange(harmonic_count, dtype=np.float64)
    return -np.pi * indices * (indices - 1.0) / max(float(harmonic_count), 1.0)


def _excitation_envelope(
    time: np.ndarray,
    *,
    fade_in_duration: float,
    steady_duration: float,
    fade_out_duration: float,
) -> np.ndarray:
    envelope = np.ones_like(time, dtype=np.float64)
    if fade_in_duration > 0.0:
        fade_in_mask = time < fade_in_duration
        u = np.clip(time[fade_in_mask] / fade_in_duration, 0.0, 1.0)
        envelope[fade_in_mask] = 0.5 - 0.5 * np.cos(np.pi * u)
    if fade_out_duration > 0.0:
        fade_out_start = fade_in_duration + steady_duration
        fade_out_mask = time >= fade_out_start
        u = np.clip((time[fade_out_mask] - fade_out_start) / fade_out_duration, 0.0, 1.0)
        envelope[fade_out_mask] = 0.5 + 0.5 * np.cos(np.pi * u)
    return np.clip(envelope, 0.0, 1.0)


def build_reference_trajectory(config: ExcitationConfig, *, max_velocity: float) -> ReferenceTrajectory:
    sample_rate = max(float(config.sample_rate), 1.0)
    dt = 1.0 / sample_rate
    cycle_duration = 1.0 / float(config.base_frequency)
    fade_in_duration = float(config.fade_in_cycles) * cycle_duration
    steady_duration = float(config.steady_cycles) * cycle_duration
    fade_out_duration = float(config.fade_out_cycles) * cycle_duration
    excitation_duration = fade_in_duration + steady_duration + fade_out_duration
    total_duration = float(config.hold_start) + excitation_duration + float(config.hold_end)
    sample_count = max(int(np.ceil(total_duration * sample_rate - 1.0e-9)), 2)

    time = np.arange(sample_count, dtype=np.float64) * dt
    position_cmd = np.zeros(sample_count, dtype=np.float64)
    velocity_cmd = np.zeros(sample_count, dtype=np.float64)
    acceleration_cmd = np.zeros(sample_count, dtype=np.float64)
    phase_name = np.full(sample_count, "hold_end", dtype="<U32")

    hold_start_end = float(config.hold_start)
    excitation_end = hold_start_end + excitation_duration
    hold_start_mask = time < hold_start_end
    hold_end_mask = time >= excitation_end
    excitation_mask = (~hold_start_mask) & (~hold_end_mask)

    phase_name[hold_start_mask] = "hold_start"
    phase_name[hold_end_mask] = "hold_end"

    if np.any(excitation_mask):
        excitation_time = time[excitation_mask] - hold_start_end
        envelope = _excitation_envelope(
            excitation_time,
            fade_in_duration=fade_in_duration,
            steady_duration=steady_duration,
            fade_out_duration=fade_out_duration,
        )
        phases = _schroeder_phases(len(config.harmonic_multipliers))
        q_raw = np.zeros(excitation_time.size, dtype=np.float64)
        for multiplier, weight, phase in zip(config.harmonic_multipliers, config.harmonic_weights, phases):
            omega = 2.0 * np.pi * float(multiplier) * float(config.base_frequency)
            q_raw += float(weight) * np.sin(omega * excitation_time + float(phase))
        q_unit = envelope * q_raw
        if np.any(envelope > 0.0):
            q_unit -= float(np.mean(q_unit[envelope > 0.0])) * envelope
        v_unit = np.gradient(q_unit, dt)
        a_unit = np.gradient(v_unit, dt)

        max_abs_position = max(float(np.max(np.abs(q_unit))), 1.0e-9)
        max_abs_velocity = max(float(np.max(np.abs(v_unit))), 1.0e-9)
        scale = min(
            float(config.position_limit) / max_abs_position,
            float(config.velocity_utilization) * float(max_velocity) / max_abs_velocity,
        )

        position_cmd[excitation_mask] = scale * q_unit
        velocity_cmd[excitation_mask] = scale * v_unit
        acceleration_cmd[excitation_mask] = scale * a_unit

        fade_in_end = fade_in_duration
        steady_end = fade_in_duration + steady_duration
        fade_out_end = fade_in_duration + steady_duration + fade_out_duration
        for local_index, t_exc in zip(np.flatnonzero(excitation_mask), excitation_time):
            if t_exc < fade_in_end:
                phase_name[local_index] = "fade_in"
                continue
            if t_exc < steady_end:
                cycle_index = int(np.floor((t_exc - fade_in_duration) / cycle_duration)) + 1
                cycle_index = min(max(cycle_index, 1), int(config.steady_cycles))
                phase_name[local_index] = f"excitation_cycle_{cycle_index:02d}"
                continue
            if t_exc < fade_out_end:
                phase_name[local_index] = "fade_out"

    position_max = float(np.max(np.abs(position_cmd)))
    velocity_max = float(np.max(np.abs(velocity_cmd)))
    velocity_limit = float(config.velocity_utilization) * float(max_velocity)
    if position_max > float(config.position_limit) + 1.0e-9:
        raise ValueError("Reference trajectory exceeds excitation.position_limit.")
    if velocity_max > velocity_limit + 1.0e-9:
        raise ValueError("Reference trajectory exceeds excitation.velocity_utilization * control.max_velocity.")

    return ReferenceTrajectory(
        time=time,
        position_cmd=position_cmd,
        velocity_cmd=velocity_cmd,
        acceleration_cmd=acceleration_cmd,
        phase_name=phase_name,
        duration_s=float(total_duration),
    )


def theoretical_velocity_for_phase(
    reference: ReferenceTrajectory,
    *,
    phase_name: str,
    feedback_position: float,
    reference_index: int,
    zero_target_velocity_threshold: float,
) -> tuple[float, int]:
    phase_mask = np.asarray(reference.phase_name).astype(str) == str(phase_name)
    candidate_indices = np.flatnonzero(phase_mask)
    if candidate_indices.size == 0:
        return float(reference.velocity_cmd[reference_index]), int(reference_index)

    reference_velocity = float(reference.velocity_cmd[reference_index])
    if abs(reference_velocity) > float(zero_target_velocity_threshold):
        sign_mask = np.sign(reference.velocity_cmd[candidate_indices]) == np.sign(reference_velocity)
        if np.any(sign_mask):
            candidate_indices = candidate_indices[sign_mask]

    candidate_positions = np.asarray(reference.position_cmd[candidate_indices], dtype=np.float64)
    distances = np.abs(candidate_positions - float(feedback_position))
    min_distance = float(np.min(distances))
    closest_mask = np.isclose(distances, min_distance, rtol=0.0, atol=1.0e-12)
    closest_indices = candidate_indices[closest_mask]
    if closest_indices.size > 1:
        closest_indices = closest_indices[np.argsort(np.abs(closest_indices - int(reference_index)))]
    matched_index = int(closest_indices[0])
    return float(reference.velocity_cmd[matched_index]), matched_index


def velocity_abort_limit_for_phase(
    *,
    phase_name: str,
    theoretical_velocity: float,
    low_speed_abort_limit: float,
    speed_abort_ratio: float,
) -> float:
    if not str(phase_name).startswith("excitation_cycle_"):
        return float("inf")
    return max(float(low_speed_abort_limit), float(speed_abort_ratio) * abs(float(theoretical_velocity)))


def zeroing_theoretical_velocity_from_position(
    *,
    filtered_position: float,
    zeroing_position_gain: float,
    zeroing_velocity_gain: float,
    zeroing_hard_velocity_limit: float,
) -> float:
    if float(zeroing_position_gain) <= 1.0e-9:
        return 0.0
    if float(zeroing_velocity_gain) <= 1.0e-9:
        return float(zeroing_hard_velocity_limit)
    return min(
        float(zeroing_hard_velocity_limit),
        abs(float(zeroing_position_gain) * float(filtered_position) / float(zeroing_velocity_gain)),
    )


def zeroing_lock_state_update(
    *,
    success_count: int,
    filtered_position: float,
    filtered_velocity: float,
    zeroing_theoretical_velocity: float,
    position_tolerance: float,
    velocity_tolerance: float,
    zeroing_velocity_limit: float,
) -> ZeroingLockState:
    inside_entry = (
        float(zeroing_theoretical_velocity) <= float(zeroing_velocity_limit)
        and abs(float(filtered_position)) <= float(position_tolerance)
        and abs(float(filtered_velocity)) <= float(velocity_tolerance)
    )
    outside_exit = (
        abs(float(filtered_position)) > 1.25 * float(position_tolerance)
        or abs(float(filtered_velocity)) > 1.25 * float(velocity_tolerance)
    )
    next_success_count = int(success_count)
    if inside_entry:
        next_success_count += 1
    elif next_success_count > 0 and outside_exit:
        next_success_count = 0
    return ZeroingLockState(
        success_count=int(next_success_count),
        inside_entry_band=bool(inside_entry),
        inside_exit_band=not bool(outside_exit),
    )


def runtime_abort_from_frame(
    *,
    frame: FeedbackLike,
    stage: str,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    phase_name: str,
    velocity_limit: float,
    torque_limit: float,
    position_limit: float,
) -> AbortEvent | None:
    if np.isfinite(float(velocity_limit)) and abs(float(frame.velocity)) > float(velocity_limit):
        return AbortEvent(
            reason="velocity_limit_exceeded",
            stage=stage,
            motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
            phase_name=str(phase_name),
            observed_velocity=float(frame.velocity),
            velocity_limit=float(velocity_limit),
        )
    if abs(float(frame.torque)) > float(torque_limit):
        return AbortEvent(
            reason="torque_limit_exceeded",
            stage=stage,
            motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
            phase_name=str(phase_name),
            feedback_torque=float(frame.torque),
            torque_limit=float(torque_limit),
        )
    effective_position_limit = float(position_limit)
    if np.isfinite(effective_position_limit) and effective_position_limit > 0.0:
        if abs(float(frame.position)) > 1.10 * effective_position_limit:
            return AbortEvent(
                reason="position_limit_exceeded",
                stage=stage,
                motor_id=int(target_motor_id),
                group_index=int(group_index),
                round_index=int(round_index),
                phase_name=str(phase_name),
                feedback_position=float(frame.position),
                position_limit=float(1.10 * effective_position_limit),
            )
    return None


def safety_margin_text(
    *,
    velocity_limit: float,
    observed_velocity: float,
    torque_limit: float,
    feedback_torque: float,
    position_limit: float,
    feedback_position: float,
) -> str:
    return (
        f"velocity_margin={velocity_limit - abs(float(observed_velocity)):+.6f}, "
        f"torque_margin={torque_limit - abs(float(feedback_torque)):+.6f}, "
        f"position_margin={position_limit - abs(float(feedback_position)):+.6f}"
    )


__all__ = [
    "AbortEvent",
    "FeedbackLike",
    "MotorCompensationParameters",
    "MotorDynamicIdentificationResult",
    "MotorIdentificationResult",
    "ReferenceSample",
    "ReferenceTrajectory",
    "RoundCapture",
    "RunResult",
    "SingleMotorController",
    "ZeroingLockState",
    "build_reference_trajectory",
    "controller_update",
    "runtime_abort_from_frame",
    "safety_margin_text",
    "theoretical_velocity_for_phase",
    "velocity_abort_limit_for_phase",
    "zeroing_lock_state_update",
    "zeroing_theoretical_velocity_from_position",
]
