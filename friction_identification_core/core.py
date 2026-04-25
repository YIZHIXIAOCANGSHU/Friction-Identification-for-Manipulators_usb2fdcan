from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class FeedbackLike(Protocol):
    motor_id: int
    position: float
    velocity: float
    torque: float
    state: int
    mos_temperature: float


def friction_torque_model(
    velocity: np.ndarray | float,
    *,
    tau_c: float,
    viscous: float,
    tau_bias: float,
) -> np.ndarray:
    velocity_array = np.asarray(velocity, dtype=np.float64)
    return float(tau_c) * np.sign(velocity_array) + float(viscous) * velocity_array + float(tau_bias)


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
class BreakawayIdentificationResult:
    torque_positive: float
    torque_negative: float
    tau_static: float
    tau_bias: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrictionIdentificationResult:
    tau_c: float
    viscous: float
    tau_bias: float
    train_rmse: float
    valid_rmse: float
    train_mask: np.ndarray
    valid_mask: np.ndarray
    torque_pred: np.ndarray
    torque_target: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InertiaIdentificationResult:
    inertia: float
    train_rmse: float
    valid_rmse: float
    train_mask: np.ndarray
    valid_mask: np.ndarray
    torque_pred: np.ndarray
    torque_target: np.ndarray
    filtered_velocity: np.ndarray
    acceleration: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    friction_rmse: float
    inertia_rmse: float
    recommended_for_compensation: bool
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MotorIdentificationResult:
    motor_id: int
    motor_name: str
    breakaway: BreakawayIdentificationResult
    friction: FrictionIdentificationResult
    inertia: InertiaIdentificationResult
    validation: ValidationResult
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    artifacts: tuple[Any, ...]
    summary_paths: Any | None
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
        if self.detail:
            parts.append(f"detail={self.detail}")
        return "Runtime abort: " + ", ".join(parts)


__all__ = [
    "AbortEvent",
    "BreakawayIdentificationResult",
    "FeedbackLike",
    "FrictionIdentificationResult",
    "InertiaIdentificationResult",
    "MotorIdentificationResult",
    "RoundCapture",
    "RunResult",
    "ValidationResult",
    "friction_torque_model",
]
