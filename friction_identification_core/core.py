from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


PIECEWISE_STATIC_LINEAR_KIND = "piecewise_static_linear_v1"


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


def piecewise_static_linear_level(
    speed_abs: np.ndarray | float,
    *,
    tau_static: float,
    tau_c: float,
    static_velocity_threshold_rad_s: float,
    static_transition_velocity_rad_s: float,
) -> np.ndarray:
    speed = np.abs(np.asarray(speed_abs, dtype=np.float64))
    v_static = max(float(static_velocity_threshold_rad_s), 0.0)
    v_transition = max(float(static_transition_velocity_rad_s), v_static + 1.0e-9)
    low_level = np.full(speed.shape, float(tau_static), dtype=np.float64)
    high_level = np.full(speed.shape, float(tau_c), dtype=np.float64)
    blend = np.clip((speed - v_static) / max(v_transition - v_static, 1.0e-9), 0.0, 1.0)
    blended = float(tau_static) + (float(tau_c) - float(tau_static)) * blend
    return np.where(speed <= v_static, low_level, np.where(speed >= v_transition, high_level, blended))


def piecewise_static_linear_torque(
    velocity: np.ndarray | float,
    *,
    acceleration: np.ndarray | float = 0.0,
    direction: np.ndarray | float | None = None,
    tau_static: float,
    tau_c: float,
    viscous: float,
    tau_bias: float,
    inertia: float,
    static_velocity_threshold_rad_s: float,
    static_transition_velocity_rad_s: float,
) -> np.ndarray:
    velocity_array = np.asarray(velocity, dtype=np.float64)
    acceleration_array = np.asarray(acceleration, dtype=np.float64)
    if direction is None:
        direction_array = np.sign(velocity_array)
    else:
        direction_array = np.asarray(direction, dtype=np.float64)
    level = piecewise_static_linear_level(
        np.abs(velocity_array),
        tau_static=float(tau_static),
        tau_c=float(tau_c),
        static_velocity_threshold_rad_s=float(static_velocity_threshold_rad_s),
        static_transition_velocity_rad_s=float(static_transition_velocity_rad_s),
    )
    return (
        float(tau_bias)
        + float(viscous) * velocity_array
        + direction_array * level
        + float(inertia) * acceleration_array
    )


@dataclass(frozen=True)
class IdentificationLimits:
    target_motor_id: int
    motor_tmax: float
    hard_speed_abs: float
    identification_speed_abs: float
    dynamic_mit_velocity_abs: float
    breakaway_scan_torque_abs: float
    compensation_torque_abs: float
    inertia_torque_abs: float

    def to_metadata(self) -> dict[str, float | int]:
        return {
            "target_motor_id": int(self.target_motor_id),
            "motor_tmax": float(self.motor_tmax),
            "hard_speed_abs": float(self.hard_speed_abs),
            "identification_speed_abs": float(self.identification_speed_abs),
            "dynamic_mit_velocity_abs": float(self.dynamic_mit_velocity_abs),
            "breakaway_scan_torque_abs": float(self.breakaway_scan_torque_abs),
            "compensation_torque_abs": float(self.compensation_torque_abs),
            "inertia_torque_abs": float(self.inertia_torque_abs),
        }


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
    filtered_velocity: np.ndarray
    estimated_acceleration: np.ndarray
    friction_term: np.ndarray
    inertia_term: np.ndarray
    guard_scale: np.ndarray
    stiction_evidence: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    kp_cmd: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    kd_cmd: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    torque_ff_cmd: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    position_error: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    velocity_error: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    tracking_ok: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    safety_ok: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    state_ok: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    saturated: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    used_for_fit: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    tau_mit_est: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_count = int(np.asarray(self.time).size)

        def float_default(name: str, fill_value: float = 0.0) -> None:
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.size == 0 and sample_count > 0:
                value = np.full(sample_count, float(fill_value), dtype=np.float64)
            object.__setattr__(self, name, value)

        def bool_default(name: str, fill_value: bool) -> None:
            value = np.asarray(getattr(self, name), dtype=bool)
            if value.size == 0 and sample_count > 0:
                value = np.full(sample_count, bool(fill_value), dtype=bool)
            object.__setattr__(self, name, value)

        for field_name in ("kp_cmd", "kd_cmd", "torque_ff_cmd", "position_error", "velocity_error", "tau_mit_est"):
            float_default(field_name, 0.0)
        bool_default("tracking_ok", True)
        bool_default("safety_ok", True)
        bool_default("state_ok", True)
        bool_default("saturated", False)
        bool_default("used_for_fit", False)
        bool_default("stiction_evidence", False)

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
class DynamicMotorFitResult:
    inertia: float
    viscous: float
    tau_c: float
    tau_bias: float
    train_rmse: float
    valid_rmse: float
    train_mask: np.ndarray
    valid_mask: np.ndarray
    torque_pred: np.ndarray
    torque_target: np.ndarray
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
class IdentifiedMotorModel:
    motor_id: int
    motor_name: str
    identified_at: str
    source_run_label: str
    tau_static: float
    tau_bias: float
    tau_c: float
    viscous: float
    inertia: float
    friction_validation_rmse: float
    inertia_validation_rmse: float
    repeat_consistency_score: float
    recommended_for_compensation: bool
    model_version: str = "1.0"
    model_kind: str = PIECEWISE_STATIC_LINEAR_KIND
    fit_method: str = "robust_constrained_ls"
    source_phases: tuple[str, ...] = ("breakaway", "speed-hold", "inertia")
    accepted_round_count: int = 0
    selected_rounds: tuple[int, ...] = ()
    confidence: float = 0.0
    quality_flags: tuple[str, ...] = ()
    publish_status: str = "not_published"
    publish_detail: str = ""
    friction_model: dict[str, Any] | None = None
    export_models: dict[str, Any] | None = None

    @staticmethod
    def from_latest_entry(entry: dict[str, Any]) -> "IdentifiedMotorModel":
        selected_rounds_raw = entry.get("selected_rounds", ())
        if not isinstance(selected_rounds_raw, (list, tuple)):
            selected_rounds_raw = ()
        source_phases_raw = entry.get("source_phases", ("breakaway", "speed-hold", "inertia"))
        if not isinstance(source_phases_raw, (list, tuple)):
            source_phases_raw = ("breakaway", "speed-hold", "inertia")
        quality_flags_raw = entry.get("quality_flags", ())
        if not isinstance(quality_flags_raw, (list, tuple)):
            quality_flags_raw = ()
        publish_status = str(
            entry.get(
                "publish_status",
                "published" if bool(entry.get("recommended_for_compensation", False)) else "not_published",
            )
        )
        return IdentifiedMotorModel(
            motor_id=int(entry["motor_id"]),
            motor_name=str(entry["motor_name"]),
            identified_at=str(entry["identified_at"]),
            source_run_label=str(entry["source_run_label"]),
            tau_static=float(entry["tau_static"]),
            tau_bias=float(entry["tau_bias"]),
            tau_c=float(entry["tau_c"]),
            viscous=float(entry["viscous"]),
            inertia=float(entry["inertia"]),
            friction_validation_rmse=float(entry["friction_validation_rmse"]),
            inertia_validation_rmse=float(entry["inertia_validation_rmse"]),
            repeat_consistency_score=float(entry["repeat_consistency_score"]),
            recommended_for_compensation=bool(entry["recommended_for_compensation"]),
            model_version=str(entry.get("model_version", "legacy_static")),
            model_kind=str(entry.get("model_kind", PIECEWISE_STATIC_LINEAR_KIND)),
            fit_method=str(entry.get("fit_method", "legacy_lstsq")),
            source_phases=tuple(str(item) for item in source_phases_raw),
            accepted_round_count=int(entry.get("accepted_round_count", len(selected_rounds_raw))),
            selected_rounds=tuple(int(item) for item in selected_rounds_raw),
            confidence=float(entry.get("confidence", 1.0 if bool(entry.get("recommended_for_compensation", False)) else 0.0)),
            quality_flags=tuple(str(item) for item in quality_flags_raw),
            publish_status=publish_status,
            publish_detail=str(entry.get("publish_detail", "legacy entry")),
            friction_model=entry.get("friction_model") if isinstance(entry.get("friction_model"), dict) else None,
            export_models=entry.get("export_models") if isinstance(entry.get("export_models"), dict) else None,
        )

    def to_latest_entry(self) -> dict[str, Any]:
        entry = {
            "motor_id": int(self.motor_id),
            "motor_name": str(self.motor_name),
            "identified_at": str(self.identified_at),
            "source_run_label": str(self.source_run_label),
            "tau_static": float(self.tau_static),
            "tau_bias": float(self.tau_bias),
            "tau_c": float(self.tau_c),
            "viscous": float(self.viscous),
            "inertia": float(self.inertia),
            "friction_validation_rmse": float(self.friction_validation_rmse),
            "inertia_validation_rmse": float(self.inertia_validation_rmse),
            "repeat_consistency_score": float(self.repeat_consistency_score),
            "recommended_for_compensation": bool(self.recommended_for_compensation),
            "model_version": str(self.model_version),
            "model_kind": str(self.model_kind),
            "fit_method": str(self.fit_method),
            "source_phases": list(self.source_phases),
            "accepted_round_count": int(self.accepted_round_count),
            "selected_rounds": list(self.selected_rounds),
            "confidence": float(self.confidence),
            "quality_flags": list(self.quality_flags),
            "publish_status": str(self.publish_status),
            "publish_detail": str(self.publish_detail),
        }
        if isinstance(self.friction_model, dict):
            entry["friction_model"] = dict(self.friction_model)
        if isinstance(self.export_models, dict):
            entry["export_models"] = dict(self.export_models)
        return entry


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
    "DynamicMotorFitResult",
    "FeedbackLike",
    "FrictionIdentificationResult",
    "IdentifiedMotorModel",
    "IdentificationLimits",
    "InertiaIdentificationResult",
    "MotorIdentificationResult",
    "PIECEWISE_STATIC_LINEAR_KIND",
    "RoundCapture",
    "RunResult",
    "ValidationResult",
    "friction_torque_model",
    "piecewise_static_linear_level",
    "piecewise_static_linear_torque",
]
