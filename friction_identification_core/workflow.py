from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from friction_identification_core.core import (
    BreakawayIdentificationResult,
    DynamicMotorFitResult,
    FrictionIdentificationResult,
    InertiaIdentificationResult,
    MotorIdentificationResult,
    PIECEWISE_STATIC_LINEAR_KIND,
    RoundCapture,
    RunResult,
    ValidationResult,
)
from friction_identification_core.capture import CaptureBuffer
from friction_identification_core.compensation import (
    load_compensation_parameters,
    run_compensation_phase as run_compensation_phase_module,
)
from friction_identification_core.identification import (
    estimate_filtered_velocity_and_acceleration,
    fit_dynamic_motor_model,
    fit_friction_model,
    fit_inertia_model,
    fit_weighted_dynamic_motor_model,
)
from friction_identification_core.io import (
    CommandTransport,
    FeedbackFrameParser,
    open_transport,
)
from friction_identification_core.results import (
    ResultStore,
    RoundArtifact,
    latest_parameters_path,
    log_info,
)
from friction_identification_core.runtime_config import Config
from friction_identification_core.safety import (
    RuntimeAbortError,
    precheck_transport,
    send_zero_then_disable,
)
from friction_identification_core.phases.breakaway import run_breakaway_phase
from friction_identification_core.phases.dynamic_mit import run_dynamic_mit_phase
from friction_identification_core.phases.inertia import run_inertia_phase
from friction_identification_core.phases.low_speed import run_low_speed_characterization_phase
from friction_identification_core.phases.speed_hold import run_speed_hold_phase
from friction_identification_core.visualization import RerunRecorder

@dataclass(frozen=True)
class _SpeedHoldPlatformStat:
    phase_name: str
    bucket: str
    commanded_velocity: float
    mean_velocity: float
    velocity_std: float
    mean_torque: float
    sample_count: int
    tracking_ratio: float
    velocity_std_ratio: float
    accepted: bool
    rejection_reason: str


def _parse_speed_hold_command_velocity(phase_name: str) -> float:
    try:
        return float(str(phase_name).rsplit("_", 1)[-1])
    except ValueError:
        return float("nan")


def _collect_speed_hold_platform_stats(capture: RoundCapture, config: Config) -> list[_SpeedHoldPlatformStat]:
    phase_names = np.asarray(capture.phase_name).astype(str)
    ordered_phase_names = list(dict.fromkeys(phase_names.tolist()))
    ratio = float(config.mit_velocity.steady_window_ratio)
    platforms: list[_SpeedHoldPlatformStat] = []
    for phase_name in ordered_phase_names:
        if not str(phase_name).startswith("speed_hold_"):
            continue
        if str(phase_name).startswith("speed_hold_train_"):
            bucket = "train"
        elif str(phase_name).startswith("speed_hold_valid_"):
            bucket = "valid"
        else:
            continue

        indices = np.flatnonzero(phase_names == phase_name)
        if indices.size == 0:
            continue
        start_index = int(np.floor((1.0 - ratio) * indices.size))
        selected = indices[start_index:]
        commanded_velocity = _parse_speed_hold_command_velocity(phase_name)
        mean_velocity = float(np.nanmean(capture.velocity[selected])) if selected.size else float("nan")
        velocity_std = float(np.nanstd(capture.velocity[selected])) if selected.size else float("nan")
        mean_torque = float(np.nanmean(capture.torque_feedback[selected])) if selected.size else float("nan")
        tracking_ratio = (
            abs(mean_velocity) / max(abs(commanded_velocity), 1.0e-6)
            if np.isfinite(commanded_velocity)
            else float("nan")
        )
        velocity_std_ratio = (
            velocity_std / max(abs(commanded_velocity), 1.0e-6)
            if np.isfinite(commanded_velocity)
            else float("nan")
        )
        direction_ok = bool(
            np.isfinite(commanded_velocity)
            and np.isfinite(mean_velocity)
            and abs(mean_velocity) > 1.0e-6
            and np.sign(mean_velocity) == np.sign(commanded_velocity)
        )
        accepted = True
        rejection_reason = ""
        if selected.size == 0:
            accepted = False
            rejection_reason = "empty_platform"
        elif selected.size < int(config.identification.min_platform_sample_count):
            accepted = False
            rejection_reason = (
                f"sample_count_below_min:{int(selected.size)}"
                f"<{int(config.identification.min_platform_sample_count)}"
            )
        elif not np.isfinite(commanded_velocity):
            accepted = False
            rejection_reason = "invalid_command_velocity"
        elif not np.isfinite(mean_velocity) or not np.isfinite(mean_torque):
            accepted = False
            rejection_reason = "non_finite_platform_statistics"
        elif not direction_ok:
            accepted = False
            rejection_reason = "direction_mismatch"
        elif tracking_ratio < float(config.identification.min_tracking_ratio):
            accepted = False
            rejection_reason = (
                f"tracking_ratio_below_min:{tracking_ratio:.3f}"
                f"<{float(config.identification.min_tracking_ratio):.3f}"
            )
        elif velocity_std_ratio > float(config.identification.max_steady_velocity_std_ratio):
            accepted = False
            rejection_reason = (
                f"velocity_std_ratio_above_max:{velocity_std_ratio:.3f}"
                f">{float(config.identification.max_steady_velocity_std_ratio):.3f}"
            )

        platforms.append(
            _SpeedHoldPlatformStat(
                phase_name=str(phase_name),
                bucket=str(bucket),
                commanded_velocity=float(commanded_velocity),
                mean_velocity=float(mean_velocity),
                velocity_std=float(velocity_std),
                mean_torque=float(mean_torque),
                sample_count=int(selected.size),
                tracking_ratio=float(tracking_ratio),
                velocity_std_ratio=float(velocity_std_ratio),
                accepted=bool(accepted),
                rejection_reason=str(rejection_reason),
            )
        )
    return platforms


def _annotate_friction_result(
    result: FrictionIdentificationResult,
    *,
    platforms: list[_SpeedHoldPlatformStat],
) -> FrictionIdentificationResult:
    metadata = dict(result.metadata)
    metadata["platforms"] = [
        {
            "phase_name": platform.phase_name,
            "bucket": platform.bucket,
            "commanded_velocity": float(platform.commanded_velocity),
            "mean_velocity": float(platform.mean_velocity),
            "velocity_std": float(platform.velocity_std),
            "mean_torque": float(platform.mean_torque),
            "sample_count": int(platform.sample_count),
            "tracking_ratio": float(platform.tracking_ratio),
            "velocity_std_ratio": float(platform.velocity_std_ratio),
            "accepted": bool(platform.accepted),
            "rejection_reason": str(platform.rejection_reason),
        }
        for platform in platforms
    ]
    metadata["accepted_train_platform_count"] = int(
        sum(1 for platform in platforms if platform.accepted and platform.bucket == "train")
    )
    metadata["accepted_valid_platform_count"] = int(
        sum(1 for platform in platforms if platform.accepted and platform.bucket == "valid")
    )
    metadata["rejected_platform_count"] = int(sum(1 for platform in platforms if not platform.accepted))
    return FrictionIdentificationResult(
        tau_c=float(result.tau_c),
        viscous=float(result.viscous),
        tau_bias=float(result.tau_bias),
        train_rmse=float(result.train_rmse),
        valid_rmse=float(result.valid_rmse),
        train_mask=np.asarray(result.train_mask, dtype=bool),
        valid_mask=np.asarray(result.valid_mask, dtype=bool),
        torque_pred=np.asarray(result.torque_pred, dtype=np.float64),
        torque_target=np.asarray(result.torque_target, dtype=np.float64),
        metadata=metadata,
    )


def _build_round_validation_result(
    friction_result: FrictionIdentificationResult,
    inertia_result: InertiaIdentificationResult,
    config: Config,
) -> ValidationResult:
    accepted_train_platform_count = int(friction_result.metadata.get("accepted_train_platform_count", 0))
    accepted_valid_platform_count = int(friction_result.metadata.get("accepted_valid_platform_count", 0))
    rejected_platform_count = int(friction_result.metadata.get("rejected_platform_count", 0))
    friction_rmse = float(friction_result.valid_rmse)
    inertia_rmse = float(inertia_result.valid_rmse)
    reasons: list[str] = []

    if accepted_train_platform_count < 2:
        reasons.append(f"accepted_train_platform_count={accepted_train_platform_count}<2")
    if accepted_valid_platform_count < 2:
        reasons.append(f"accepted_valid_platform_count={accepted_valid_platform_count}<2")
    if not np.isfinite(float(friction_result.tau_c)) or float(friction_result.tau_c) < 0.0:
        reasons.append(f"invalid_tau_c={float(friction_result.tau_c):+.6f}")
    if not np.isfinite(float(friction_result.viscous)) or float(friction_result.viscous) < 0.0:
        reasons.append(f"invalid_viscous={float(friction_result.viscous):+.6f}")
    if not np.isfinite(float(inertia_result.inertia)) or float(inertia_result.inertia) < 0.0:
        reasons.append(f"invalid_inertia={float(inertia_result.inertia):+.6f}")
    friction_threshold = float(config.identification.friction_rmse_publish_threshold)
    inertia_threshold = float(config.identification.inertia_rmse_publish_threshold)
    if not np.isfinite(friction_rmse) or friction_rmse > friction_threshold:
        reasons.append(f"friction_rmse={friction_rmse:.6f}>{friction_threshold:.6f}")
    if not np.isfinite(inertia_rmse) or inertia_rmse > inertia_threshold:
        reasons.append(f"inertia_rmse={inertia_rmse:.6f}>{inertia_threshold:.6f}")

    recommended = not reasons
    status = "accepted" if recommended else "rejected"
    detail = (
        f"friction_rmse={friction_rmse:.6f}, inertia_rmse={inertia_rmse:.6f}"
        if recommended
        else "; ".join(reasons)
    )
    return ValidationResult(
        friction_rmse=friction_rmse,
        inertia_rmse=inertia_rmse,
        recommended_for_compensation=bool(recommended),
        detail=detail,
        metadata={
            "status": status,
            "accepted_train_platform_count": accepted_train_platform_count,
            "accepted_valid_platform_count": accepted_valid_platform_count,
            "rejected_platform_count": rejected_platform_count,
            "reasons": list(reasons),
        },
    )


def _breakaway_scan_limit_status(breakaway_result: BreakawayIdentificationResult) -> dict[str, object]:
    metadata = dict(breakaway_result.metadata)
    scan_limit = float(metadata.get("scan_max_torque", float("nan")))
    torque_step = float(metadata.get("torque_step", 0.0))
    tolerance = max(0.5 * torque_step, 1.0e-9)
    positive = bool(metadata.get("positive_scan_limit_reached", False))
    negative = bool(metadata.get("negative_scan_limit_reached", False))
    if np.isfinite(scan_limit) and scan_limit > 0.0:
        if not positive and np.isfinite(float(breakaway_result.torque_positive)):
            positive = bool(np.isclose(abs(float(breakaway_result.torque_positive)), scan_limit, atol=tolerance, rtol=0.0))
        if not negative and np.isfinite(float(breakaway_result.torque_negative)):
            negative = bool(np.isclose(abs(float(breakaway_result.torque_negative)), scan_limit, atol=tolerance, rtol=0.0))
    both = bool(positive and negative) or bool(metadata.get("both_scan_limits_reached", False))
    return {
        "positive": bool(positive),
        "negative": bool(negative),
        "both": bool(both),
        "scan_max_torque": scan_limit,
    }


def _apply_breakaway_scan_limit_validation(
    validation_result: ValidationResult,
    breakaway_result: BreakawayIdentificationResult,
) -> ValidationResult:
    limit_status = _breakaway_scan_limit_status(breakaway_result)
    metadata = dict(validation_result.metadata)
    metadata["breakaway_scan_limit"] = limit_status
    if not bool(limit_status["both"]):
        return ValidationResult(
            friction_rmse=float(validation_result.friction_rmse),
            inertia_rmse=float(validation_result.inertia_rmse),
            recommended_for_compensation=bool(validation_result.recommended_for_compensation),
            detail=str(validation_result.detail),
            metadata=metadata,
        )

    reason = "breakaway_scan_limit_reached=both"
    reasons = list(metadata.get("reasons", []))
    if reason not in reasons:
        reasons.append(reason)
    metadata["reasons"] = reasons
    metadata["status"] = "rejected"
    detail = str(validation_result.detail)
    if not detail or detail in {"accepted", "not_run"} or bool(validation_result.recommended_for_compensation):
        detail = reason
    elif reason not in detail:
        detail = f"{detail}; {reason}"
    return ValidationResult(
        friction_rmse=float(validation_result.friction_rmse),
        inertia_rmse=float(validation_result.inertia_rmse),
        recommended_for_compensation=False,
        detail=detail,
        metadata=metadata,
    )


def _late_portion_mask(phase_names: np.ndarray, *, prefix: str, ratio: float) -> np.ndarray:
    phase_names = np.asarray(phase_names).astype(str)
    mask = np.zeros(phase_names.size, dtype=bool)
    ordered_phase_names = list(dict.fromkeys(phase_names.tolist()))
    for phase_name in ordered_phase_names:
        if not str(phase_name).startswith(prefix):
            continue
        indices = np.flatnonzero(phase_names == phase_name)
        if indices.size == 0:
            continue
        start_index = int(np.floor((1.0 - float(ratio)) * indices.size))
        mask[indices[start_index:]] = True
    return mask


def _empty_friction_result(size: int, *, status: str) -> FrictionIdentificationResult:
    return FrictionIdentificationResult(
        tau_c=float("nan"),
        viscous=float("nan"),
        tau_bias=float("nan"),
        train_rmse=float("nan"),
        valid_rmse=float("nan"),
        train_mask=np.zeros(size, dtype=bool),
        valid_mask=np.zeros(size, dtype=bool),
        torque_pred=np.full(size, np.nan, dtype=np.float64),
        torque_target=np.full(size, np.nan, dtype=np.float64),
        metadata={"status": status},
    )


def _empty_inertia_result(size: int, *, status: str) -> InertiaIdentificationResult:
    return InertiaIdentificationResult(
        inertia=float("nan"),
        train_rmse=float("nan"),
        valid_rmse=float("nan"),
        train_mask=np.zeros(size, dtype=bool),
        valid_mask=np.zeros(size, dtype=bool),
        torque_pred=np.full(size, np.nan, dtype=np.float64),
        torque_target=np.full(size, np.nan, dtype=np.float64),
        filtered_velocity=np.full(size, np.nan, dtype=np.float64),
        acceleration=np.full(size, np.nan, dtype=np.float64),
        metadata={"status": status},
    )


def _empty_validation_result(*, status: str) -> ValidationResult:
    return ValidationResult(
        friction_rmse=float("nan"),
        inertia_rmse=float("nan"),
        recommended_for_compensation=False,
        detail=status,
        metadata={"status": status},
    )


def _inertia_candidate_score(result: InertiaIdentificationResult) -> tuple[int, float]:
    if str(result.metadata.get("status", "")) == "ok" and np.isfinite(float(result.valid_rmse)):
        return (0, float(result.valid_rmse))
    if str(result.metadata.get("status", "")) == "ok" and np.isfinite(float(result.train_rmse)):
        return (1, float(result.train_rmse))
    return (2, float("inf"))


def _fit_inertia_model_with_candidate_windows(
    *,
    config: Config,
    capture: RoundCapture,
    friction_result: FrictionIdentificationResult,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> InertiaIdentificationResult:
    candidate_windows = tuple(int(window) for window in config.identification.inertia_savgol_window_candidates)
    candidate_results: list[InertiaIdentificationResult] = []
    candidate_summaries: list[dict[str, object]] = []

    for window in candidate_windows:
        result = fit_inertia_model(
            capture.time,
            capture.velocity,
            capture.torque_feedback,
            friction_result=friction_result,
            train_mask=train_mask,
            valid_mask=valid_mask,
            savgol_window=int(window),
            savgol_polyorder=int(config.identification.savgol_polyorder),
        )
        candidate_results.append(result)
        candidate_summaries.append(
            {
                "window": int(window),
                "status": str(result.metadata.get("status", "unknown")),
                "inertia": float(result.inertia),
                "train_rmse": float(result.train_rmse),
                "valid_rmse": float(result.valid_rmse),
            }
        )

    if not candidate_results:
        return _empty_inertia_result(capture.sample_count, status="no_inertia_savgol_window_candidates")

    selected_index = min(range(len(candidate_results)), key=lambda index: _inertia_candidate_score(candidate_results[index]))
    selected = candidate_results[selected_index]
    score_kind, _ = _inertia_candidate_score(selected)
    metadata = dict(selected.metadata)
    metadata["selected_savgol_window"] = int(candidate_windows[selected_index])
    metadata["savgol_window_candidates"] = candidate_summaries
    metadata["candidate_selection_metric"] = "valid_rmse" if score_kind == 0 else ("train_rmse" if score_kind == 1 else "none")
    return InertiaIdentificationResult(
        inertia=float(selected.inertia),
        train_rmse=float(selected.train_rmse),
        valid_rmse=float(selected.valid_rmse),
        train_mask=np.asarray(selected.train_mask, dtype=bool),
        valid_mask=np.asarray(selected.valid_mask, dtype=bool),
        torque_pred=np.asarray(selected.torque_pred, dtype=np.float64),
        torque_target=np.asarray(selected.torque_target, dtype=np.float64),
        filtered_velocity=np.asarray(selected.filtered_velocity, dtype=np.float64),
        acceleration=np.asarray(selected.acceleration, dtype=np.float64),
        metadata=metadata,
    )


def _fit_dynamic_mit_from_capture(capture: RoundCapture, config: Config) -> DynamicMotorFitResult:
    phase_names = np.asarray(capture.phase_name).astype(str)
    dynamic_mask = np.asarray([name.startswith("dynamic_mit_") for name in phase_names], dtype=bool)
    used_mask = dynamic_mask & np.asarray(capture.used_for_fit, dtype=bool)
    train_mask = used_mask & np.asarray([name.startswith("dynamic_mit_train") for name in phase_names], dtype=bool)
    valid_mask = used_mask & np.asarray([name.startswith("dynamic_mit_valid") for name in phase_names], dtype=bool)
    torque_target = (
        np.asarray(capture.tau_mit_est, dtype=np.float64)
        if bool(config.dynamic_mit.use_mit_estimated_torque)
        else np.asarray(capture.torque_feedback, dtype=np.float64)
    )
    metadata = {
        "status": "not_run",
        "source_torque": "tau_mit_est" if bool(config.dynamic_mit.use_mit_estimated_torque) else "torque_feedback",
        "used_sample_count": int(np.count_nonzero(used_mask)),
        "train_sample_count": int(np.count_nonzero(train_mask)),
        "valid_sample_count": int(np.count_nonzero(valid_mask)),
        "use_for_publish": bool(config.dynamic_mit.use_for_publish),
    }
    if np.count_nonzero(train_mask) < int(config.dynamic_mit.min_fit_sample_count):
        metadata["status"] = "insufficient_dynamic_mit_samples"
        return DynamicMotorFitResult(
            inertia=float("nan"),
            viscous=float("nan"),
            tau_c=float("nan"),
            tau_bias=float("nan"),
            train_rmse=float("nan"),
            valid_rmse=float("nan"),
            train_mask=train_mask,
            valid_mask=valid_mask,
            torque_pred=np.full(capture.sample_count, np.nan, dtype=np.float64),
            torque_target=torque_target,
            metadata=metadata,
        )
    filtered_velocity, measured_acceleration = estimate_filtered_velocity_and_acceleration(
        capture.time,
        capture.velocity,
        savgol_window=int(config.identification.savgol_window),
        savgol_polyorder=int(config.identification.savgol_polyorder),
    )
    result = fit_dynamic_motor_model(
        filtered_velocity,
        measured_acceleration,
        torque_target,
        train_mask=train_mask,
        valid_mask=valid_mask,
    )
    merged_metadata = dict(result.metadata)
    merged_metadata.update(metadata)
    merged_metadata["status"] = str(result.metadata.get("status", "ok"))
    return DynamicMotorFitResult(
        inertia=float(result.inertia),
        viscous=float(result.viscous),
        tau_c=float(result.tau_c),
        tau_bias=float(result.tau_bias),
        train_rmse=float(result.train_rmse),
        valid_rmse=float(result.valid_rmse),
        train_mask=np.asarray(result.train_mask, dtype=bool),
        valid_mask=np.asarray(result.valid_mask, dtype=bool),
        torque_pred=np.asarray(result.torque_pred, dtype=np.float64),
        torque_target=np.asarray(result.torque_target, dtype=np.float64),
        metadata=merged_metadata,
    )


def _friction_result_from_dynamic(dynamic_result: DynamicMotorFitResult) -> FrictionIdentificationResult:
    return FrictionIdentificationResult(
        tau_c=float(dynamic_result.tau_c),
        viscous=float(dynamic_result.viscous),
        tau_bias=float(dynamic_result.tau_bias),
        train_rmse=float(dynamic_result.train_rmse),
        valid_rmse=float(dynamic_result.valid_rmse),
        train_mask=np.asarray(dynamic_result.train_mask, dtype=bool),
        valid_mask=np.asarray(dynamic_result.valid_mask, dtype=bool),
        torque_pred=np.asarray(dynamic_result.torque_pred, dtype=np.float64),
        torque_target=np.asarray(dynamic_result.torque_target, dtype=np.float64),
        metadata={"status": str(dynamic_result.metadata.get("status", "ok")), "source": "dynamic_mit"},
    )


def _inertia_result_from_dynamic(dynamic_result: DynamicMotorFitResult, sample_count: int) -> InertiaIdentificationResult:
    return InertiaIdentificationResult(
        inertia=float(dynamic_result.inertia),
        train_rmse=float(dynamic_result.train_rmse),
        valid_rmse=float(dynamic_result.valid_rmse),
        train_mask=np.asarray(dynamic_result.train_mask, dtype=bool),
        valid_mask=np.asarray(dynamic_result.valid_mask, dtype=bool),
        torque_pred=np.asarray(dynamic_result.torque_pred, dtype=np.float64),
        torque_target=np.asarray(dynamic_result.torque_target, dtype=np.float64),
        filtered_velocity=np.full(sample_count, np.nan, dtype=np.float64),
        acceleration=np.full(sample_count, np.nan, dtype=np.float64),
        metadata={"status": str(dynamic_result.metadata.get("status", "ok")), "source": "dynamic_mit"},
    )


def _validation_result_from_dynamic(dynamic_result: DynamicMotorFitResult, config: Config) -> ValidationResult:
    dynamic_rmse = float(dynamic_result.valid_rmse)
    reasons: list[str] = []
    if not np.isfinite(float(dynamic_result.inertia)) or float(dynamic_result.inertia) < 0.0:
        reasons.append(f"invalid_inertia={float(dynamic_result.inertia):+.6f}")
    if not np.isfinite(float(dynamic_result.viscous)) or float(dynamic_result.viscous) < 0.0:
        reasons.append(f"invalid_viscous={float(dynamic_result.viscous):+.6f}")
    if not np.isfinite(float(dynamic_result.tau_c)) or float(dynamic_result.tau_c) < 0.0:
        reasons.append(f"invalid_tau_c={float(dynamic_result.tau_c):+.6f}")
    if not np.isfinite(dynamic_rmse) or dynamic_rmse > float(config.identification.inertia_rmse_publish_threshold):
        reasons.append(f"dynamic_mit_rmse={dynamic_rmse:.6f}>{float(config.identification.inertia_rmse_publish_threshold):.6f}")
    recommended = not reasons
    return ValidationResult(
        friction_rmse=dynamic_rmse,
        inertia_rmse=dynamic_rmse,
        recommended_for_compensation=bool(recommended),
        detail=f"dynamic_mit_rmse={dynamic_rmse:.6f}" if recommended else "; ".join(reasons),
        metadata={
            "status": "accepted" if recommended else "rejected",
            "model_kind": "dynamic_mit_v1",
            "dynamic_mit_validation_rmse": dynamic_rmse,
            "reasons": reasons,
        },
    )


def _masked_rmse(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    residual = np.asarray(target, dtype=np.float64)[mask] - np.asarray(prediction, dtype=np.float64)[mask]
    return float(np.sqrt(np.mean(residual**2)))


def _fit_joint_static_dynamic_model(
    capture: RoundCapture,
    config: Config,
    *,
    platforms: list[_SpeedHoldPlatformStat],
) -> DynamicMotorFitResult:
    phase_names = np.asarray(capture.phase_name).astype(str)
    filtered_velocity, measured_acceleration = estimate_filtered_velocity_and_acceleration(
        capture.time,
        capture.velocity,
        savgol_window=int(config.identification.savgol_window),
        savgol_polyorder=int(config.identification.savgol_polyorder),
    )
    velocity_parts: list[np.ndarray] = []
    acceleration_parts: list[np.ndarray] = []
    torque_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []

    def append_rows(
        source: str,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        torque: np.ndarray,
        *,
        train_mask: np.ndarray,
        valid_mask: np.ndarray,
        weight: float,
    ) -> None:
        velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
        acceleration = np.asarray(acceleration, dtype=np.float64).reshape(-1)
        torque = np.asarray(torque, dtype=np.float64).reshape(-1)
        train_mask = np.asarray(train_mask, dtype=bool).reshape(-1)
        valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if velocity.size == 0:
            return
        if not (velocity.size == acceleration.size == torque.size == train_mask.size == valid_mask.size):
            raise ValueError(f"joint source {source} arrays must have matching sizes.")
        velocity_parts.append(velocity)
        acceleration_parts.append(acceleration)
        torque_parts.append(torque)
        train_parts.append(train_mask)
        valid_parts.append(valid_mask)
        weight_parts.append(np.full(velocity.size, float(weight), dtype=np.float64))
        source_parts.append(np.asarray([str(source)] * velocity.size))

    accepted_platforms = [platform for platform in platforms if bool(platform.accepted)]
    if accepted_platforms:
        platform_velocity = np.asarray([platform.mean_velocity for platform in accepted_platforms], dtype=np.float64)
        platform_torque = np.asarray([platform.mean_torque for platform in accepted_platforms], dtype=np.float64)
        append_rows(
            "speed_hold",
            platform_velocity,
            np.zeros(platform_velocity.size, dtype=np.float64),
            platform_torque,
            train_mask=np.asarray([platform.bucket == "train" for platform in accepted_platforms], dtype=bool),
            valid_mask=np.asarray([platform.bucket == "valid" for platform in accepted_platforms], dtype=bool),
            weight=float(config.identification.joint_speed_hold_weight),
        )

    inertia_train_mask = np.asarray([name.startswith("inertia_train_") for name in phase_names], dtype=bool)
    inertia_valid_mask = np.asarray([name.startswith("inertia_valid_") for name in phase_names], dtype=bool)
    inertia_mask = inertia_train_mask | inertia_valid_mask
    if np.any(inertia_mask):
        indices = np.flatnonzero(inertia_mask)
        append_rows(
            "inertia",
            filtered_velocity[indices],
            measured_acceleration[indices],
            np.asarray(capture.torque_feedback, dtype=np.float64)[indices],
            train_mask=inertia_train_mask[indices],
            valid_mask=inertia_valid_mask[indices],
            weight=float(config.identification.joint_inertia_weight),
        )

    dynamic_train_mask = np.asarray([name.startswith("dynamic_mit_train") for name in phase_names], dtype=bool)
    dynamic_valid_mask = np.asarray([name.startswith("dynamic_mit_valid") for name in phase_names], dtype=bool)
    dynamic_used_mask = (dynamic_train_mask | dynamic_valid_mask) & np.asarray(capture.used_for_fit, dtype=bool)
    if np.any(dynamic_used_mask):
        indices = np.flatnonzero(dynamic_used_mask)
        dynamic_torque = (
            np.asarray(capture.tau_mit_est, dtype=np.float64)
            if bool(config.dynamic_mit.use_mit_estimated_torque)
            else np.asarray(capture.torque_feedback, dtype=np.float64)
        )
        append_rows(
            "dynamic_mit",
            filtered_velocity[indices],
            measured_acceleration[indices],
            dynamic_torque[indices],
            train_mask=dynamic_train_mask[indices],
            valid_mask=dynamic_valid_mask[indices],
            weight=float(config.identification.joint_dynamic_mit_weight),
        )

    if not velocity_parts:
        return DynamicMotorFitResult(
            inertia=float("nan"),
            viscous=float("nan"),
            tau_c=float("nan"),
            tau_bias=float("nan"),
            train_rmse=float("nan"),
            valid_rmse=float("nan"),
            train_mask=np.zeros(0, dtype=bool),
            valid_mask=np.zeros(0, dtype=bool),
            torque_pred=np.zeros(0, dtype=np.float64),
            torque_target=np.zeros(0, dtype=np.float64),
            metadata={"status": "insufficient_joint_samples"},
        )

    velocity = np.concatenate(velocity_parts)
    acceleration = np.concatenate(acceleration_parts)
    torque = np.concatenate(torque_parts)
    train_mask = np.concatenate(train_parts)
    valid_mask = np.concatenate(valid_parts)
    sample_weight = np.concatenate(weight_parts)
    source_names = np.concatenate(source_parts).astype(str)

    result = fit_weighted_dynamic_motor_model(
        velocity,
        acceleration,
        torque,
        train_mask=train_mask,
        valid_mask=valid_mask,
        sample_weight=sample_weight,
        min_train_samples=int(config.identification.min_joint_fit_sample_count),
    )
    metadata = dict(result.metadata)
    source_sample_counts = {
        source: int(np.count_nonzero(source_names == source))
        for source in ("speed_hold", "inertia", "dynamic_mit")
    }
    source_train_sample_counts = {
        source: int(np.count_nonzero((source_names == source) & result.train_mask))
        for source in ("speed_hold", "inertia", "dynamic_mit")
    }
    source_valid_sample_counts = {
        source: int(np.count_nonzero((source_names == source) & result.valid_mask))
        for source in ("speed_hold", "inertia", "dynamic_mit")
    }
    inertia_dynamic_mask = np.asarray(
        [(source == "inertia" or source == "dynamic_mit") for source in source_names],
        dtype=bool,
    )
    metadata.update(
        {
            "model_kind": "joint_static_dynamic_v1",
            "fit_method": "joint_weighted_robust_constrained_ls",
            "source_torque": "tau_mit_est" if bool(config.dynamic_mit.use_mit_estimated_torque) else "torque_feedback",
            "source_sample_counts": source_sample_counts,
            "source_train_sample_counts": source_train_sample_counts,
            "source_valid_sample_counts": source_valid_sample_counts,
            "source_weights": {
                "speed_hold": float(config.identification.joint_speed_hold_weight),
                "inertia": float(config.identification.joint_inertia_weight),
                "dynamic_mit": float(config.identification.joint_dynamic_mit_weight),
            },
            "speed_hold_train_rmse": _masked_rmse(torque, result.torque_pred, (source_names == "speed_hold") & result.train_mask),
            "speed_hold_valid_rmse": _masked_rmse(torque, result.torque_pred, (source_names == "speed_hold") & result.valid_mask),
            "inertia_train_rmse": _masked_rmse(torque, result.torque_pred, (source_names == "inertia") & result.train_mask),
            "inertia_valid_rmse": _masked_rmse(torque, result.torque_pred, (source_names == "inertia") & result.valid_mask),
            "dynamic_mit_train_rmse": _masked_rmse(torque, result.torque_pred, (source_names == "dynamic_mit") & result.train_mask),
            "dynamic_mit_valid_rmse": _masked_rmse(torque, result.torque_pred, (source_names == "dynamic_mit") & result.valid_mask),
            "inertia_dynamic_train_rmse": _masked_rmse(torque, result.torque_pred, inertia_dynamic_mask & result.train_mask),
            "inertia_dynamic_valid_rmse": _masked_rmse(torque, result.torque_pred, inertia_dynamic_mask & result.valid_mask),
        }
    )
    return DynamicMotorFitResult(
        inertia=float(result.inertia),
        viscous=float(result.viscous),
        tau_c=float(result.tau_c),
        tau_bias=float(result.tau_bias),
        train_rmse=float(result.train_rmse),
        valid_rmse=float(result.valid_rmse),
        train_mask=np.asarray(result.train_mask, dtype=bool),
        valid_mask=np.asarray(result.valid_mask, dtype=bool),
        torque_pred=np.asarray(result.torque_pred, dtype=np.float64),
        torque_target=np.asarray(result.torque_target, dtype=np.float64),
        metadata=metadata,
    )


def _friction_result_from_joint(joint_result: DynamicMotorFitResult) -> FrictionIdentificationResult:
    metadata = dict(joint_result.metadata)
    metadata["source"] = "joint_static_dynamic"
    return FrictionIdentificationResult(
        tau_c=float(joint_result.tau_c),
        viscous=float(joint_result.viscous),
        tau_bias=float(joint_result.tau_bias),
        train_rmse=float(metadata.get("speed_hold_train_rmse", joint_result.train_rmse)),
        valid_rmse=float(metadata.get("speed_hold_valid_rmse", joint_result.valid_rmse)),
        train_mask=np.asarray(joint_result.train_mask, dtype=bool),
        valid_mask=np.asarray(joint_result.valid_mask, dtype=bool),
        torque_pred=np.asarray(joint_result.torque_pred, dtype=np.float64),
        torque_target=np.asarray(joint_result.torque_target, dtype=np.float64),
        metadata=metadata,
    )


def _inertia_result_from_joint(joint_result: DynamicMotorFitResult) -> InertiaIdentificationResult:
    metadata = dict(joint_result.metadata)
    metadata["source"] = "joint_static_dynamic"
    return InertiaIdentificationResult(
        inertia=float(joint_result.inertia),
        train_rmse=float(metadata.get("inertia_dynamic_train_rmse", joint_result.train_rmse)),
        valid_rmse=float(metadata.get("inertia_dynamic_valid_rmse", joint_result.valid_rmse)),
        train_mask=np.asarray(joint_result.train_mask, dtype=bool),
        valid_mask=np.asarray(joint_result.valid_mask, dtype=bool),
        torque_pred=np.asarray(joint_result.torque_pred, dtype=np.float64),
        torque_target=np.asarray(joint_result.torque_target, dtype=np.float64),
        filtered_velocity=np.full(joint_result.torque_target.size, np.nan, dtype=np.float64),
        acceleration=np.full(joint_result.torque_target.size, np.nan, dtype=np.float64),
        metadata=metadata,
    )


def _validation_result_from_joint(joint_result: DynamicMotorFitResult, config: Config) -> ValidationResult:
    metadata = dict(joint_result.metadata)
    source_valid_counts = metadata.get("source_valid_sample_counts", {})
    if not isinstance(source_valid_counts, dict):
        source_valid_counts = {}
    friction_rmse = float(metadata.get("speed_hold_valid_rmse", joint_result.valid_rmse))
    inertia_rmse = float(metadata.get("inertia_dynamic_valid_rmse", joint_result.valid_rmse))
    reasons: list[str] = []
    if str(metadata.get("status", "")) != "ok":
        reasons.append(f"joint_fit_status={metadata.get('status', 'unknown')}")
    if int(metadata.get("fit_train_sample_count", 0)) < int(config.identification.min_joint_fit_sample_count):
        reasons.append(
            f"fit_train_sample_count={int(metadata.get('fit_train_sample_count', 0))}"
            f"<{int(config.identification.min_joint_fit_sample_count)}"
        )
    if int(source_valid_counts.get("speed_hold", 0)) < 2:
        reasons.append(f"speed_hold_valid_sample_count={int(source_valid_counts.get('speed_hold', 0))}<2")
    if not np.isfinite(float(joint_result.inertia)) or float(joint_result.inertia) < 0.0:
        reasons.append(f"invalid_inertia={float(joint_result.inertia):+.6f}")
    if not np.isfinite(float(joint_result.viscous)) or float(joint_result.viscous) < 0.0:
        reasons.append(f"invalid_viscous={float(joint_result.viscous):+.6f}")
    if not np.isfinite(float(joint_result.tau_c)) or float(joint_result.tau_c) < 0.0:
        reasons.append(f"invalid_tau_c={float(joint_result.tau_c):+.6f}")
    if not np.isfinite(friction_rmse) or friction_rmse > float(config.identification.friction_rmse_publish_threshold):
        reasons.append(f"friction_rmse={friction_rmse:.6f}>{float(config.identification.friction_rmse_publish_threshold):.6f}")
    if not np.isfinite(inertia_rmse) or inertia_rmse > float(config.identification.inertia_rmse_publish_threshold):
        reasons.append(f"inertia_rmse={inertia_rmse:.6f}>{float(config.identification.inertia_rmse_publish_threshold):.6f}")

    recommended = not reasons
    detail = (
        f"joint_friction_rmse={friction_rmse:.6f}, joint_inertia_rmse={inertia_rmse:.6f}"
        if recommended
        else "; ".join(reasons)
    )
    return ValidationResult(
        friction_rmse=friction_rmse,
        inertia_rmse=inertia_rmse,
        recommended_for_compensation=bool(recommended),
        detail=detail,
        metadata={
            "status": "accepted" if recommended else "rejected",
            "model_kind": "joint_static_dynamic_v1",
            "joint_valid_rmse": float(joint_result.valid_rmse),
            "source_valid_sample_counts": source_valid_counts,
            "source_weights": metadata.get("source_weights", {}),
            "reasons": reasons,
        },
    )


def _validation_score(validation: ValidationResult) -> float:
    values = [
        float(value)
        for value in (validation.friction_rmse, validation.inertia_rmse)
        if np.isfinite(float(value))
    ]
    if not values:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def _should_use_joint_validation(static_validation: ValidationResult, joint_validation: ValidationResult) -> bool:
    static_recommended = bool(static_validation.recommended_for_compensation)
    joint_recommended = bool(joint_validation.recommended_for_compensation)
    if joint_recommended and not static_recommended:
        return True
    if static_recommended and not joint_recommended:
        return False
    return _validation_score(joint_validation) <= _validation_score(static_validation)


def _joint_candidate_summary(joint_result: DynamicMotorFitResult, joint_validation: ValidationResult) -> dict[str, object]:
    return {
        "status": str(joint_validation.metadata.get("status", joint_result.metadata.get("status", "unknown"))),
        "recommended_for_compensation": bool(joint_validation.recommended_for_compensation),
        "friction_rmse": float(joint_validation.friction_rmse),
        "inertia_rmse": float(joint_validation.inertia_rmse),
        "score": _validation_score(joint_validation),
        "detail": str(joint_validation.detail),
        "fit_train_sample_count": int(joint_result.metadata.get("fit_train_sample_count", 0)),
        "rejected_train_sample_count": int(joint_result.metadata.get("rejected_train_sample_count", 0)),
    }


def _low_speed_fit_sample_counts(capture: RoundCapture) -> dict[str, int]:
    phase_names = np.asarray(capture.phase_name).astype(str)
    stiction_evidence = np.asarray(capture.stiction_evidence, dtype=bool)
    used_for_fit = np.asarray(capture.used_for_fit, dtype=bool)
    low_speed_steady_mask = np.asarray(
        [
            name.startswith("low_speed_hold_") or name.startswith("low_speed_micro_")
            for name in phase_names
        ],
        dtype=bool,
    )
    fit_mask = low_speed_steady_mask & used_for_fit & ~stiction_evidence
    velocity = np.asarray(capture.velocity, dtype=np.float64)
    return {
        "positive": int(np.count_nonzero(fit_mask & (velocity > 0.0))),
        "negative": int(np.count_nonzero(fit_mask & (velocity < 0.0))),
    }


def _piecewise_static_linear_export_payload(
    *,
    config: Config,
    capture: RoundCapture,
    friction_result: FrictionIdentificationResult,
    breakaway_result: BreakawayIdentificationResult,
    inertia_result: InertiaIdentificationResult,
    validation_result: ValidationResult,
) -> tuple[dict[str, object], dict[str, object]]:
    tau_c = max(float(friction_result.tau_c), 0.0) if np.isfinite(float(friction_result.tau_c)) else 0.0
    viscous = max(float(friction_result.viscous), 0.0) if np.isfinite(float(friction_result.viscous)) else 0.0
    tau_bias = float(friction_result.tau_bias) if np.isfinite(float(friction_result.tau_bias)) else 0.0
    inertia = max(float(inertia_result.inertia), 0.0) if np.isfinite(float(inertia_result.inertia)) else 0.0
    tau_static = (
        float(breakaway_result.tau_static)
        if np.isfinite(float(breakaway_result.tau_static)) and float(breakaway_result.tau_static) > 0.0
        else tau_c
    )
    parameters = {
        "tau_static": float(tau_static),
        "tau_c": float(tau_c),
        "viscous": float(viscous),
        "tau_bias": float(tau_bias),
        "inertia": float(inertia),
        "static_velocity_threshold_rad_s": float(config.compensation.static_velocity_threshold_rad_s),
        "static_transition_velocity_rad_s": float(config.compensation.static_transition_velocity_rad_s),
        "breakaway_positive": float(breakaway_result.torque_positive),
        "breakaway_negative": float(breakaway_result.torque_negative),
    }
    low_speed_counts = _low_speed_fit_sample_counts(capture)
    metadata = {
        "fit_source_model_kind": str(validation_result.metadata.get("model_kind", "static_v1")),
        "low_speed_fit_sample_counts": low_speed_counts,
        "min_low_speed_fit_samples_per_direction": int(config.identification.min_low_speed_fit_samples_per_direction),
        "friction_fit_metadata": dict(friction_result.metadata),
        "inertia_fit_metadata": dict(inertia_result.metadata),
    }
    friction_model: dict[str, object] = {
        "kind": PIECEWISE_STATIC_LINEAR_KIND,
        "equation": (
            "tau=tau_bias+viscous*v+direction*level(abs(v))+inertia*a; "
            "level=tau_static for abs(v)<=v_static, linear transition to tau_c, tau_c for abs(v)>=v_transition"
        ),
        "parameters": parameters,
        "train_rmse": float(friction_result.train_rmse),
        "valid_rmse": float(validation_result.friction_rmse),
        "metadata": metadata,
    }
    export_models: dict[str, object] = {
        "embedded_piecewise_linear_friction": {
            "kind": PIECEWISE_STATIC_LINEAR_KIND,
            "tau_static": float(tau_static),
            "tau_c": float(tau_c),
            "viscous": float(viscous),
            "tau_bias": float(tau_bias),
            "inertia": float(inertia),
            "static_velocity_threshold_rad_s": float(config.compensation.static_velocity_threshold_rad_s),
            "static_transition_velocity_rad_s": float(config.compensation.static_transition_velocity_rad_s),
        }
    }
    return friction_model, export_models


def _identify_round(
    *,
    config: Config,
    capture: RoundCapture,
    mode: str,
    breakaway_result: BreakawayIdentificationResult,
) -> MotorIdentificationResult:
    sample_count = capture.sample_count
    phase_names = np.asarray(capture.phase_name).astype(str)
    friction_result = _empty_friction_result(sample_count, status="not_run")
    inertia_result = _empty_inertia_result(sample_count, status="not_run")
    validation_result = _empty_validation_result(status="not_run")
    speed_hold_platforms = _collect_speed_hold_platform_stats(capture, config)
    dynamic_result = _fit_dynamic_mit_from_capture(capture, config)

    if mode in {"identify-all", "speed-hold", "inertia"}:
        platform_velocity = np.asarray([platform.mean_velocity for platform in speed_hold_platforms], dtype=np.float64)
        platform_torque = np.asarray([platform.mean_torque for platform in speed_hold_platforms], dtype=np.float64)
        friction_train_mask = np.asarray(
            [platform.accepted and platform.bucket == "train" for platform in speed_hold_platforms],
            dtype=bool,
        )
        friction_valid_mask = np.asarray(
            [platform.accepted and platform.bucket == "valid" for platform in speed_hold_platforms],
            dtype=bool,
        )
        friction_result = fit_friction_model(
            platform_velocity,
            platform_torque,
            train_mask=friction_train_mask,
            valid_mask=friction_valid_mask,
        )
        friction_result = _annotate_friction_result(friction_result, platforms=speed_hold_platforms)

    if mode in {"identify-all", "inertia"}:
        inertia_train_mask = np.asarray([name.startswith("inertia_train_") for name in phase_names], dtype=bool)
        inertia_valid_mask = np.asarray([name.startswith("inertia_valid_") for name in phase_names], dtype=bool)
        inertia_result = _fit_inertia_model_with_candidate_windows(
            config=config,
            capture=capture,
            friction_result=friction_result,
            train_mask=inertia_train_mask,
            valid_mask=inertia_valid_mask,
        )
        validation_result = _build_round_validation_result(friction_result, inertia_result, config)
        has_dynamic_mit_samples = bool(np.any(np.asarray([name.startswith("dynamic_mit_") for name in phase_names], dtype=bool)))
        if mode == "identify-all" and has_dynamic_mit_samples:
            static_validation = validation_result
            joint_result = _fit_joint_static_dynamic_model(
                capture,
                config,
                platforms=speed_hold_platforms,
            )
            if str(joint_result.metadata.get("status", "")) == "ok":
                joint_validation = _validation_result_from_joint(joint_result, config)
                if _should_use_joint_validation(static_validation, joint_validation):
                    friction_result = _friction_result_from_joint(joint_result)
                    inertia_result = _inertia_result_from_joint(joint_result)
                    validation_result = ValidationResult(
                        friction_rmse=float(joint_validation.friction_rmse),
                        inertia_rmse=float(joint_validation.inertia_rmse),
                        recommended_for_compensation=bool(joint_validation.recommended_for_compensation),
                        detail=str(joint_validation.detail),
                        metadata={
                            **dict(joint_validation.metadata),
                            "model_selection": "joint_selected",
                            "static_only_validation": static_validation.metadata,
                            "static_only_friction_rmse": float(static_validation.friction_rmse),
                            "static_only_inertia_rmse": float(static_validation.inertia_rmse),
                        },
                    )
                else:
                    validation_result = ValidationResult(
                        friction_rmse=float(static_validation.friction_rmse),
                        inertia_rmse=float(static_validation.inertia_rmse),
                        recommended_for_compensation=bool(static_validation.recommended_for_compensation),
                        detail=str(static_validation.detail),
                        metadata={
                            **dict(static_validation.metadata),
                            "model_kind": "static_v1",
                            "model_selection": "static_selected_over_joint",
                            "joint_candidate": _joint_candidate_summary(joint_result, joint_validation),
                        },
                    )
    elif mode == "speed-hold":
        validation_result = ValidationResult(
            friction_rmse=float(friction_result.valid_rmse),
            inertia_rmse=float("nan"),
            recommended_for_compensation=False,
            detail="speed-hold debug mode",
            metadata={
                "status": "partial",
                "accepted_train_platform_count": int(friction_result.metadata.get("accepted_train_platform_count", 0)),
                "accepted_valid_platform_count": int(friction_result.metadata.get("accepted_valid_platform_count", 0)),
                "rejected_platform_count": int(friction_result.metadata.get("rejected_platform_count", 0)),
            },
        )
    elif mode == "breakaway":
        validation_result = _empty_validation_result(status="breakaway_only")
    elif mode == "dynamic-mit":
        friction_result = _friction_result_from_dynamic(dynamic_result)
        inertia_result = _inertia_result_from_dynamic(dynamic_result, sample_count)
        validation_result = _validation_result_from_dynamic(dynamic_result, config)

    if mode in {"identify-all", "inertia"}:
        validation_result = _apply_breakaway_scan_limit_validation(validation_result, breakaway_result)

    fit_model_kind = str(validation_result.metadata.get("model_kind", "static_v1"))
    model_kind = PIECEWISE_STATIC_LINEAR_KIND if mode == "identify-all" else fit_model_kind
    source_phases = ["breakaway", "speed-hold", "inertia"]
    if np.any(np.asarray([name.startswith("low_speed_") for name in phase_names], dtype=bool)):
        source_phases.insert(1, "low-speed")
    if fit_model_kind in {"dynamic_mit_v1", "joint_static_dynamic_v1"} and np.any(np.asarray([name.startswith("dynamic_mit_") for name in phase_names], dtype=bool)):
        source_phases.append("dynamic-mit")
    friction_model: dict[str, object] | None = None
    export_models: dict[str, object] | None = None
    if mode == "identify-all":
        friction_model, export_models = _piecewise_static_linear_export_payload(
            config=config,
            capture=capture,
            friction_result=friction_result,
            breakaway_result=breakaway_result,
            inertia_result=inertia_result,
            validation_result=validation_result,
        )
    return MotorIdentificationResult(
        motor_id=int(capture.target_motor_id),
        motor_name=str(capture.motor_name),
        breakaway=breakaway_result,
        friction=friction_result,
        inertia=inertia_result,
        validation=validation_result,
        metadata={
            "mode": str(mode),
            "model_kind": model_kind,
            "fit_model_kind": fit_model_kind,
            "source_phases": source_phases,
            "steady_window_ratio": float(config.mit_velocity.steady_window_ratio),
            "repeat_index": int(capture.group_index),
            "round_index": int(capture.round_index),
            "dynamic_mit": {
                "status": str(dynamic_result.metadata.get("status", "not_run")),
                "train_rmse": float(dynamic_result.train_rmse),
                "valid_rmse": float(dynamic_result.valid_rmse),
                "inertia": float(dynamic_result.inertia),
                "viscous": float(dynamic_result.viscous),
                "tau_c": float(dynamic_result.tau_c),
                "tau_bias": float(dynamic_result.tau_bias),
                "use_for_publish": bool(config.dynamic_mit.use_for_publish),
                "source_torque": str(dynamic_result.metadata.get("source_torque", "torque_feedback")),
            },
            **(
                {
                    "friction_model": friction_model,
                    "export_models": export_models,
                }
                if friction_model is not None and export_models is not None
                else {}
            ),
        },
    )


def _empty_breakaway_result(*, status: str) -> BreakawayIdentificationResult:
    return BreakawayIdentificationResult(
        torque_positive=float("nan"),
        torque_negative=float("nan"),
        tau_static=float("nan"),
        tau_bias=float("nan"),
        metadata={"status": status},
    )




def _run_motor_round(
    *,
    config: Config,
    transport: CommandTransport,
    parser: FeedbackFrameParser,
    rerun_recorder: RerunRecorder,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    mode: str,
) -> tuple[RoundCapture, MotorIdentificationResult]:
    if config.transport.flush_input_before_round:
        transport.reset_input_buffer()
        parser.reset()

    capture_buffer = CaptureBuffer(
        target_motor_id=int(target_motor_id),
        motor_name=config.motors.name_for(int(target_motor_id)),
    )
    breakaway_result = _empty_breakaway_result(status="not_run")
    if mode in {"identify-all", "breakaway"}:
        breakaway_result = run_breakaway_phase(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
        )
    if mode == "identify-all" and bool(config.low_speed.enabled):
        run_low_speed_characterization_phase(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
        )
    if mode in {"identify-all", "speed-hold", "inertia"}:
        run_speed_hold_phase(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
        )
    if mode in {"identify-all", "inertia"}:
        run_inertia_phase(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
        )
    if mode == "dynamic-mit" or (mode == "identify-all" and bool(config.dynamic_mit.enabled)):
        run_dynamic_mit_phase(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
        )

    capture = capture_buffer.build(
        group_index=int(group_index),
        round_index=int(round_index),
        metadata={
            "mode": str(mode),
            "enabled_motor_ids": list(config.enabled_motor_ids),
            "hard_speed_abort_abs": float(config.safety.hard_speed_abort_abs),
            "moving_velocity_threshold": float(config.safety.moving_velocity_threshold),
            "repeat_count": int(config.identification.repeat_count),
        },
    )
    identification = _identify_round(
        config=config,
        capture=capture,
        mode=mode,
        breakaway_result=breakaway_result,
    )
    rerun_recorder.log_round_stop(
        group_index=int(group_index),
        round_index=int(round_index),
        motor_id=int(target_motor_id),
        phase_name="completed",
        stage=mode,
    )
    return capture, identification


def _run_mode(
    config: Config,
    *,
    mode: str,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    store = ResultStore(config, mode=mode)
    parser = FeedbackFrameParser(max_motor_id=max(config.motor_ids))
    rerun_recorder = RerunRecorder(
        store.rerun_recording_path,
        motor_ids=config.motor_ids,
        motor_names={motor_id: config.motors.name_for(motor_id) for motor_id in config.motor_ids},
        mode=mode,
        show_viewer=show_rerun_viewer,
    )
    artifacts: list[RoundArtifact] = []
    transport = transport_factory() if transport_factory is not None else open_transport(config)

    try:
        precheck_transport(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
        )
        total_rounds = int(config.identification.repeat_count) * len(config.enabled_motor_ids)
        current_round = 0
        for group_index in range(1, int(config.identification.repeat_count) + 1):
            for target_motor_id in config.enabled_motor_ids:
                current_round += 1
                rerun_recorder.log_round_timing(
                    group_index=int(group_index),
                    round_index=int(current_round),
                    active_motor_id=int(target_motor_id),
                    planned_duration_s=0.0,
                    actual_capture_duration_s=0.0,
                    sync_wait_duration_s=0.0,
                    round_total_duration_s=0.0,
                )
                log_info(
                    f"Starting {mode} round {current_round}/{total_rounds}: "
                    f"repeat={group_index}, motor_id={target_motor_id}"
                )
                round_started = time.monotonic()
                capture, identification = _run_motor_round(
                    config=config,
                    transport=transport,
                    parser=parser,
                    rerun_recorder=rerun_recorder,
                    target_motor_id=int(target_motor_id),
                    group_index=int(group_index),
                    round_index=int(current_round),
                    mode=mode,
                )
                rerun_recorder.log_round_timing(
                    group_index=int(group_index),
                    round_index=int(current_round),
                    active_motor_id=int(target_motor_id),
                    planned_duration_s=float(capture.time[-1]) if capture.sample_count else 0.0,
                    actual_capture_duration_s=float(capture.time[-1]) if capture.sample_count else 0.0,
                    sync_wait_duration_s=0.0,
                    round_total_duration_s=float(time.monotonic() - round_started),
                )
                capture_path = store.save_capture(capture)
                identification_path = store.save_identification(capture, identification)
                artifacts.append(
                    RoundArtifact(
                        capture=capture,
                        identification=identification,
                        capture_path=capture_path,
                        identification_path=identification_path,
                    )
                )
                log_info(
                    f"motor_id={target_motor_id} finished: "
                    f"tau_static={float(identification.breakaway.tau_static):+.4f}, "
                    f"tau_c={float(identification.friction.tau_c):+.4f}, "
                    f"viscous={float(identification.friction.viscous):+.4f}, "
                    f"inertia={float(identification.inertia.inertia):+.4f}"
                )

        if mode == "identify-all":
            store.save_latest_parameters(artifacts)
        summary_paths = store.save_summary(artifacts)
        rerun_recorder.log_summary(
            summary_path=summary_paths.run_summary_path,
            report_path=summary_paths.run_summary_report_path,
        )
        return RunResult(
            artifacts=tuple(artifacts),
            summary_paths=summary_paths,
            manifest_path=store.manifest_path,
        )
    except RuntimeAbortError as exc:
        rerun_recorder.log_abort_event(exc.event.to_payload())
        store.record_abort_event(exc.event.to_payload())
        store.finalize()
        raise
    finally:
        rerun_recorder.close()
        transport.close()


def run_identify_all(
    config: Config,
    *,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    return _run_mode(
        config,
        mode="identify-all",
        transport_factory=transport_factory,
        show_rerun_viewer=show_rerun_viewer,
    )


def run_breakaway(
    config: Config,
    *,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    return _run_mode(
        config,
        mode="breakaway",
        transport_factory=transport_factory,
        show_rerun_viewer=show_rerun_viewer,
    )


def run_speed_hold(
    config: Config,
    *,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    return _run_mode(
        config,
        mode="speed-hold",
        transport_factory=transport_factory,
        show_rerun_viewer=show_rerun_viewer,
    )


def run_inertia(
    config: Config,
    *,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    return _run_mode(
        config,
        mode="inertia",
        transport_factory=transport_factory,
        show_rerun_viewer=show_rerun_viewer,
    )


def run_dynamic_mit(
    config: Config,
    *,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> RunResult:
    return _run_mode(
        config,
        mode="dynamic-mit",
        transport_factory=transport_factory,
        show_rerun_viewer=show_rerun_viewer,
    )


def _compensation_capture_metadata(
    *,
    config: Config,
    parameters,
    mode: str = "compensation",
    abort_event: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "mode": str(mode),
        "enabled_motor_ids": list(config.enabled_motor_ids),
        "hard_speed_abort_abs": float(config.safety.hard_speed_abort_abs),
        "moving_velocity_threshold": float(config.safety.moving_velocity_threshold),
        "latest_parameters_path": str(latest_parameters_path(config)),
        "identified_at": str(parameters.identified_at),
        "source_run_label": str(parameters.source_run_label),
        "model_kind": str(parameters.model_kind),
        "publish_status": str(parameters.publish_status),
        "publish_detail": str(parameters.publish_detail),
        "accepted_round_count": int(parameters.accepted_round_count),
        "selected_rounds": list(parameters.selected_rounds),
        "recommended_for_compensation": bool(parameters.recommended_for_compensation),
    }
    if abort_event is not None:
        metadata["abort_event"] = dict(abort_event)
    return metadata


def run_compensation(
    config: Config,
    *,
    transport_factory: Callable[[], CommandTransport] | None = None,
    show_rerun_viewer: bool = False,
    max_runtime_s: float | None = None,
) -> RunResult:
    if len(config.enabled_motor_ids) != 1:
        raise ValueError("compensation mode requires exactly one enabled motor_id.")

    target_motor_id = int(config.enabled_motor_ids[0])
    parameters = load_compensation_parameters(config, target_motor_id=target_motor_id)
    if not bool(parameters.recommended_for_compensation):
        log_info(
            f"Warning: motor_id={int(target_motor_id)} latest model is not recommended_for_compensation, "
            f"source_run_label={parameters.source_run_label}"
        )
    if str(parameters.publish_status) != "published":
        log_info(
            f"Warning: motor_id={int(target_motor_id)} using unpublished model because "
            f"compensation.require_published_model={str(bool(config.compensation.require_published_model)).lower()}, "
            f"publish_status={parameters.publish_status}"
        )

    store = ResultStore(config, mode="compensation")
    parser = FeedbackFrameParser(max_motor_id=max(config.motor_ids))
    rerun_recorder = RerunRecorder(
        store.rerun_recording_path,
        motor_ids=config.motor_ids,
        motor_names={motor_id: config.motors.name_for(motor_id) for motor_id in config.motor_ids},
        mode="compensation",
        show_viewer=show_rerun_viewer,
    )
    transport = transport_factory() if transport_factory is not None else open_transport(config)
    capture_buffer = CaptureBuffer(
        target_motor_id=int(target_motor_id),
        motor_name=config.motors.name_for(int(target_motor_id)),
    )
    hard_aborted = False

    try:
        precheck_transport(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
        )
        rerun_recorder.log_round_timing(
            group_index=1,
            round_index=1,
            active_motor_id=int(target_motor_id),
            planned_duration_s=0.0 if max_runtime_s is None else float(max_runtime_s),
            actual_capture_duration_s=0.0,
            sync_wait_duration_s=0.0,
            round_total_duration_s=0.0,
        )
        log_info(
            f"Starting compensation round 1/1: motor_id={int(target_motor_id)}, "
            f"source_run_label={parameters.source_run_label}, publish_status={parameters.publish_status}"
        )
        round_started = time.monotonic()
        run_compensation_phase_module(
            config=config,
            transport=transport,
            parser=parser,
            rerun_recorder=rerun_recorder,
            capture_buffer=capture_buffer,
            target_motor_id=int(target_motor_id),
            group_index=1,
            round_index=1,
            parameters=parameters,
            max_runtime_s=max_runtime_s,
        )
        capture = capture_buffer.build(
            group_index=1,
            round_index=1,
            metadata=_compensation_capture_metadata(config=config, parameters=parameters),
        )
        rerun_recorder.log_round_timing(
            group_index=1,
            round_index=1,
            active_motor_id=int(target_motor_id),
            planned_duration_s=0.0 if max_runtime_s is None else float(max_runtime_s),
            actual_capture_duration_s=float(capture.time[-1]) if capture.sample_count else 0.0,
            sync_wait_duration_s=0.0,
            round_total_duration_s=float(time.monotonic() - round_started),
        )
        capture_path = store.save_capture(capture)
        rerun_recorder.log_round_stop(
            group_index=1,
            round_index=1,
            motor_id=int(target_motor_id),
            phase_name="completed",
            stage="compensation",
        )
        store.finalize()
        log_info(
            f"motor_id={int(target_motor_id)} compensation finished: "
            f"source_run_label={parameters.source_run_label}, capture_samples={int(capture.sample_count)}"
        )
        return RunResult(
            artifacts=(capture_path,),
            summary_paths=None,
            manifest_path=store.manifest_path,
        )
    except RuntimeAbortError as exc:
        hard_aborted = True
        rerun_recorder.log_abort_event(exc.event.to_payload())
        if capture_buffer.time_log:
            capture = capture_buffer.build(
                group_index=1,
                round_index=1,
                metadata=_compensation_capture_metadata(
                    config=config,
                    parameters=parameters,
                    abort_event=exc.event.to_payload(),
                ),
            )
            capture_path = store.save_capture(capture)
            log_info(f"Saved compensation abort capture: {capture_path}")
        store.record_abort_event(exc.event.to_payload())
        store.finalize()
        raise
    finally:
        try:
            if not hard_aborted:
                send_zero_then_disable(
                    config=config,
                    transport=transport,
                    target_motor_id=int(target_motor_id),
                    semantic_mode="mit_torque",
                )
        finally:
            rerun_recorder.close()
            transport.close()


__all__ = [
    "run_breakaway",
    "run_compensation",
    "run_dynamic_mit",
    "run_identify_all",
    "run_inertia",
    "run_speed_hold",
]
