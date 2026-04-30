from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter

from friction_identification_core.core import (
    DynamicMotorFitResult,
    FrictionIdentificationResult,
    InertiaIdentificationResult,
    ValidationResult,
    friction_torque_model,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    residual = np.asarray(y_true, dtype=np.float64)[mask] - np.asarray(y_pred, dtype=np.float64)[mask]
    return float(np.sqrt(np.mean(residual**2)))


def _robust_residual_threshold(residual: np.ndarray, target: np.ndarray, active_mask: np.ndarray) -> float:
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    finite_mask = active_mask & np.isfinite(residual) & np.isfinite(target)
    if not np.any(finite_mask):
        return float("inf")

    active_residual = np.abs(residual[finite_mask])
    median = float(np.median(active_residual))
    mad = float(np.median(np.abs(active_residual - median)))
    sigma = 1.4826 * mad
    target_scale = float(np.nanmedian(np.abs(target[finite_mask])))
    residual_floor = max(1.0e-9, 0.02 * target_scale)
    if np.isfinite(sigma) and sigma > 1.0e-12:
        return max(float(median + 3.0 * sigma), residual_floor)
    return max(float(median * 4.0), residual_floor)


def _robust_refit_mask(
    design: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    active_mask = (
        np.asarray(train_mask, dtype=bool).reshape(-1)
        & np.isfinite(target)
        & np.all(np.isfinite(design), axis=1)
    )
    original_mask = active_mask.copy()
    if np.count_nonzero(active_mask) < 4:
        return active_mask, np.zeros_like(active_mask, dtype=bool)

    for _ in range(3):
        coefficients, *_ = np.linalg.lstsq(design[active_mask], target[active_mask], rcond=None)
        prediction = np.asarray(design @ coefficients, dtype=np.float64)
        residual = target - prediction
        threshold = _robust_residual_threshold(residual, target, active_mask)
        robust_mask = active_mask & (np.abs(residual) <= threshold)
        if np.count_nonzero(robust_mask) < 3 or np.array_equal(robust_mask, active_mask):
            break
        active_mask = robust_mask

    residual_mask = original_mask & ~active_mask
    return active_mask, residual_mask


def _bounded_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    *,
    lower_bounds: tuple[float, ...],
    upper_bounds: tuple[float, ...],
) -> np.ndarray:
    result = lsq_linear(
        np.asarray(design, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        bounds=(np.asarray(lower_bounds, dtype=np.float64), np.asarray(upper_bounds, dtype=np.float64)),
        method="trf",
    )
    return np.asarray(result.x, dtype=np.float64)


def _smooth_signal(signal: np.ndarray, *, window: int, polyorder: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size < 3:
        return signal.copy()

    window = max(int(window), 3)
    if window % 2 == 0:
        window += 1
    if window > signal.size:
        window = signal.size if signal.size % 2 == 1 else signal.size - 1
    if window <= polyorder:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
    if window > signal.size:
        window = signal.size if signal.size % 2 == 1 else signal.size - 1
    if window <= polyorder or window < 3:
        return signal.copy()
    return savgol_filter(signal, window_length=window, polyorder=int(polyorder), mode="interp")


def estimate_filtered_velocity_and_acceleration(
    time_s: np.ndarray,
    velocity: np.ndarray,
    *,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    filtered_velocity = _smooth_signal(velocity, window=savgol_window, polyorder=savgol_polyorder)
    acceleration = np.gradient(filtered_velocity, time_s, edge_order=1) if time_s.size >= 2 else np.zeros_like(filtered_velocity)
    return np.asarray(filtered_velocity, dtype=np.float64), np.asarray(acceleration, dtype=np.float64)


def fit_friction_model(
    velocity: np.ndarray,
    torque: np.ndarray,
    *,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> FrictionIdentificationResult:
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    torque = np.asarray(torque, dtype=np.float64).reshape(-1)
    train_mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)

    torque_pred = np.full(velocity.size, np.nan, dtype=np.float64)
    metadata: dict[str, object] = {
        "train_sample_count": int(np.count_nonzero(train_mask)),
        "valid_sample_count": int(np.count_nonzero(valid_mask)),
    }

    if not np.any(train_mask):
        metadata["status"] = "insufficient_train_samples"
        return FrictionIdentificationResult(
            tau_c=float("nan"),
            viscous=float("nan"),
            tau_bias=float("nan"),
            train_rmse=float("nan"),
            valid_rmse=float("nan"),
            train_mask=train_mask,
            valid_mask=valid_mask,
            torque_pred=torque_pred,
            torque_target=torque,
            metadata=metadata,
        )

    full_design = np.column_stack([np.sign(velocity), velocity, np.ones(velocity.size)])
    fit_train_mask, rejected_train_mask = _robust_refit_mask(full_design, torque, train_mask)
    design = full_design[fit_train_mask]
    coefficients = _bounded_lstsq(
        design,
        torque[fit_train_mask],
        lower_bounds=(0.0, 0.0, -np.inf),
        upper_bounds=(np.inf, np.inf, np.inf),
    )
    tau_c, viscous, tau_bias = [float(item) for item in coefficients.tolist()]
    torque_pred = friction_torque_model(velocity, tau_c=tau_c, viscous=viscous, tau_bias=tau_bias)
    metadata["status"] = "ok"
    metadata["fit_train_sample_count"] = int(np.count_nonzero(fit_train_mask))
    metadata["rejected_train_sample_count"] = int(np.count_nonzero(rejected_train_mask))
    return FrictionIdentificationResult(
        tau_c=tau_c,
        viscous=viscous,
        tau_bias=tau_bias,
        train_rmse=_rmse(torque, torque_pred, fit_train_mask),
        valid_rmse=_rmse(torque, torque_pred, valid_mask),
        train_mask=fit_train_mask,
        valid_mask=valid_mask,
        torque_pred=np.asarray(torque_pred, dtype=np.float64),
        torque_target=torque,
        metadata=metadata,
    )


def fit_inertia_model(
    time_s: np.ndarray,
    velocity: np.ndarray,
    torque: np.ndarray,
    *,
    friction_result: FrictionIdentificationResult,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    savgol_window: int,
    savgol_polyorder: int,
) -> InertiaIdentificationResult:
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    torque = np.asarray(torque, dtype=np.float64).reshape(-1)
    train_mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    filtered_velocity, acceleration = estimate_filtered_velocity_and_acceleration(
        time_s,
        velocity,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )
    friction_torque = friction_torque_model(
        filtered_velocity,
        tau_c=float(friction_result.tau_c),
        viscous=float(friction_result.viscous),
        tau_bias=float(friction_result.tau_bias),
    )
    residual = torque - friction_torque
    torque_pred = np.full(time_s.size, np.nan, dtype=np.float64)
    metadata: dict[str, object] = {
        "train_sample_count": int(np.count_nonzero(train_mask)),
        "valid_sample_count": int(np.count_nonzero(valid_mask)),
        "savgol_window": int(savgol_window),
        "savgol_polyorder": int(savgol_polyorder),
    }

    if not np.any(train_mask):
        metadata["status"] = "insufficient_train_samples"
        return InertiaIdentificationResult(
            inertia=float("nan"),
            train_rmse=float("nan"),
            valid_rmse=float("nan"),
            train_mask=train_mask,
            valid_mask=valid_mask,
            torque_pred=torque_pred,
            torque_target=torque,
            filtered_velocity=filtered_velocity,
            acceleration=acceleration,
            metadata=metadata,
        )

    acc_train = acceleration[train_mask]
    residual_train = residual[train_mask]
    denominator = float(np.dot(acc_train, acc_train))
    inertia = float(np.dot(acc_train, residual_train) / denominator) if denominator > 1.0e-9 else float("nan")
    if np.isfinite(inertia):
        inertia = max(float(inertia), 0.0)
    torque_pred = friction_torque + inertia * acceleration
    metadata["status"] = "ok" if np.isfinite(inertia) else "singular_train_acceleration"
    return InertiaIdentificationResult(
        inertia=inertia,
        train_rmse=_rmse(torque, torque_pred, train_mask),
        valid_rmse=_rmse(torque, torque_pred, valid_mask),
        train_mask=train_mask,
        valid_mask=valid_mask,
        torque_pred=np.asarray(torque_pred, dtype=np.float64),
        torque_target=torque,
        filtered_velocity=np.asarray(filtered_velocity, dtype=np.float64),
        acceleration=np.asarray(acceleration, dtype=np.float64),
        metadata=metadata,
    )


def fit_dynamic_motor_model(
    velocity: np.ndarray,
    acceleration: np.ndarray,
    torque: np.ndarray,
    *,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> DynamicMotorFitResult:
    return fit_weighted_dynamic_motor_model(
        velocity,
        acceleration,
        torque,
        train_mask=train_mask,
        valid_mask=valid_mask,
        sample_weight=None,
    )


def fit_weighted_dynamic_motor_model(
    velocity: np.ndarray,
    acceleration: np.ndarray,
    torque: np.ndarray,
    *,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    sample_weight: np.ndarray | None = None,
    min_train_samples: int = 4,
) -> DynamicMotorFitResult:
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    acceleration = np.asarray(acceleration, dtype=np.float64).reshape(-1)
    torque = np.asarray(torque, dtype=np.float64).reshape(-1)
    train_mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if sample_weight is None:
        sample_weight = np.ones(velocity.size, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if not (velocity.size == acceleration.size == torque.size == train_mask.size == valid_mask.size == sample_weight.size):
        raise ValueError("velocity, acceleration, torque, train_mask, valid_mask, and sample_weight must have the same size.")

    torque_pred = np.full(velocity.size, np.nan, dtype=np.float64)
    metadata: dict[str, object] = {
        "train_sample_count": int(np.count_nonzero(train_mask)),
        "valid_sample_count": int(np.count_nonzero(valid_mask)),
        "fit_method": "weighted_robust_constrained_ls",
    }
    finite_mask = (
        np.isfinite(velocity)
        & np.isfinite(acceleration)
        & np.isfinite(torque)
        & np.isfinite(sample_weight)
        & (sample_weight > 0.0)
    )
    train_mask = train_mask & finite_mask
    valid_mask = valid_mask & finite_mask
    if np.count_nonzero(train_mask) < int(min_train_samples):
        metadata["status"] = "insufficient_train_samples"
        return DynamicMotorFitResult(
            inertia=float("nan"),
            viscous=float("nan"),
            tau_c=float("nan"),
            tau_bias=float("nan"),
            train_rmse=float("nan"),
            valid_rmse=float("nan"),
            train_mask=train_mask,
            valid_mask=valid_mask,
            torque_pred=torque_pred,
            torque_target=torque,
            metadata=metadata,
        )

    design = np.column_stack([acceleration, velocity, np.sign(velocity), np.ones(velocity.size)])
    fit_train_mask, rejected_train_mask = _robust_refit_mask(design, torque, train_mask)
    train_weights = np.sqrt(sample_weight[fit_train_mask])
    weighted_design = design[fit_train_mask] * train_weights[:, np.newaxis]
    weighted_torque = torque[fit_train_mask] * train_weights
    coefficients = _bounded_lstsq(
        weighted_design,
        weighted_torque,
        lower_bounds=(0.0, 0.0, 0.0, -np.inf),
        upper_bounds=(np.inf, np.inf, np.inf, np.inf),
    )
    inertia, viscous, tau_c, tau_bias = [float(item) for item in coefficients.tolist()]
    torque_pred = design @ coefficients
    metadata["status"] = "ok"
    metadata["fit_train_sample_count"] = int(np.count_nonzero(fit_train_mask))
    metadata["rejected_train_sample_count"] = int(np.count_nonzero(rejected_train_mask))
    metadata["sample_weight_sum"] = float(np.sum(sample_weight[fit_train_mask]))
    return DynamicMotorFitResult(
        inertia=float(inertia),
        viscous=float(viscous),
        tau_c=float(tau_c),
        tau_bias=float(tau_bias),
        train_rmse=_rmse(torque, torque_pred, fit_train_mask),
        valid_rmse=_rmse(torque, torque_pred, valid_mask),
        train_mask=fit_train_mask,
        valid_mask=valid_mask,
        torque_pred=np.asarray(torque_pred, dtype=np.float64),
        torque_target=torque,
        metadata=metadata,
    )


def build_validation_result(
    friction_result: FrictionIdentificationResult,
    inertia_result: InertiaIdentificationResult,
    *,
    recommended_friction_rmse: float = 0.15,
    recommended_inertia_rmse: float = 0.20,
) -> ValidationResult:
    friction_rmse = float(friction_result.valid_rmse)
    inertia_rmse = float(inertia_result.valid_rmse)
    recommended = bool(
        np.isfinite(friction_rmse)
        and np.isfinite(inertia_rmse)
        and friction_rmse <= float(recommended_friction_rmse)
        and inertia_rmse <= float(recommended_inertia_rmse)
    )
    detail = (
        f"friction_rmse={friction_rmse:.6f}, inertia_rmse={inertia_rmse:.6f}"
        if np.isfinite(friction_rmse) and np.isfinite(inertia_rmse)
        else "validation metrics unavailable"
    )
    return ValidationResult(
        friction_rmse=friction_rmse,
        inertia_rmse=inertia_rmse,
        recommended_for_compensation=recommended,
        detail=detail,
        metadata={
            "recommended_friction_rmse": float(recommended_friction_rmse),
            "recommended_inertia_rmse": float(recommended_inertia_rmse),
        },
    )


__all__ = [
    "build_validation_result",
    "estimate_filtered_velocity_and_acceleration",
    "fit_dynamic_motor_model",
    "fit_weighted_dynamic_motor_model",
    "fit_friction_model",
    "fit_inertia_model",
]
