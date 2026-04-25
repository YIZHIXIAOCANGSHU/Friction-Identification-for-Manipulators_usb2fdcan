from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from friction_identification_core.core import (
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

    design = np.column_stack([np.sign(velocity[train_mask]), velocity[train_mask], np.ones(np.count_nonzero(train_mask))])
    coefficients, *_ = np.linalg.lstsq(design, torque[train_mask], rcond=None)
    tau_c, viscous, tau_bias = [float(item) for item in coefficients.tolist()]
    torque_pred = friction_torque_model(velocity, tau_c=tau_c, viscous=viscous, tau_bias=tau_bias)
    metadata["status"] = "ok"
    return FrictionIdentificationResult(
        tau_c=tau_c,
        viscous=viscous,
        tau_bias=tau_bias,
        train_rmse=_rmse(torque, torque_pred, train_mask),
        valid_rmse=_rmse(torque, torque_pred, valid_mask),
        train_mask=train_mask,
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
    "fit_friction_model",
    "fit_inertia_model",
]
