from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from friction_identification_core.core import IdentificationLimits
from send import damiao as damiao_socketcan

if TYPE_CHECKING:
    from friction_identification_core.runtime_config import Config


def motor_tmax_from_config(config: "Config", *, target_motor_id: int) -> float:
    motor_type = config.transport.motor_types[config.motor_index(int(target_motor_id))]
    limits = damiao_socketcan.get_motor_limits(motor_type)
    return abs(float(limits.tmax))


def identification_limits_for_motor(config: "Config", *, target_motor_id: int) -> IdentificationLimits:
    target_index = config.motor_index(int(target_motor_id))
    motor_tmax = motor_tmax_from_config(config, target_motor_id=int(target_motor_id))
    hard_speed_abs = abs(float(config.safety.hard_speed_abort_abs))
    identification_speed_abs = hard_speed_abs * float(config.identification.generation_safety_margin_ratio)
    compensation_torque_abs = motor_tmax * float(config.compensation.torque_limit_ratio)
    return IdentificationLimits(
        target_motor_id=int(target_motor_id),
        motor_tmax=float(motor_tmax),
        hard_speed_abs=float(hard_speed_abs),
        identification_speed_abs=float(identification_speed_abs),
        dynamic_mit_velocity_abs=float(config.dynamic_mit.velocity_limit),
        breakaway_scan_torque_abs=float(config.breakaway.scan_max_torque[target_index]),
        compensation_torque_abs=float(compensation_torque_abs),
        inertia_torque_abs=float(compensation_torque_abs * float(config.compensation.max_inertia_torque_ratio)),
    )


def validate_abs_less_than(values: np.ndarray | list[float] | tuple[float, ...], *, limit_abs: float, name: str) -> None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    max_abs = float(np.nanmax(np.abs(array)))
    if max_abs >= float(limit_abs):
        raise ValueError(f"{name} must be < identification_speed_abs ({float(limit_abs):.6f} rad/s); max_abs={max_abs:.6f}.")


def validate_abs_less_or_equal(values: np.ndarray | list[float] | tuple[float, ...], *, limit_abs: float, name: str) -> None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    max_abs = float(np.nanmax(np.abs(array)))
    if max_abs > float(limit_abs) + 1.0e-9:
        raise ValueError(f"{name} must be <= {float(limit_abs):.6f}; max_abs={max_abs:.6f}.")


def validate_configured_identification_limits(config: "Config") -> None:
    for motor_id in config.motor_ids:
        limits = identification_limits_for_motor(config, target_motor_id=int(motor_id))
        if not np.isfinite(limits.motor_tmax) or limits.motor_tmax <= 0.0:
            raise ValueError(f"motor_tmax must be finite and > 0 for motor_id={int(motor_id)}.")
        if limits.breakaway_scan_torque_abs > limits.motor_tmax + 1.0e-9:
            raise ValueError(
                "breakaway.scan_max_torque must be <= motor torque limit for "
                f"motor_id={int(motor_id)} ({limits.motor_tmax:.6f} Nm)."
            )
        validate_abs_less_than(
            config.low_speed.speed_points,
            limit_abs=limits.identification_speed_abs,
            name="low_speed.speed_points",
        )
        validate_abs_less_than(
            (float(config.low_speed.micro_motion_velocity_limit),),
            limit_abs=limits.identification_speed_abs,
            name="low_speed.micro_motion_velocity_limit",
        )
        validate_abs_less_than(
            config.identification.steady_speed_points,
            limit_abs=limits.identification_speed_abs,
            name="identification.steady_speed_points",
        )
        validate_abs_less_than(
            config.inertia.waypoints,
            limit_abs=limits.identification_speed_abs,
            name="inertia.waypoints",
        )
        validate_abs_less_than(
            (float(config.dynamic_mit.velocity_limit),),
            limit_abs=limits.identification_speed_abs,
            name="dynamic_mit.velocity_limit",
        )


__all__ = [
    "identification_limits_for_motor",
    "motor_tmax_from_config",
    "validate_abs_less_or_equal",
    "validate_abs_less_than",
    "validate_configured_identification_limits",
]
