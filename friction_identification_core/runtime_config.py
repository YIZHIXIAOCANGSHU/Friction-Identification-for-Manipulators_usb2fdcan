from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("default.yaml")

_REJECTED_TOP_LEVEL_KEYS = frozenset({"control", "excitation", "step_torque"})


def _parse_int(value: Any) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def _as_int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(_parse_int(item) for item in values)


def _expand_float_vector(values: Any, size: int, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 1:
        return np.full(size, float(array[0]), dtype=np.float64)
    if array.size != size:
        raise ValueError(f"{name} must contain either 1 or {size} values.")
    return array.astype(np.float64, copy=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config YAML root must be a mapping.")
    return payload


@dataclass(frozen=True)
class MotorsConfig:
    ids: tuple[int, ...]
    names: tuple[str, ...]
    enabled_ids: tuple[int, ...]

    def name_for(self, motor_id: int) -> str:
        try:
            index = self.ids.index(int(motor_id))
        except ValueError as exc:
            raise KeyError(f"Unknown motor_id: {motor_id}") from exc
        return self.names[index]


@dataclass(frozen=True)
class TransportConfig:
    read_timeout: float
    read_chunk_size: int
    flush_input_before_round: bool
    sync_timeout: float
    interface: str
    nominal_bitrate: int
    data_bitrate: int
    configure_interface: bool
    force_fd: bool
    motor_can_ids: tuple[int, ...]
    motor_mst_ids: tuple[int, ...]
    motor_types: tuple[str, ...]


@dataclass(frozen=True)
class SafetyConfig:
    hard_speed_abort_abs: float
    moving_velocity_threshold: float
    moving_hold_ms: int
    post_abort_disable_delay_ms: int


@dataclass(frozen=True)
class BreakawayConfig:
    torque_step: float
    hold_duration: float
    scan_max_torque: np.ndarray


@dataclass(frozen=True)
class MitVelocityConfig:
    kd_speed: np.ndarray
    ramp_acceleration: float
    steady_hold_duration: float
    steady_window_ratio: float


@dataclass(frozen=True)
class LowSpeedConfig:
    enabled: bool
    speed_points: tuple[float, ...]
    ramp_acceleration: float
    hold_duration: float
    micro_motion_enabled: bool
    micro_motion_velocity_limit: float
    micro_motion_frequency_hz: float
    micro_motion_kp: float
    micro_motion_kd: float
    micro_motion_record_duration: float
    train_valid_split: float


@dataclass(frozen=True)
class InertiaConfig:
    waypoints: tuple[float, ...]


@dataclass(frozen=True)
class DynamicMitConfig:
    enabled: bool
    trajectory_type: str
    position_amplitude: float
    velocity_limit: float
    frequency_hz: float
    frequency_range_hz: tuple[float, float]
    cycle_count: int
    warmup_duration: float
    record_duration: float
    kp: float
    kd: float
    train_valid_split: float
    max_position_error: float
    max_velocity_error: float
    saturation_torque_ratio: float
    use_mit_estimated_torque: bool
    use_for_publish: bool
    min_fit_sample_count: int


@dataclass(frozen=True)
class IdentificationConfig:
    steady_speed_points: tuple[float, ...]
    repeat_count: int
    generation_safety_margin_ratio: float
    phase_quality_retry_count: int
    min_low_speed_fit_samples_per_direction: int
    savgol_window: int
    savgol_polyorder: int
    inertia_savgol_window_candidates: tuple[int, ...]
    joint_speed_hold_weight: float
    joint_inertia_weight: float
    joint_dynamic_mit_weight: float
    min_joint_fit_sample_count: int
    min_tracking_ratio: float
    max_steady_velocity_std_ratio: float
    min_publishable_rounds: int
    min_platform_sample_count: int
    friction_rmse_publish_threshold: float
    inertia_rmse_publish_threshold: float


@dataclass(frozen=True)
class CompensationConfig:
    require_published_model: bool
    torque_limit_ratio: float
    torque_slew_rate_nm_s: float
    static_velocity_threshold_rad_s: float
    static_transition_velocity_rad_s: float
    soft_abort_start_ratio: float
    soft_abort_stop_ratio: float
    max_inertia_torque_ratio: float
    direction_hold_samples: int


@dataclass(frozen=True)
class OutputConfig:
    results_dir: Path
    summary_filename: str
    summary_csv_filename: str
    summary_report_filename: str
    latest_parameters_json_filename: str


@dataclass(frozen=True)
class Config:
    motors: MotorsConfig
    transport: TransportConfig
    safety: SafetyConfig
    breakaway: BreakawayConfig
    mit_velocity: MitVelocityConfig
    low_speed: LowSpeedConfig
    inertia: InertiaConfig
    dynamic_mit: DynamicMitConfig
    identification: IdentificationConfig
    compensation: CompensationConfig
    output: OutputConfig
    config_path: Path
    project_root: Path = PROJECT_ROOT

    @property
    def motor_ids(self) -> tuple[int, ...]:
        return self.motors.ids

    @property
    def enabled_motor_ids(self) -> tuple[int, ...]:
        return self.motors.enabled_ids

    @property
    def motor_count(self) -> int:
        return len(self.motors.ids)

    @property
    def group_count(self) -> int:
        return int(self.identification.repeat_count)

    @property
    def results_dir(self) -> Path:
        return self.output.results_dir

    def motor_index(self, motor_id: int) -> int:
        try:
            return self.motors.ids.index(int(motor_id))
        except ValueError as exc:
            raise KeyError(f"Unknown motor_id: {motor_id}") from exc

    def resolve_project_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()


def _parse_motors(raw: dict[str, Any]) -> MotorsConfig:
    motor_ids = tuple(sorted(set(_as_int_tuple(raw.get("ids", range(1, 8))))))
    if not motor_ids:
        raise ValueError("motors.ids must not be empty.")
    if motor_ids != tuple(range(min(motor_ids), max(motor_ids) + 1)):
        raise ValueError("motors.ids must be a contiguous ascending sequence.")

    names_raw = raw.get("names")
    if names_raw is None:
        names = tuple(f"motor_{motor_id:02d}" for motor_id in motor_ids)
    else:
        names = tuple(str(item) for item in names_raw)
        if len(names) != len(motor_ids):
            raise ValueError("motors.names must match motors.ids length.")

    enabled_ids = tuple(sorted(set(_as_int_tuple(raw.get("enabled", motor_ids)))))
    if not enabled_ids:
        raise ValueError("motors.enabled must not be empty.")
    for motor_id in enabled_ids:
        if motor_id not in motor_ids:
            raise ValueError(f"Enabled motor_id {motor_id} is not present in motors.ids.")

    return MotorsConfig(ids=motor_ids, names=names, enabled_ids=enabled_ids)


def _parse_transport(raw: dict[str, Any], motor_ids: tuple[int, ...]) -> TransportConfig:
    motor_can_ids = _as_int_tuple(raw.get("motor_can_ids", motor_ids))
    motor_mst_ids = _as_int_tuple(raw.get("motor_mst_ids", tuple(0x10 + motor_id for motor_id in motor_ids)))
    motor_types = tuple(str(item) for item in raw.get("motor_types", ("DM4310",) * len(motor_ids)))
    return TransportConfig(
        read_timeout=float(raw.get("read_timeout", 0.02)),
        read_chunk_size=max(int(raw.get("read_chunk_size", 256)), 19),
        flush_input_before_round=bool(raw.get("flush_input_before_round", True)),
        sync_timeout=max(float(raw.get("sync_timeout", 2.0)), 0.1),
        interface=str(raw.get("interface", "can0")),
        nominal_bitrate=int(raw.get("nominal_bitrate", 1_000_000)),
        data_bitrate=int(raw.get("data_bitrate", 5_000_000)),
        configure_interface=bool(raw.get("configure_interface", False)),
        force_fd=bool(raw.get("force_fd", True)),
        motor_can_ids=motor_can_ids,
        motor_mst_ids=motor_mst_ids,
        motor_types=motor_types,
    )


def _parse_safety(raw: dict[str, Any]) -> SafetyConfig:
    return SafetyConfig(
        hard_speed_abort_abs=float(raw.get("hard_speed_abort_abs", 10.0)),
        moving_velocity_threshold=float(raw.get("moving_velocity_threshold", 0.2)),
        moving_hold_ms=int(raw.get("moving_hold_ms", 50)),
        post_abort_disable_delay_ms=int(raw.get("post_abort_disable_delay_ms", 80)),
    )


def _parse_breakaway(raw: dict[str, Any], motor_count: int) -> BreakawayConfig:
    return BreakawayConfig(
        torque_step=float(raw.get("torque_step", 0.01)),
        hold_duration=float(raw.get("hold_duration", 0.25)),
        scan_max_torque=_expand_float_vector(
            raw.get("scan_max_torque", 0.80),
            motor_count,
            name="breakaway.scan_max_torque",
        ),
    )


def _parse_mit_velocity(raw: dict[str, Any], motor_count: int) -> MitVelocityConfig:
    return MitVelocityConfig(
        kd_speed=_expand_float_vector(raw.get("kd_speed", 1.0), motor_count, name="mit_velocity.kd_speed"),
        ramp_acceleration=float(raw.get("ramp_acceleration", 2.0)),
        steady_hold_duration=float(raw.get("steady_hold_duration", 1.0)),
        steady_window_ratio=float(raw.get("steady_window_ratio", 0.5)),
    )


def _parse_low_speed(raw: dict[str, Any]) -> LowSpeedConfig:
    return LowSpeedConfig(
        enabled=bool(raw.get("enabled", True)),
        speed_points=tuple(float(item) for item in raw.get("speed_points", (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00))),
        ramp_acceleration=float(raw.get("ramp_acceleration", 0.5)),
        hold_duration=float(raw.get("hold_duration", 1.0)),
        micro_motion_enabled=bool(raw.get("micro_motion_enabled", True)),
        micro_motion_velocity_limit=float(raw.get("micro_motion_velocity_limit", 0.12)),
        micro_motion_frequency_hz=float(raw.get("micro_motion_frequency_hz", 0.5)),
        micro_motion_kp=float(raw.get("micro_motion_kp", 0.0)),
        micro_motion_kd=float(raw.get("micro_motion_kd", 0.25)),
        micro_motion_record_duration=float(raw.get("micro_motion_record_duration", 6.0)),
        train_valid_split=float(raw.get("train_valid_split", 0.7)),
    )


def _parse_inertia(raw: dict[str, Any]) -> InertiaConfig:
    default_waypoints = (0.0, 2.0, 4.0, 6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0, -4.0, -2.0, 0.0)
    return InertiaConfig(
        waypoints=tuple(float(item) for item in raw.get("waypoints", default_waypoints)),
    )


def _parse_frequency_range(raw_value: Any) -> tuple[float, float]:
    values = tuple(float(item) for item in raw_value)
    if len(values) != 2:
        raise ValueError("dynamic_mit.frequency_range_hz must contain exactly 2 values.")
    return (float(values[0]), float(values[1]))


def _parse_dynamic_mit(raw: dict[str, Any]) -> DynamicMitConfig:
    return DynamicMitConfig(
        enabled=bool(raw.get("enabled", True)),
        trajectory_type=str(raw.get("trajectory_type", "sine")),
        position_amplitude=float(raw.get("position_amplitude", 0.5)),
        velocity_limit=float(raw.get("velocity_limit", 4.0)),
        frequency_hz=float(raw.get("frequency_hz", 0.25)),
        frequency_range_hz=_parse_frequency_range(raw.get("frequency_range_hz", (0.1, 1.0))),
        cycle_count=max(int(raw.get("cycle_count", 3)), 1),
        warmup_duration=float(raw.get("warmup_duration", 0.5)),
        record_duration=float(raw.get("record_duration", 8.0)),
        kp=float(raw.get("kp", 8.0)),
        kd=float(raw.get("kd", 0.8)),
        train_valid_split=float(raw.get("train_valid_split", 0.7)),
        max_position_error=float(raw.get("max_position_error", 0.5)),
        max_velocity_error=float(raw.get("max_velocity_error", 2.0)),
        saturation_torque_ratio=float(raw.get("saturation_torque_ratio", 0.95)),
        use_mit_estimated_torque=bool(raw.get("use_mit_estimated_torque", False)),
        use_for_publish=bool(raw.get("use_for_publish", False)),
        min_fit_sample_count=max(int(raw.get("min_fit_sample_count", 50)), 1),
    )


def _parse_identification(raw: dict[str, Any]) -> IdentificationConfig:
    steady_speed_points = tuple(float(item) for item in raw.get("steady_speed_points", (0.5, 1.0, 2.0, 4.0, 6.0)))
    savgol_window = max(int(raw.get("savgol_window", 21)), 3)
    savgol_polyorder = max(int(raw.get("savgol_polyorder", 3)), 1)
    inertia_candidates_raw = raw.get("inertia_savgol_window_candidates", (savgol_window,))
    if inertia_candidates_raw is None:
        inertia_savgol_window_candidates = ()
    elif isinstance(inertia_candidates_raw, (int, float, str)):
        inertia_savgol_window_candidates = (int(inertia_candidates_raw),)
    else:
        inertia_savgol_window_candidates = tuple(int(item) for item in inertia_candidates_raw)
    return IdentificationConfig(
        steady_speed_points=steady_speed_points,
        repeat_count=max(int(raw.get("repeat_count", 3)), 1),
        generation_safety_margin_ratio=float(raw.get("generation_safety_margin_ratio", 0.80)),
        phase_quality_retry_count=max(int(raw.get("phase_quality_retry_count", 1)), 0),
        min_low_speed_fit_samples_per_direction=max(int(raw.get("min_low_speed_fit_samples_per_direction", 8)), 1),
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        inertia_savgol_window_candidates=inertia_savgol_window_candidates,
        joint_speed_hold_weight=float(raw.get("joint_speed_hold_weight", 3.0)),
        joint_inertia_weight=float(raw.get("joint_inertia_weight", 1.0)),
        joint_dynamic_mit_weight=float(raw.get("joint_dynamic_mit_weight", 1.0)),
        min_joint_fit_sample_count=max(int(raw.get("min_joint_fit_sample_count", 20)), 1),
        min_tracking_ratio=float(raw.get("min_tracking_ratio", 0.75)),
        max_steady_velocity_std_ratio=float(raw.get("max_steady_velocity_std_ratio", 0.15)),
        min_publishable_rounds=max(int(raw.get("min_publishable_rounds", 2)), 1),
        min_platform_sample_count=max(int(raw.get("min_platform_sample_count", 3)), 1),
        friction_rmse_publish_threshold=float(raw.get("friction_rmse_publish_threshold", 0.15)),
        inertia_rmse_publish_threshold=float(raw.get("inertia_rmse_publish_threshold", 0.20)),
    )


def _parse_compensation(raw: dict[str, Any]) -> CompensationConfig:
    return CompensationConfig(
        require_published_model=bool(raw.get("require_published_model", True)),
        torque_limit_ratio=float(raw.get("torque_limit_ratio", 0.25)),
        torque_slew_rate_nm_s=float(raw.get("torque_slew_rate_nm_s", 2.0)),
        static_velocity_threshold_rad_s=float(raw.get("static_velocity_threshold_rad_s", 0.20)),
        static_transition_velocity_rad_s=float(raw.get("static_transition_velocity_rad_s", 0.50)),
        soft_abort_start_ratio=float(raw.get("soft_abort_start_ratio", 0.60)),
        soft_abort_stop_ratio=float(raw.get("soft_abort_stop_ratio", 0.85)),
        max_inertia_torque_ratio=float(raw.get("max_inertia_torque_ratio", 0.25)),
        direction_hold_samples=max(int(raw.get("direction_hold_samples", 3)), 1),
    )


def _parse_output(raw: dict[str, Any], *, project_root: Path) -> OutputConfig:
    results_dir = Path(raw.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = (project_root / results_dir).resolve()
    return OutputConfig(
        results_dir=results_dir,
        summary_filename=str(raw.get("summary_filename", "hardware_identification_summary.npz")),
        summary_csv_filename=str(raw.get("summary_csv_filename", "hardware_identification_summary.csv")),
        summary_report_filename=str(raw.get("summary_report_filename", "hardware_identification_summary.md")),
        latest_parameters_json_filename=str(raw.get("latest_parameters_json_filename", "latest_motor_parameters.json")),
    )


def load_config(path: str | Path) -> Config:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    payload = _load_yaml(candidate)

    rejected = sorted(_REJECTED_TOP_LEVEL_KEYS & payload.keys())
    if rejected:
        raise ValueError("Unsupported legacy config section(s): " + ", ".join(rejected))

    motors = _parse_motors(payload.get("motors", {}))
    config = Config(
        motors=motors,
        transport=_parse_transport(payload.get("transport", {}), motors.ids),
        safety=_parse_safety(payload.get("safety", {})),
        breakaway=_parse_breakaway(payload.get("breakaway", {}), len(motors.ids)),
        mit_velocity=_parse_mit_velocity(payload.get("mit_velocity", {}), len(motors.ids)),
        low_speed=_parse_low_speed(payload.get("low_speed", {})),
        inertia=_parse_inertia(payload.get("inertia", {})),
        dynamic_mit=_parse_dynamic_mit(payload.get("dynamic_mit", {})),
        identification=_parse_identification(payload.get("identification", {})),
        compensation=_parse_compensation(payload.get("compensation", {})),
        output=_parse_output(payload.get("output", {}), project_root=PROJECT_ROOT),
        config_path=candidate,
    )

    from send import damiao as damiao_socketcan

    motor_count = len(config.motor_ids)
    if len(config.transport.motor_can_ids) != motor_count:
        raise ValueError("transport.motor_can_ids must match motors.ids length.")
    if len(config.transport.motor_mst_ids) != motor_count:
        raise ValueError("transport.motor_mst_ids must match motors.ids length.")
    if len(config.transport.motor_types) != motor_count:
        raise ValueError("transport.motor_types must match motors.ids length.")
    if len(set(config.transport.motor_can_ids)) != motor_count:
        raise ValueError("transport.motor_can_ids must be unique.")
    if len(set(config.transport.motor_mst_ids)) != motor_count:
        raise ValueError("transport.motor_mst_ids must be unique.")
    valid_motor_types = {member.name for member in damiao_socketcan.DM_Motor_Type}
    invalid_motor_types = sorted(set(config.transport.motor_types) - valid_motor_types)
    if invalid_motor_types:
        raise ValueError("transport.motor_types contains unsupported values: " + ", ".join(invalid_motor_types))

    if config.transport.read_timeout < 0.0:
        raise ValueError("transport.read_timeout must be >= 0.")
    if config.transport.nominal_bitrate <= 0:
        raise ValueError("transport.nominal_bitrate must be > 0.")
    if config.transport.data_bitrate <= 0:
        raise ValueError("transport.data_bitrate must be > 0.")

    if not np.isfinite(float(config.safety.hard_speed_abort_abs)) or config.safety.hard_speed_abort_abs <= 0.0:
        raise ValueError("safety.hard_speed_abort_abs must be > 0.")
    if not np.isfinite(float(config.safety.moving_velocity_threshold)) or config.safety.moving_velocity_threshold <= 0.0:
        raise ValueError("safety.moving_velocity_threshold must be > 0.")
    if int(config.safety.moving_hold_ms) <= 0:
        raise ValueError("safety.moving_hold_ms must be > 0.")
    if int(config.safety.post_abort_disable_delay_ms) <= 0:
        raise ValueError("safety.post_abort_disable_delay_ms must be > 0.")

    if not np.isfinite(float(config.breakaway.torque_step)) or config.breakaway.torque_step <= 0.0:
        raise ValueError("breakaway.torque_step must be > 0.")
    if not np.isfinite(float(config.breakaway.hold_duration)) or config.breakaway.hold_duration <= 0.0:
        raise ValueError("breakaway.hold_duration must be > 0.")
    if np.any(~np.isfinite(config.breakaway.scan_max_torque)) or np.any(config.breakaway.scan_max_torque <= 0.0):
        raise ValueError("breakaway.scan_max_torque must all be > 0.")

    if np.any(~np.isfinite(config.mit_velocity.kd_speed)) or np.any(config.mit_velocity.kd_speed < 0.0):
        raise ValueError("mit_velocity.kd_speed must all be >= 0.")
    if not np.isfinite(float(config.mit_velocity.ramp_acceleration)) or config.mit_velocity.ramp_acceleration <= 0.0:
        raise ValueError("mit_velocity.ramp_acceleration must be > 0.")
    if not np.isfinite(float(config.mit_velocity.steady_hold_duration)) or config.mit_velocity.steady_hold_duration <= 0.0:
        raise ValueError("mit_velocity.steady_hold_duration must be > 0.")
    if not (0.0 < float(config.mit_velocity.steady_window_ratio) <= 1.0):
        raise ValueError("mit_velocity.steady_window_ratio must be within (0, 1].")

    if not config.low_speed.speed_points:
        raise ValueError("low_speed.speed_points must not be empty.")
    if any(not np.isfinite(float(point)) or float(point) <= 0.0 for point in config.low_speed.speed_points):
        raise ValueError("low_speed.speed_points must all be > 0.")
    if max(abs(float(point)) for point in config.low_speed.speed_points) >= float(config.safety.hard_speed_abort_abs):
        raise ValueError("low_speed.speed_points must be < safety.hard_speed_abort_abs.")
    if not np.isfinite(float(config.low_speed.ramp_acceleration)) or config.low_speed.ramp_acceleration <= 0.0:
        raise ValueError("low_speed.ramp_acceleration must be > 0.")
    if not np.isfinite(float(config.low_speed.hold_duration)) or config.low_speed.hold_duration < 0.0:
        raise ValueError("low_speed.hold_duration must be >= 0.")
    if not np.isfinite(float(config.low_speed.micro_motion_velocity_limit)) or config.low_speed.micro_motion_velocity_limit <= 0.0:
        raise ValueError("low_speed.micro_motion_velocity_limit must be > 0.")
    if float(config.low_speed.micro_motion_velocity_limit) >= float(config.safety.hard_speed_abort_abs):
        raise ValueError("low_speed.micro_motion_velocity_limit must be < safety.hard_speed_abort_abs.")
    if not np.isfinite(float(config.low_speed.micro_motion_frequency_hz)) or config.low_speed.micro_motion_frequency_hz <= 0.0:
        raise ValueError("low_speed.micro_motion_frequency_hz must be > 0.")
    if not np.isfinite(float(config.low_speed.micro_motion_kp)) or config.low_speed.micro_motion_kp < 0.0:
        raise ValueError("low_speed.micro_motion_kp must be >= 0.")
    if not np.isfinite(float(config.low_speed.micro_motion_kd)) or config.low_speed.micro_motion_kd < 0.0:
        raise ValueError("low_speed.micro_motion_kd must be >= 0.")
    if not np.isfinite(float(config.low_speed.micro_motion_record_duration)) or config.low_speed.micro_motion_record_duration < 0.0:
        raise ValueError("low_speed.micro_motion_record_duration must be >= 0.")
    if not (0.0 < float(config.low_speed.train_valid_split) < 1.0):
        raise ValueError("low_speed.train_valid_split must be within (0, 1).")

    if len(config.inertia.waypoints) < 2:
        raise ValueError("inertia.waypoints must contain at least 2 values.")
    if any(not np.isfinite(float(point)) for point in config.inertia.waypoints):
        raise ValueError("inertia.waypoints must all be finite.")
    if abs(float(config.inertia.waypoints[0])) > 1.0e-9 or abs(float(config.inertia.waypoints[-1])) > 1.0e-9:
        raise ValueError("inertia.waypoints must start and end at 0.")

    if config.dynamic_mit.trajectory_type not in {"sine", "chirp", "trapezoid_velocity"}:
        raise ValueError("dynamic_mit.trajectory_type must be one of: sine, chirp, trapezoid_velocity.")
    if not np.isfinite(float(config.dynamic_mit.position_amplitude)) or config.dynamic_mit.position_amplitude <= 0.0:
        raise ValueError("dynamic_mit.position_amplitude must be > 0.")
    if not np.isfinite(float(config.dynamic_mit.velocity_limit)) or config.dynamic_mit.velocity_limit <= 0.0:
        raise ValueError("dynamic_mit.velocity_limit must be > 0.")
    if float(config.dynamic_mit.velocity_limit) >= float(config.safety.hard_speed_abort_abs):
        raise ValueError("dynamic_mit.velocity_limit must be < safety.hard_speed_abort_abs.")
    if not np.isfinite(float(config.dynamic_mit.frequency_hz)) or config.dynamic_mit.frequency_hz <= 0.0:
        raise ValueError("dynamic_mit.frequency_hz must be > 0.")
    low_freq, high_freq = config.dynamic_mit.frequency_range_hz
    if not (np.isfinite(low_freq) and np.isfinite(high_freq) and 0.0 < low_freq < high_freq):
        raise ValueError("dynamic_mit.frequency_range_hz must be finite and increasing.")
    if int(config.dynamic_mit.cycle_count) <= 0:
        raise ValueError("dynamic_mit.cycle_count must be > 0.")
    if not np.isfinite(float(config.dynamic_mit.warmup_duration)) or config.dynamic_mit.warmup_duration < 0.0:
        raise ValueError("dynamic_mit.warmup_duration must be >= 0.")
    if not np.isfinite(float(config.dynamic_mit.record_duration)) or config.dynamic_mit.record_duration <= 0.0:
        raise ValueError("dynamic_mit.record_duration must be > 0.")
    if not np.isfinite(float(config.dynamic_mit.kp)) or config.dynamic_mit.kp < 0.0:
        raise ValueError("dynamic_mit.kp must be >= 0.")
    if not np.isfinite(float(config.dynamic_mit.kd)) or config.dynamic_mit.kd < 0.0:
        raise ValueError("dynamic_mit.kd must be >= 0.")
    if not (0.0 < float(config.dynamic_mit.train_valid_split) < 1.0):
        raise ValueError("dynamic_mit.train_valid_split must be within (0, 1).")
    if not np.isfinite(float(config.dynamic_mit.max_position_error)) or config.dynamic_mit.max_position_error <= 0.0:
        raise ValueError("dynamic_mit.max_position_error must be > 0.")
    if not np.isfinite(float(config.dynamic_mit.max_velocity_error)) or config.dynamic_mit.max_velocity_error <= 0.0:
        raise ValueError("dynamic_mit.max_velocity_error must be > 0.")
    if not (0.0 < float(config.dynamic_mit.saturation_torque_ratio) <= 1.0):
        raise ValueError("dynamic_mit.saturation_torque_ratio must be within (0, 1].")
    if int(config.dynamic_mit.min_fit_sample_count) <= 0:
        raise ValueError("dynamic_mit.min_fit_sample_count must be > 0.")

    if not config.identification.steady_speed_points:
        raise ValueError("identification.steady_speed_points must not be empty.")
    if any(point <= 0.0 for point in config.identification.steady_speed_points):
        raise ValueError("identification.steady_speed_points must all be > 0.")
    if config.identification.repeat_count <= 0:
        raise ValueError("identification.repeat_count must be > 0.")
    if not np.isfinite(float(config.identification.generation_safety_margin_ratio)) or not (0.0 < float(config.identification.generation_safety_margin_ratio) < 1.0):
        raise ValueError("identification.generation_safety_margin_ratio must be within (0, 1).")
    if int(config.identification.phase_quality_retry_count) < 0:
        raise ValueError("identification.phase_quality_retry_count must be >= 0.")
    if int(config.identification.min_low_speed_fit_samples_per_direction) <= 0:
        raise ValueError("identification.min_low_speed_fit_samples_per_direction must be > 0.")
    if config.identification.savgol_window <= config.identification.savgol_polyorder:
        raise ValueError("identification.savgol_window must be larger than identification.savgol_polyorder.")
    if not config.identification.inertia_savgol_window_candidates:
        raise ValueError("identification.inertia_savgol_window_candidates must not be empty.")
    for window in config.identification.inertia_savgol_window_candidates:
        if int(window) < 3:
            raise ValueError("identification.inertia_savgol_window_candidates values must be >= 3.")
        if int(window) % 2 == 0:
            raise ValueError("identification.inertia_savgol_window_candidates values must be odd.")
        if int(window) <= int(config.identification.savgol_polyorder):
            raise ValueError(
                "identification.inertia_savgol_window_candidates values must be larger than "
                "identification.savgol_polyorder."
            )
    if not np.isfinite(float(config.identification.joint_speed_hold_weight)) or config.identification.joint_speed_hold_weight <= 0.0:
        raise ValueError("identification.joint_speed_hold_weight must be > 0.")
    if not np.isfinite(float(config.identification.joint_inertia_weight)) or config.identification.joint_inertia_weight <= 0.0:
        raise ValueError("identification.joint_inertia_weight must be > 0.")
    if not np.isfinite(float(config.identification.joint_dynamic_mit_weight)) or config.identification.joint_dynamic_mit_weight <= 0.0:
        raise ValueError("identification.joint_dynamic_mit_weight must be > 0.")
    if int(config.identification.min_joint_fit_sample_count) <= 0:
        raise ValueError("identification.min_joint_fit_sample_count must be > 0.")
    if not (0.0 < float(config.identification.min_tracking_ratio) <= 1.0):
        raise ValueError("identification.min_tracking_ratio must be within (0, 1].")
    if not np.isfinite(float(config.identification.max_steady_velocity_std_ratio)) or config.identification.max_steady_velocity_std_ratio <= 0.0:
        raise ValueError("identification.max_steady_velocity_std_ratio must be > 0.")
    if int(config.identification.min_publishable_rounds) <= 0:
        raise ValueError("identification.min_publishable_rounds must be > 0.")
    if int(config.identification.min_platform_sample_count) <= 0:
        raise ValueError("identification.min_platform_sample_count must be > 0.")
    if not np.isfinite(float(config.identification.friction_rmse_publish_threshold)) or config.identification.friction_rmse_publish_threshold <= 0.0:
        raise ValueError("identification.friction_rmse_publish_threshold must be > 0.")
    if not np.isfinite(float(config.identification.inertia_rmse_publish_threshold)) or config.identification.inertia_rmse_publish_threshold <= 0.0:
        raise ValueError("identification.inertia_rmse_publish_threshold must be > 0.")

    if not np.isfinite(float(config.compensation.torque_limit_ratio)) or not (0.0 < float(config.compensation.torque_limit_ratio) <= 1.0):
        raise ValueError("compensation.torque_limit_ratio must be within (0, 1].")
    if not np.isfinite(float(config.compensation.torque_slew_rate_nm_s)) or config.compensation.torque_slew_rate_nm_s <= 0.0:
        raise ValueError("compensation.torque_slew_rate_nm_s must be > 0.")
    if not np.isfinite(float(config.compensation.static_velocity_threshold_rad_s)) or config.compensation.static_velocity_threshold_rad_s < 0.0:
        raise ValueError("compensation.static_velocity_threshold_rad_s must be >= 0.")
    if (
        not np.isfinite(float(config.compensation.static_transition_velocity_rad_s))
        or float(config.compensation.static_transition_velocity_rad_s) <= float(config.compensation.static_velocity_threshold_rad_s)
    ):
        raise ValueError("compensation.static_transition_velocity_rad_s must be > compensation.static_velocity_threshold_rad_s.")
    if not np.isfinite(float(config.compensation.soft_abort_start_ratio)) or not (0.0 < float(config.compensation.soft_abort_start_ratio) < 1.0):
        raise ValueError("compensation.soft_abort_start_ratio must be within (0, 1).")
    if not np.isfinite(float(config.compensation.soft_abort_stop_ratio)) or not (float(config.compensation.soft_abort_start_ratio) < float(config.compensation.soft_abort_stop_ratio) <= 1.0):
        raise ValueError("compensation.soft_abort_stop_ratio must be within (soft_abort_start_ratio, 1].")
    if not np.isfinite(float(config.compensation.max_inertia_torque_ratio)) or not (0.0 <= float(config.compensation.max_inertia_torque_ratio) <= 1.0):
        raise ValueError("compensation.max_inertia_torque_ratio must be within [0, 1].")
    if int(config.compensation.direction_hold_samples) <= 0:
        raise ValueError("compensation.direction_hold_samples must be > 0.")

    from friction_identification_core.limits import validate_configured_identification_limits

    validate_configured_identification_limits(config)

    return config


def _parse_motor_override(
    raw: str | None,
    available_ids: tuple[int, ...],
    *,
    source_name: str,
) -> tuple[int, ...] | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text == "all":
        return available_ids

    parsed: list[int] = []
    seen: set[int] = set()
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        motor_id = int(token)
        if motor_id not in available_ids:
            raise ValueError(f"motor_id {motor_id} is not present in {source_name}.")
        if motor_id in seen:
            continue
        seen.add(motor_id)
        parsed.append(motor_id)
    if not parsed:
        raise ValueError("--motors did not resolve to any valid motor_id.")
    return tuple(sorted(parsed))


def apply_overrides(
    config: Config,
    *,
    output: str | None = None,
    motors: str | None = None,
) -> Config:
    updated = config
    if output:
        updated = replace(updated, output=replace(updated.output, results_dir=updated.resolve_project_path(output)))

    overridden_motor_ids = _parse_motor_override(
        motors,
        updated.enabled_motor_ids,
        source_name="config motors.enabled",
    )
    if overridden_motor_ids is not None:
        updated = replace(updated, motors=replace(updated.motors, enabled_ids=overridden_motor_ids))

    return updated


__all__ = [
    "BreakawayConfig",
    "Config",
    "DEFAULT_CONFIG_PATH",
    "DynamicMitConfig",
    "IdentificationConfig",
    "InertiaConfig",
    "LowSpeedConfig",
    "MitVelocityConfig",
    "MotorsConfig",
    "OutputConfig",
    "PROJECT_ROOT",
    "SafetyConfig",
    "TransportConfig",
    "apply_overrides",
    "load_config",
]
