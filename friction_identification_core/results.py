from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from friction_identification_core.core import (
    PIECEWISE_STATIC_LINEAR_KIND,
    IdentifiedMotorModel,
    MotorIdentificationResult,
    RoundCapture,
    piecewise_static_linear_torque,
)
from friction_identification_core.limits import identification_limits_for_motor
from friction_identification_core.runtime_config import Config


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def log_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def utc_now_iso8601() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def filesystem_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with open(target, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {target}")
    return payload


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return value


def _json_scalar(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(_normalize_json_value(payload), ensure_ascii=False))


def _finite_median_or(values: list[float], fallback: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size:
        return float(np.nanmedian(finite))
    return float(fallback)


def _artifact_model_metric_values(artifacts: list["RoundArtifact"], name: str) -> list[float]:
    values: list[float] = []
    for artifact in artifacts:
        friction_model = artifact.identification.metadata.get("friction_model", {})
        if not isinstance(friction_model, dict) or name not in friction_model:
            continue
        try:
            values.append(float(friction_model[name]))
        except (TypeError, ValueError):
            continue
    return values


def _configured_acceleration_abs(config: Config, *, motor_id: int) -> float:
    candidates = [
        abs(float(config.mit_velocity.ramp_acceleration)),
        abs(float(config.low_speed.ramp_acceleration)) if bool(config.low_speed.enabled) else 0.0,
    ]
    if bool(config.dynamic_mit.enabled):
        from friction_identification_core.phases.dynamic_mit import build_dynamic_mit_trajectory

        trajectory = build_dynamic_mit_trajectory(config, target_motor_id=int(motor_id))
        if trajectory.acceleration.size:
            candidates.append(float(np.nanmax(np.abs(trajectory.acceleration))))
    finite = [float(item) for item in candidates if np.isfinite(float(item))]
    return max(finite) if finite else 0.0


def _piecewise_model_envelope(
    config: Config,
    *,
    motor_id: int,
    tau_static: float,
    tau_bias: float,
    tau_c: float,
    viscous: float,
    inertia: float,
) -> dict[str, Any]:
    limits = identification_limits_for_motor(config, target_motor_id=int(motor_id))
    velocity_abs = float(limits.identification_speed_abs)
    acceleration_abs = _configured_acceleration_abs(config, motor_id=int(motor_id))
    velocities = np.linspace(-velocity_abs, velocity_abs, 401, dtype=np.float64)
    accelerations = np.asarray([-acceleration_abs, 0.0, acceleration_abs], dtype=np.float64)
    max_abs_torque = 0.0
    for acceleration in accelerations:
        direction = np.sign(velocities)
        torque = piecewise_static_linear_torque(
            velocities,
            acceleration=float(acceleration),
            direction=direction,
            tau_static=float(tau_static),
            tau_c=float(tau_c),
            viscous=float(viscous),
            tau_bias=float(tau_bias),
            inertia=float(inertia),
            static_velocity_threshold_rad_s=float(config.compensation.static_velocity_threshold_rad_s),
            static_transition_velocity_rad_s=float(config.compensation.static_transition_velocity_rad_s),
        )
        max_abs_torque = max(max_abs_torque, float(np.nanmax(np.abs(torque))))
        zero_direction_torque = piecewise_static_linear_torque(
            np.asarray([0.0, 0.0], dtype=np.float64),
            acceleration=float(acceleration),
            direction=np.asarray([-1.0, 1.0], dtype=np.float64),
            tau_static=float(tau_static),
            tau_c=float(tau_c),
            viscous=float(viscous),
            tau_bias=float(tau_bias),
            inertia=float(inertia),
            static_velocity_threshold_rad_s=float(config.compensation.static_velocity_threshold_rad_s),
            static_transition_velocity_rad_s=float(config.compensation.static_transition_velocity_rad_s),
        )
        max_abs_torque = max(max_abs_torque, float(np.nanmax(np.abs(zero_direction_torque))))

    max_inertia_torque = abs(float(inertia)) * float(acceleration_abs)
    torque_ok = max_abs_torque <= float(limits.compensation_torque_abs) + 1.0e-9
    inertia_ok = max_inertia_torque <= float(limits.inertia_torque_abs) + 1.0e-9
    return {
        "status": "ok" if bool(torque_ok and inertia_ok) else "model_exceeds_compensation_budget",
        "max_abs_torque": float(max_abs_torque),
        "compensation_torque_abs": float(limits.compensation_torque_abs),
        "max_inertia_torque": float(max_inertia_torque),
        "inertia_torque_abs": float(limits.inertia_torque_abs),
        "velocity_abs": float(velocity_abs),
        "acceleration_abs": float(acceleration_abs),
    }


def _aggregate_piecewise_export_model(
    artifacts: list["RoundArtifact"],
    *,
    config: Config,
    motor_id: int,
    tau_static: float,
    tau_bias: float,
    tau_c: float,
    viscous: float,
    inertia: float,
    friction_validation_rmse: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tau_static = float(tau_static) if np.isfinite(float(tau_static)) else 0.0
    tau_bias = float(tau_bias) if np.isfinite(float(tau_bias)) else 0.0
    tau_c = float(tau_c) if np.isfinite(float(tau_c)) else 0.0
    viscous = float(viscous) if np.isfinite(float(viscous)) else 0.0
    inertia = float(inertia) if np.isfinite(float(inertia)) else 0.0
    breakaway_positive = _finite_median_or(
        [float(artifact.identification.breakaway.torque_positive) for artifact in artifacts],
        abs(tau_static),
    )
    breakaway_negative = _finite_median_or(
        [float(artifact.identification.breakaway.torque_negative) for artifact in artifacts],
        -abs(tau_static),
    )
    train_rmse = _finite_median_or(_artifact_model_metric_values(artifacts, "train_rmse"), friction_validation_rmse)
    valid_rmse = _finite_median_or(_artifact_model_metric_values(artifacts, "valid_rmse"), friction_validation_rmse)
    parameters = {
        "tau_static": float(tau_static),
        "tau_c": float(tau_c),
        "viscous": float(viscous),
        "tau_bias": float(tau_bias),
        "inertia": float(inertia),
        "static_velocity_threshold_rad_s": float(config.compensation.static_velocity_threshold_rad_s),
        "static_transition_velocity_rad_s": float(config.compensation.static_transition_velocity_rad_s),
        "breakaway_positive": float(breakaway_positive),
        "breakaway_negative": float(breakaway_negative),
    }
    envelope = _piecewise_model_envelope(
        config,
        motor_id=int(motor_id),
        tau_static=float(tau_static),
        tau_bias=float(tau_bias),
        tau_c=float(tau_c),
        viscous=float(viscous),
        inertia=float(inertia),
    )
    friction_model = {
        "kind": PIECEWISE_STATIC_LINEAR_KIND,
        "equation": (
            "tau=tau_bias+viscous*v+direction*level(abs(v))+inertia*a; "
            "level=tau_static for abs(v)<=v_static, linear transition to tau_c, tau_c for abs(v)>=v_transition"
        ),
        "parameters": parameters,
        "train_rmse": float(train_rmse),
        "valid_rmse": float(valid_rmse),
        "metadata": {
            "aggregation": "median_selected_rounds",
            "selected_round_count": int(len(artifacts)),
            "envelope": envelope,
        },
    }
    export_models = {
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


def _nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if not np.any(np.isfinite(array)):
        return float("nan")
    return float(np.nanmean(array))


def _nanstd(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if not np.any(np.isfinite(array)):
        return float("nan")
    return float(np.nanstd(array))


def _nanmedian(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if not np.any(np.isfinite(array)):
        return float("nan")
    return float(np.nanmedian(array))


def latest_parameters_path(config: Config) -> Path:
    return Path(config.results_dir) / config.output.latest_parameters_json_filename


def load_latest_parameters(config: Config) -> dict[str, Any]:
    path = latest_parameters_path(config)
    if not path.exists():
        raise ValueError(f"latest motor parameters file does not exist: {path}")
    payload = read_json(path)
    motors = payload.get("motors")
    if not isinstance(motors, dict):
        raise ValueError(f"latest motor parameters file has invalid 'motors': {path}")
    return payload


@dataclass(frozen=True)
class RoundArtifact:
    capture: RoundCapture
    identification: MotorIdentificationResult
    capture_path: Path
    identification_path: Path


@dataclass(frozen=True)
class SummaryPaths:
    run_summary_path: Path
    run_summary_csv_path: Path
    run_summary_report_path: Path
    root_summary_path: Path
    root_summary_csv_path: Path
    root_summary_report_path: Path
    manifest_path: Path
    rerun_recording_path: Path


class ResultStore:
    def __init__(self, config: Config, *, mode: str) -> None:
        self._config = config
        self._mode = str(mode)
        self.results_dir = ensure_directory(config.results_dir)
        self.latest_parameters_path = self.results_dir / self._config.output.latest_parameters_json_filename
        self.run_label = f"{filesystem_timestamp()}_{self._mode}"
        self.run_dir = ensure_directory(self.results_dir / "runs" / self.run_label)
        self.summary_dir = ensure_directory(self.run_dir / "summary")
        self.rerun_recording_path = self.run_dir / f"{self._mode}.rrd"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self._manifest: dict[str, Any] = {
            "run_label": self.run_label,
            "mode": self._mode,
            "start_time": utc_now_iso8601(),
            "end_time": None,
            "repeat_count": int(config.identification.repeat_count),
            "motor_order": list(config.enabled_motor_ids),
            "capture_files": [],
            "identification_files": [],
            "summary_files": {},
            "rerun_recording_path": str(self.rerun_recording_path),
            "config_path": str(config.config_path),
        }
        self._latest_publish_payload: dict[str, np.ndarray] | None = None
        self._write_manifest()

    def _write_manifest(self) -> None:
        write_json(self.manifest_path, self._manifest)

    def record_abort_event(self, payload: dict[str, Any]) -> None:
        self._manifest["abort_event"] = _normalize_json_value(payload)
        self._write_manifest()

    def finalize(self) -> None:
        self._manifest["end_time"] = utc_now_iso8601()
        self._write_manifest()

    def _motor_dir(self, group_index: int, motor_id: int) -> Path:
        return ensure_directory(self.run_dir / f"group_{int(group_index):02d}" / f"motor_{int(motor_id):02d}")

    def save_capture(self, capture: RoundCapture) -> Path:
        path = self._motor_dir(capture.group_index, capture.target_motor_id) / "capture.npz"
        np.savez(
            path,
            time=np.asarray(capture.time, dtype=np.float64),
            motor_id=np.asarray(capture.motor_id, dtype=np.int64),
            position=np.asarray(capture.position, dtype=np.float64),
            velocity=np.asarray(capture.velocity, dtype=np.float64),
            torque_feedback=np.asarray(capture.torque_feedback, dtype=np.float64),
            command_raw=np.asarray(capture.command_raw, dtype=np.float64),
            command=np.asarray(capture.command, dtype=np.float64),
            position_cmd=np.asarray(capture.position_cmd, dtype=np.float64),
            velocity_cmd=np.asarray(capture.velocity_cmd, dtype=np.float64),
            acceleration_cmd=np.asarray(capture.acceleration_cmd, dtype=np.float64),
            phase_name=np.asarray(capture.phase_name),
            state=np.asarray(capture.state, dtype=np.uint8),
            mos_temperature=np.asarray(capture.mos_temperature, dtype=np.float64),
            id_match_ok=np.asarray(capture.id_match_ok, dtype=bool),
            filtered_velocity=np.asarray(capture.filtered_velocity, dtype=np.float64),
            estimated_acceleration=np.asarray(capture.estimated_acceleration, dtype=np.float64),
            friction_term=np.asarray(capture.friction_term, dtype=np.float64),
            inertia_term=np.asarray(capture.inertia_term, dtype=np.float64),
            guard_scale=np.asarray(capture.guard_scale, dtype=np.float64),
            stiction_evidence=np.asarray(capture.stiction_evidence, dtype=bool),
            kp_cmd=np.asarray(capture.kp_cmd, dtype=np.float64),
            kd_cmd=np.asarray(capture.kd_cmd, dtype=np.float64),
            torque_ff_cmd=np.asarray(capture.torque_ff_cmd, dtype=np.float64),
            position_error=np.asarray(capture.position_error, dtype=np.float64),
            velocity_error=np.asarray(capture.velocity_error, dtype=np.float64),
            tracking_ok=np.asarray(capture.tracking_ok, dtype=bool),
            safety_ok=np.asarray(capture.safety_ok, dtype=bool),
            state_ok=np.asarray(capture.state_ok, dtype=bool),
            saturated=np.asarray(capture.saturated, dtype=bool),
            used_for_fit=np.asarray(capture.used_for_fit, dtype=bool),
            tau_mit_est=np.asarray(capture.tau_mit_est, dtype=np.float64),
            metadata=_json_scalar(capture.metadata),
        )
        self._manifest["capture_files"].append(str(path))
        self._write_manifest()
        return path

    def save_identification(self, capture: RoundCapture, identification: MotorIdentificationResult) -> Path:
        path = self._motor_dir(capture.group_index, capture.target_motor_id) / "identification.npz"
        np.savez(
            path,
            motor_id=np.asarray(int(identification.motor_id), dtype=np.int64),
            breakaway_positive=np.asarray(float(identification.breakaway.torque_positive), dtype=np.float64),
            breakaway_negative=np.asarray(float(identification.breakaway.torque_negative), dtype=np.float64),
            tau_static=np.asarray(float(identification.breakaway.tau_static), dtype=np.float64),
            breakaway_tau_bias=np.asarray(float(identification.breakaway.tau_bias), dtype=np.float64),
            tau_c=np.asarray(float(identification.friction.tau_c), dtype=np.float64),
            viscous=np.asarray(float(identification.friction.viscous), dtype=np.float64),
            friction_tau_bias=np.asarray(float(identification.friction.tau_bias), dtype=np.float64),
            friction_train_rmse=np.asarray(float(identification.friction.train_rmse), dtype=np.float64),
            friction_valid_rmse=np.asarray(float(identification.friction.valid_rmse), dtype=np.float64),
            inertia=np.asarray(float(identification.inertia.inertia), dtype=np.float64),
            inertia_train_rmse=np.asarray(float(identification.inertia.train_rmse), dtype=np.float64),
            inertia_valid_rmse=np.asarray(float(identification.inertia.valid_rmse), dtype=np.float64),
            validation_friction_rmse=np.asarray(float(identification.validation.friction_rmse), dtype=np.float64),
            validation_inertia_rmse=np.asarray(float(identification.validation.inertia_rmse), dtype=np.float64),
            recommended_for_compensation=np.asarray(
                bool(identification.validation.recommended_for_compensation),
                dtype=bool,
            ),
            friction_train_mask=np.asarray(identification.friction.train_mask, dtype=bool),
            friction_valid_mask=np.asarray(identification.friction.valid_mask, dtype=bool),
            friction_torque_pred=np.asarray(identification.friction.torque_pred, dtype=np.float64),
            friction_torque_target=np.asarray(identification.friction.torque_target, dtype=np.float64),
            inertia_train_mask=np.asarray(identification.inertia.train_mask, dtype=bool),
            inertia_valid_mask=np.asarray(identification.inertia.valid_mask, dtype=bool),
            inertia_torque_pred=np.asarray(identification.inertia.torque_pred, dtype=np.float64),
            inertia_torque_target=np.asarray(identification.inertia.torque_target, dtype=np.float64),
            filtered_velocity=np.asarray(identification.inertia.filtered_velocity, dtype=np.float64),
            acceleration=np.asarray(identification.inertia.acceleration, dtype=np.float64),
            metadata=_json_scalar(identification.metadata),
            breakaway_metadata=_json_scalar(identification.breakaway.metadata),
            friction_metadata=_json_scalar(identification.friction.metadata),
            inertia_metadata=_json_scalar(identification.inertia.metadata),
            validation_metadata=_json_scalar(identification.validation.metadata),
        )
        self._manifest["identification_files"].append(str(path))
        self._write_manifest()
        return path

    def save_summary(self, artifacts: list[RoundArtifact]) -> SummaryPaths:
        payload = self._latest_publish_payload
        if payload is None:
            payload = self._build_summary_payload(artifacts, existing_latest=self._load_existing_latest_parameters())

        run_summary_path = self.summary_dir / self._config.output.summary_filename
        run_summary_csv_path = self.summary_dir / self._config.output.summary_csv_filename
        run_summary_report_path = self.summary_dir / self._config.output.summary_report_filename
        np.savez(run_summary_path, **payload)
        self._write_summary_csv(run_summary_csv_path, payload)
        self._write_summary_report(run_summary_report_path, payload)

        root_summary_path = self.results_dir / self._config.output.summary_filename
        root_summary_csv_path = self.results_dir / self._config.output.summary_csv_filename
        root_summary_report_path = self.results_dir / self._config.output.summary_report_filename
        shutil.copyfile(run_summary_path, root_summary_path)
        shutil.copyfile(run_summary_csv_path, root_summary_csv_path)
        shutil.copyfile(run_summary_report_path, root_summary_report_path)

        self._manifest["summary_files"] = {
            "run_summary_path": str(run_summary_path),
            "run_summary_csv_path": str(run_summary_csv_path),
            "run_summary_report_path": str(run_summary_report_path),
            "root_summary_path": str(root_summary_path),
            "root_summary_csv_path": str(root_summary_csv_path),
            "root_summary_report_path": str(root_summary_report_path),
        }
        self.finalize()
        return SummaryPaths(
            run_summary_path=run_summary_path,
            run_summary_csv_path=run_summary_csv_path,
            run_summary_report_path=run_summary_report_path,
            root_summary_path=root_summary_path,
            root_summary_csv_path=root_summary_csv_path,
            root_summary_report_path=root_summary_report_path,
            manifest_path=self.manifest_path,
            rerun_recording_path=self.rerun_recording_path,
        )

    def save_latest_parameters(self, artifacts: list[RoundArtifact]) -> Path:
        existing = self._load_existing_latest_parameters()
        payload = self._build_summary_payload(artifacts, existing_latest=existing)
        self._latest_publish_payload = payload
        rows = self._summary_rows(payload)
        merged_motors = dict(existing.get("motors", {}))
        updated_at = utc_now_iso8601()

        for row in rows:
            if int(row["round_count"]) <= 0:
                continue
            if bool(row["recommended_for_compensation"]):
                confidence = 1.0 / (1.0 + max(float(row["repeat_consistency_score"]), 0.0))
            else:
                confidence = 0.0
            quality_flags: list[str] = []
            if float(row["friction_validation_rmse"]) > float(self._config.identification.friction_rmse_publish_threshold):
                quality_flags.append("friction_rmse_above_threshold")
            if float(row["inertia_validation_rmse"]) > float(self._config.identification.inertia_rmse_publish_threshold):
                quality_flags.append("inertia_rmse_above_threshold")
            model = IdentifiedMotorModel(
                motor_id=int(row["motor_id"]),
                motor_name=str(row["motor_name"]),
                identified_at=updated_at,
                source_run_label=self.run_label,
                tau_static=float(row["tau_static"]),
                tau_bias=float(row["tau_bias"]),
                tau_c=float(row["tau_c"]),
                viscous=float(row["viscous"]),
                inertia=float(row["inertia"]),
                friction_validation_rmse=float(row["friction_validation_rmse"]),
                inertia_validation_rmse=float(row["inertia_validation_rmse"]),
                repeat_consistency_score=float(row["repeat_consistency_score"]),
                recommended_for_compensation=bool(row["recommended_for_compensation"]),
                model_version="1.0",
                model_kind=str(row.get("model_kind", PIECEWISE_STATIC_LINEAR_KIND)),
                fit_method=(
                    "piecewise_static_linear_from_identification"
                    if str(row.get("model_kind", PIECEWISE_STATIC_LINEAR_KIND)) == PIECEWISE_STATIC_LINEAR_KIND
                    else (
                        "dynamic_mit_robust_constrained_ls"
                        if str(row.get("model_kind", PIECEWISE_STATIC_LINEAR_KIND)) == "dynamic_mit_v1"
                        else (
                            "joint_weighted_robust_constrained_ls"
                            if str(row.get("model_kind", PIECEWISE_STATIC_LINEAR_KIND)) == "joint_static_dynamic_v1"
                            else "robust_constrained_ls"
                        )
                    )
                ),
                source_phases=tuple(str(item) for item in row.get("source_phases", ("breakaway", "speed-hold", "inertia"))),
                accepted_round_count=int(row["accepted_round_count"]),
                selected_rounds=tuple(int(item) for item in row["selected_rounds"]),
                confidence=float(confidence),
                quality_flags=tuple(quality_flags),
                publish_status=str(row["publish_status"]),
                publish_detail=str(row["publish_detail"]),
                friction_model=row.get("friction_model") if isinstance(row.get("friction_model"), dict) else None,
                export_models=row.get("export_models") if isinstance(row.get("export_models"), dict) else None,
            )
            motor_key = str(int(row["motor_id"]))
            entry = model.to_latest_entry()
            previous_entry = merged_motors.get(motor_key)
            if model.publish_status != "published" and isinstance(previous_entry, dict):
                previous_published_model: dict[str, Any] | None = None
                previous_status = str(
                    previous_entry.get(
                        "publish_status",
                        "published" if bool(previous_entry.get("recommended_for_compensation", False)) else "",
                    )
                )
                if previous_status == "published":
                    previous_published_model = dict(previous_entry)
                elif isinstance(previous_entry.get("previous_published_model"), dict):
                    previous_published_model = dict(previous_entry["previous_published_model"])
                if previous_published_model is not None:
                    previous_published_model.pop("previous_published_model", None)
                    entry["previous_published_model"] = previous_published_model
            merged_motors[motor_key] = entry

        ordered_motors = {
            motor_id: merged_motors[motor_id]
            for motor_id in sorted(merged_motors, key=lambda item: int(item))
        }
        latest_payload = {
            "updated_at": updated_at,
            "results_dir": str(self.results_dir),
            "speed_limit_rad_s": float(self._config.safety.hard_speed_abort_abs),
            "motors": ordered_motors,
        }
        write_json(self.latest_parameters_path, latest_payload)
        self._manifest["latest_parameters_path"] = str(self.latest_parameters_path)
        self._write_manifest()
        return self.latest_parameters_path

    def _load_existing_latest_parameters(self) -> dict[str, Any]:
        if not self.latest_parameters_path.exists():
            return {}
        payload = read_json(self.latest_parameters_path)
        motors = payload.get("motors", {})
        if not isinstance(motors, dict):
            raise ValueError(f"latest motor parameters file has invalid 'motors': {self.latest_parameters_path}")
        return payload

    def _build_summary_payload(
        self,
        artifacts: list[RoundArtifact],
        *,
        existing_latest: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        motor_ids = list(self._config.motor_ids)
        motor_names = [self._config.motors.name_for(motor_id) for motor_id in motor_ids]
        existing_motors = {}
        if isinstance(existing_latest, dict):
            raw_motors = existing_latest.get("motors", {})
            if isinstance(raw_motors, dict):
                existing_motors = raw_motors
        count = len(motor_ids)
        round_count = np.zeros(count, dtype=np.int64)
        accepted_round_count = np.zeros(count, dtype=np.int64)
        tau_static = np.full(count, np.nan, dtype=np.float64)
        tau_static_std = np.full(count, np.nan, dtype=np.float64)
        tau_bias = np.full(count, np.nan, dtype=np.float64)
        tau_bias_std = np.full(count, np.nan, dtype=np.float64)
        tau_c = np.full(count, np.nan, dtype=np.float64)
        tau_c_std = np.full(count, np.nan, dtype=np.float64)
        viscous = np.full(count, np.nan, dtype=np.float64)
        viscous_std = np.full(count, np.nan, dtype=np.float64)
        inertia = np.full(count, np.nan, dtype=np.float64)
        inertia_std = np.full(count, np.nan, dtype=np.float64)
        friction_validation_rmse = np.full(count, np.nan, dtype=np.float64)
        inertia_validation_rmse = np.full(count, np.nan, dtype=np.float64)
        repeat_consistency_score = np.full(count, np.nan, dtype=np.float64)
        recommended_for_compensation = np.zeros(count, dtype=bool)
        publish_status = np.asarray(["not_run"] * count, dtype="<U64")
        publish_detail = np.asarray([""] * count, dtype="<U512")
        selected_rounds_json = np.asarray(["[]"] * count, dtype="<U128")
        model_kind = np.asarray([PIECEWISE_STATIC_LINEAR_KIND] * count, dtype="<U64")
        source_phases_json = np.asarray(["[]"] * count, dtype="<U256")
        friction_model_json_values = ["{}"] * count
        export_models_json_values = ["{}"] * count
        history: dict[str, list[dict[str, Any]]] = {}

        for index, motor_id in enumerate(motor_ids):
            motor_artifacts = [artifact for artifact in artifacts if artifact.capture.target_motor_id == motor_id]
            accepted_artifacts = [
                artifact
                for artifact in motor_artifacts
                if bool(artifact.identification.validation.recommended_for_compensation)
            ]
            round_count[index] = len(motor_artifacts)
            accepted_round_count[index] = len(accepted_artifacts)
            history[str(motor_id)] = []
            if not motor_artifacts:
                continue
            selected_artifacts = accepted_artifacts if accepted_artifacts else motor_artifacts
            selected_rounds = [int(artifact.capture.group_index) for artifact in accepted_artifacts]
            selected_rounds_json[index] = json.dumps(selected_rounds, ensure_ascii=False)
            selected_model_kinds = [
                str(item.identification.metadata.get("model_kind", PIECEWISE_STATIC_LINEAR_KIND))
                for item in selected_artifacts
            ]
            model_kind[index] = selected_model_kinds[0] if selected_model_kinds else PIECEWISE_STATIC_LINEAR_KIND
            selected_source_phases = []
            for item in selected_artifacts:
                phases = item.identification.metadata.get("source_phases", ("breakaway", "speed-hold", "inertia"))
                if isinstance(phases, (list, tuple)):
                    selected_source_phases.extend(str(phase) for phase in phases)
            source_phases_json[index] = json.dumps(list(dict.fromkeys(selected_source_phases)), ensure_ascii=False)

            static_values = [float(item.identification.breakaway.tau_static) for item in selected_artifacts]
            bias_values = [float(item.identification.friction.tau_bias) for item in selected_artifacts]
            coulomb_values = [float(item.identification.friction.tau_c) for item in selected_artifacts]
            viscous_values = [float(item.identification.friction.viscous) for item in selected_artifacts]
            inertia_values = [float(item.identification.inertia.inertia) for item in selected_artifacts]
            friction_rmse_values = [float(item.identification.validation.friction_rmse) for item in selected_artifacts]
            inertia_rmse_values = [float(item.identification.validation.inertia_rmse) for item in selected_artifacts]

            tau_static[index] = _nanmedian(static_values)
            tau_static_std[index] = _nanstd(static_values)
            tau_bias[index] = _nanmedian(bias_values)
            tau_bias_std[index] = _nanstd(bias_values)
            tau_c[index] = _nanmedian(coulomb_values)
            tau_c_std[index] = _nanstd(coulomb_values)
            viscous[index] = _nanmedian(viscous_values)
            viscous_std[index] = _nanstd(viscous_values)
            inertia[index] = _nanmedian(inertia_values)
            inertia_std[index] = _nanstd(inertia_values)
            friction_validation_rmse[index] = _nanmean(friction_rmse_values)
            inertia_validation_rmse[index] = _nanmean(inertia_rmse_values)
            friction_model, export_models = _aggregate_piecewise_export_model(
                selected_artifacts,
                config=self._config,
                motor_id=int(motor_id),
                tau_static=float(tau_static[index]),
                tau_bias=float(tau_bias[index]),
                tau_c=float(tau_c[index]),
                viscous=float(viscous[index]),
                inertia=float(inertia[index]),
                friction_validation_rmse=float(friction_validation_rmse[index]),
            )
            friction_model_json_values[index] = json.dumps(_normalize_json_value(friction_model), ensure_ascii=False)
            export_models_json_values[index] = json.dumps(_normalize_json_value(export_models), ensure_ascii=False)

            relative_terms: list[float] = []
            for mean_value, std_value in (
                (tau_static[index], tau_static_std[index]),
                (tau_c[index], tau_c_std[index]),
                (viscous[index], viscous_std[index]),
                (inertia[index], inertia_std[index]),
            ):
                if np.isfinite(mean_value) and np.isfinite(std_value):
                    relative_terms.append(float(std_value / max(abs(float(mean_value)), 1.0e-6)))
            repeat_consistency_score[index] = max(relative_terms) if relative_terms else float("nan")
            min_publishable_rounds = int(self._config.identification.min_publishable_rounds)
            previous_model = existing_motors.get(str(int(motor_id)))
            has_previous_published_model = False
            if isinstance(previous_model, dict):
                previous_status = str(
                    previous_model.get(
                        "publish_status",
                        "published" if bool(previous_model.get("recommended_for_compensation", False)) else "",
                    )
                )
                has_previous_published_model = (
                    previous_status == "published"
                    or isinstance(previous_model.get("previous_published_model"), dict)
                )
            recommended_for_compensation[index] = bool(accepted_round_count[index] >= min_publishable_rounds)
            if int(accepted_round_count[index]) >= min_publishable_rounds:
                publish_status[index] = "published"
                publish_detail[index] = (
                    f"published {int(accepted_round_count[index])}/{int(round_count[index])} accepted rounds"
                )
            elif int(accepted_round_count[index]) > 0:
                publish_status[index] = "not_published"
                publish_detail[index] = (
                    f"accepted_round_count={int(accepted_round_count[index])}, required={min_publishable_rounds}"
                )
            else:
                publish_status[index] = "rejected"
                publish_detail[index] = "no accepted rounds in current run"
            envelope = friction_model.get("metadata", {}).get("envelope", {}) if isinstance(friction_model, dict) else {}
            if isinstance(envelope, dict) and str(envelope.get("status", "ok")) == "model_exceeds_compensation_budget":
                recommended_for_compensation[index] = False
                publish_status[index] = "rejected"
                publish_detail[index] = (
                    "model_exceeds_compensation_budget: "
                    f"max_abs_torque={float(envelope.get('max_abs_torque', float('nan'))):.6f}, "
                    f"compensation_torque_abs={float(envelope.get('compensation_torque_abs', float('nan'))):.6f}, "
                    f"max_inertia_torque={float(envelope.get('max_inertia_torque', float('nan'))):.6f}, "
                    f"inertia_torque_abs={float(envelope.get('inertia_torque_abs', float('nan'))):.6f}"
                )
            if has_previous_published_model and publish_status[index] != "published":
                publish_detail[index] = (
                    f"{str(publish_detail[index])}; previous published model retained for reference"
                )

            for artifact in motor_artifacts:
                history[str(motor_id)].append(
                    {
                        "group_index": int(artifact.capture.group_index),
                        "round_index": int(artifact.capture.round_index),
                        "capture_path": str(artifact.capture_path),
                        "identification_path": str(artifact.identification_path),
                        "tau_static": float(artifact.identification.breakaway.tau_static),
                        "tau_bias": float(artifact.identification.friction.tau_bias),
                        "tau_c": float(artifact.identification.friction.tau_c),
                        "viscous": float(artifact.identification.friction.viscous),
                        "inertia": float(artifact.identification.inertia.inertia),
                        "friction_rmse": float(artifact.identification.validation.friction_rmse),
                        "inertia_rmse": float(artifact.identification.validation.inertia_rmse),
                        "recommended_for_compensation": bool(
                            artifact.identification.validation.recommended_for_compensation
                        ),
                        "validation_status": str(artifact.identification.validation.metadata.get("status", "unknown")),
                        "validation_detail": str(artifact.identification.validation.detail),
                        "model_kind": str(artifact.identification.metadata.get("model_kind", "static_v1")),
                        "friction_model": _normalize_json_value(
                            artifact.identification.metadata.get("friction_model", {})
                        ),
                        "export_models": _normalize_json_value(
                            artifact.identification.metadata.get("export_models", {})
                        ),
                        "inertia_savgol_window": _normalize_json_value(
                            artifact.identification.inertia.metadata.get(
                                "selected_savgol_window",
                                artifact.identification.inertia.metadata.get("savgol_window"),
                            )
                        ),
                        "inertia_savgol_candidates": _normalize_json_value(
                            artifact.identification.inertia.metadata.get("savgol_window_candidates", [])
                        ),
                        "dynamic_mit": _normalize_json_value(artifact.identification.metadata.get("dynamic_mit", {})),
                        "selected_for_publish": bool(artifact.identification.validation.recommended_for_compensation),
                    }
                )

        return {
            "motor_ids": np.asarray(motor_ids, dtype=np.int64),
            "motor_names": np.asarray(motor_names),
            "round_count": round_count,
            "accepted_round_count": accepted_round_count,
            "tau_static": tau_static,
            "tau_static_std": tau_static_std,
            "tau_bias": tau_bias,
            "tau_bias_std": tau_bias_std,
            "tau_c": tau_c,
            "tau_c_std": tau_c_std,
            "viscous": viscous,
            "viscous_std": viscous_std,
            "inertia": inertia,
            "inertia_std": inertia_std,
            "friction_validation_rmse": friction_validation_rmse,
            "inertia_validation_rmse": inertia_validation_rmse,
            "repeat_consistency_score": repeat_consistency_score,
            "recommended_for_compensation": recommended_for_compensation,
            "publish_status": np.asarray(publish_status),
            "publish_detail": np.asarray(publish_detail),
            "selected_rounds_json": np.asarray(selected_rounds_json),
            "model_kind": np.asarray(model_kind),
            "source_phases_json": np.asarray(source_phases_json),
            "friction_model_json": np.asarray(friction_model_json_values),
            "export_models_json": np.asarray(export_models_json_values),
            "history_json": np.asarray(json.dumps(history, ensure_ascii=False)),
        }

    def _summary_rows(self, payload: dict[str, np.ndarray]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        motor_ids = np.asarray(payload["motor_ids"], dtype=np.int64)
        motor_names = np.asarray(payload["motor_names"]).astype(str)
        publish_status = np.asarray(payload["publish_status"]).astype(str)
        publish_detail = np.asarray(payload["publish_detail"]).astype(str)
        selected_rounds_json = np.asarray(payload["selected_rounds_json"]).astype(str)
        model_kind = np.asarray(payload.get("model_kind", np.asarray([PIECEWISE_STATIC_LINEAR_KIND] * motor_ids.size))).astype(str)
        source_phases_json = np.asarray(payload.get("source_phases_json", np.asarray(["[]"] * motor_ids.size))).astype(str)
        friction_model_json = np.asarray(payload.get("friction_model_json", np.asarray(["{}"] * motor_ids.size))).astype(str)
        export_models_json = np.asarray(payload.get("export_models_json", np.asarray(["{}"] * motor_ids.size))).astype(str)
        for index, motor_id in enumerate(motor_ids.tolist()):
            selected_rounds = json.loads(selected_rounds_json[index]) if selected_rounds_json[index] else []
            source_phases = json.loads(source_phases_json[index]) if source_phases_json[index] else []
            friction_model = json.loads(friction_model_json[index]) if friction_model_json[index] else {}
            export_models = json.loads(export_models_json[index]) if export_models_json[index] else {}
            rows.append(
                {
                    "motor_id": int(motor_id),
                    "motor_name": str(motor_names[index]),
                    "round_count": int(payload["round_count"][index]),
                    "accepted_round_count": int(payload["accepted_round_count"][index]),
                    "tau_static": float(payload["tau_static"][index]),
                    "tau_static_std": float(payload["tau_static_std"][index]),
                    "tau_bias": float(payload["tau_bias"][index]),
                    "tau_bias_std": float(payload["tau_bias_std"][index]),
                    "tau_c": float(payload["tau_c"][index]),
                    "tau_c_std": float(payload["tau_c_std"][index]),
                    "viscous": float(payload["viscous"][index]),
                    "viscous_std": float(payload["viscous_std"][index]),
                    "inertia": float(payload["inertia"][index]),
                    "inertia_std": float(payload["inertia_std"][index]),
                    "friction_validation_rmse": float(payload["friction_validation_rmse"][index]),
                    "inertia_validation_rmse": float(payload["inertia_validation_rmse"][index]),
                    "repeat_consistency_score": float(payload["repeat_consistency_score"][index]),
                    "recommended_for_compensation": bool(payload["recommended_for_compensation"][index]),
                    "publish_status": str(publish_status[index]),
                    "publish_detail": str(publish_detail[index]),
                    "selected_rounds": list(selected_rounds),
                    "model_kind": str(model_kind[index]),
                    "source_phases": list(source_phases),
                    "friction_model": friction_model if isinstance(friction_model, dict) else {},
                    "export_models": export_models if isinstance(export_models, dict) else {},
                }
            )
        return rows

    def _write_summary_csv(self, path: Path, payload: dict[str, np.ndarray]) -> None:
        rows = self._summary_rows(payload)
        fieldnames = list(rows[0].keys()) if rows else [
            "motor_id",
            "motor_name",
            "round_count",
            "accepted_round_count",
            "tau_static",
            "tau_static_std",
            "tau_bias",
            "tau_bias_std",
            "tau_c",
            "tau_c_std",
            "viscous",
            "viscous_std",
            "inertia",
            "inertia_std",
            "friction_validation_rmse",
            "inertia_validation_rmse",
            "repeat_consistency_score",
            "recommended_for_compensation",
            "publish_status",
            "publish_detail",
            "selected_rounds",
            "model_kind",
            "source_phases",
        ]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def _write_summary_report(self, path: Path, payload: dict[str, np.ndarray]) -> None:
        rows = self._summary_rows(payload)
        history_text = str(np.asarray(payload["history_json"]).item())
        history = json.loads(history_text) if history_text else {}
        lines = [
            "# Hardware Identification Summary",
            "",
            "| Motor | accepted/total | tau_static | tau_c | viscous | inertia | friction RMSE | inertia RMSE | publish status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{int(row['motor_id']):02d} {row['motor_name']}",
                        f"{int(row['accepted_round_count'])}/{int(row['round_count'])}",
                        f"{float(row['tau_static']):.6f}",
                        f"{float(row['tau_c']):.6f}",
                        f"{float(row['viscous']):.6f}",
                        f"{float(row['inertia']):.6f}",
                        f"{float(row['friction_validation_rmse']):.6f}",
                        f"{float(row['inertia_validation_rmse']):.6f}",
                        f"{row['publish_status']} ({row['model_kind']})",
                    ]
                )
                + " |"
            )
        for row in rows:
            motor_history = history.get(str(int(row["motor_id"])), [])
            selected_rounds = ",".join(str(item) for item in row["selected_rounds"]) or "-"
            lines.extend(
                [
                    "",
                    f"## Motor {int(row['motor_id']):02d} {row['motor_name']}",
                    "",
                    f"- publish_status: `{row['publish_status']}`",
                    f"- publish_detail: `{row['publish_detail']}`",
                    f"- model_kind: `{row['model_kind']}`",
                    f"- source_phases: `{','.join(str(item) for item in row['source_phases']) or '-'}`",
                    f"- selected_rounds: `{selected_rounds}`",
                    f"- repeat_consistency_score: `{float(row['repeat_consistency_score']):.6f}`",
                ]
            )
            for item in motor_history:
                dynamic_mit = item.get("dynamic_mit", {})
                dynamic_text = ""
                if isinstance(dynamic_mit, dict) and dynamic_mit:
                    dynamic_text = (
                        f", dynamic_mit_status={dynamic_mit.get('status', 'not_run')}"
                        f", dynamic_mit_valid_rmse={float(dynamic_mit.get('valid_rmse', float('nan'))):.6f}"
                        f", dynamic_mit_use_for_publish={'yes' if bool(dynamic_mit.get('use_for_publish', False)) else 'no'}"
                    )
                inertia_window = item.get("inertia_savgol_window")
                inertia_text = ""
                if inertia_window is not None:
                    inertia_text = f", inertia_savgol_window={inertia_window}"
                lines.append(
                    "- "
                    + ", ".join(
                        [
                            f"group={int(item['group_index'])}",
                            f"round={int(item['round_index'])}",
                            f"selected_for_publish={'yes' if bool(item['selected_for_publish']) else 'no'}",
                            f"model_kind={item.get('model_kind', 'static_v1')}",
                            f"validation_status={item['validation_status']}",
                            f"friction_rmse={float(item['friction_rmse']):.6f}",
                            f"inertia_rmse={float(item['inertia_rmse']):.6f}",
                            f"detail={item['validation_detail']}{inertia_text}{dynamic_text}",
                        ]
                    )
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "ResultStore",
    "RoundArtifact",
    "SummaryPaths",
    "ensure_directory",
    "filesystem_timestamp",
    "latest_parameters_path",
    "load_latest_parameters",
    "log_info",
    "read_json",
    "utc_now_iso8601",
    "write_json",
]
