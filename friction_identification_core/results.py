from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from friction_identification_core.core import MotorIdentificationResult, RoundCapture
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
        payload = self._build_summary_payload(artifacts)

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
        payload = self._build_summary_payload(artifacts)
        rows = self._summary_rows(payload)
        existing = self._load_existing_latest_parameters()
        merged_motors = dict(existing.get("motors", {}))
        updated_at = utc_now_iso8601()

        for row in rows:
            if int(row["round_count"]) <= 0:
                continue
            merged_motors[str(int(row["motor_id"]))] = {
                "motor_id": int(row["motor_id"]),
                "motor_name": str(row["motor_name"]),
                "identified_at": updated_at,
                "source_run_label": self.run_label,
                "tau_static": float(row["tau_static"]),
                "tau_bias": float(row["tau_bias"]),
                "tau_c": float(row["tau_c"]),
                "viscous": float(row["viscous"]),
                "inertia": float(row["inertia"]),
                "friction_validation_rmse": float(row["friction_validation_rmse"]),
                "inertia_validation_rmse": float(row["inertia_validation_rmse"]),
                "repeat_consistency_score": float(row["repeat_consistency_score"]),
                "recommended_for_compensation": bool(row["recommended_for_compensation"]),
            }

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

    def _build_summary_payload(self, artifacts: list[RoundArtifact]) -> dict[str, np.ndarray]:
        motor_ids = list(self._config.motor_ids)
        motor_names = [self._config.motors.name_for(motor_id) for motor_id in motor_ids]
        count = len(motor_ids)
        round_count = np.zeros(count, dtype=np.int64)
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
        history: dict[str, list[dict[str, Any]]] = {}

        for index, motor_id in enumerate(motor_ids):
            motor_artifacts = [artifact for artifact in artifacts if artifact.capture.target_motor_id == motor_id]
            round_count[index] = len(motor_artifacts)
            history[str(motor_id)] = []
            if not motor_artifacts:
                continue

            static_values = [float(item.identification.breakaway.tau_static) for item in motor_artifacts]
            bias_values = [float(item.identification.breakaway.tau_bias) for item in motor_artifacts]
            coulomb_values = [float(item.identification.friction.tau_c) for item in motor_artifacts]
            viscous_values = [float(item.identification.friction.viscous) for item in motor_artifacts]
            inertia_values = [float(item.identification.inertia.inertia) for item in motor_artifacts]
            friction_rmse_values = [float(item.identification.validation.friction_rmse) for item in motor_artifacts]
            inertia_rmse_values = [float(item.identification.validation.inertia_rmse) for item in motor_artifacts]

            tau_static[index] = _nanmean(static_values)
            tau_static_std[index] = _nanstd(static_values)
            tau_bias[index] = _nanmean(bias_values)
            tau_bias_std[index] = _nanstd(bias_values)
            tau_c[index] = _nanmean(coulomb_values)
            tau_c_std[index] = _nanstd(coulomb_values)
            viscous[index] = _nanmean(viscous_values)
            viscous_std[index] = _nanstd(viscous_values)
            inertia[index] = _nanmean(inertia_values)
            inertia_std[index] = _nanstd(inertia_values)
            friction_validation_rmse[index] = _nanmean(friction_rmse_values)
            inertia_validation_rmse[index] = _nanmean(inertia_rmse_values)

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
            recommended_for_compensation[index] = bool(
                np.all([artifact.identification.validation.recommended_for_compensation for artifact in motor_artifacts])
                and (
                    not np.isfinite(repeat_consistency_score[index])
                    or float(repeat_consistency_score[index]) <= 0.20
                )
            )

            for artifact in motor_artifacts:
                history[str(motor_id)].append(
                    {
                        "group_index": int(artifact.capture.group_index),
                        "round_index": int(artifact.capture.round_index),
                        "capture_path": str(artifact.capture_path),
                        "identification_path": str(artifact.identification_path),
                        "tau_static": float(artifact.identification.breakaway.tau_static),
                        "tau_bias": float(artifact.identification.breakaway.tau_bias),
                        "tau_c": float(artifact.identification.friction.tau_c),
                        "viscous": float(artifact.identification.friction.viscous),
                        "inertia": float(artifact.identification.inertia.inertia),
                        "friction_rmse": float(artifact.identification.validation.friction_rmse),
                        "inertia_rmse": float(artifact.identification.validation.inertia_rmse),
                        "recommended_for_compensation": bool(
                            artifact.identification.validation.recommended_for_compensation
                        ),
                    }
                )

        return {
            "motor_ids": np.asarray(motor_ids, dtype=np.int64),
            "motor_names": np.asarray(motor_names),
            "round_count": round_count,
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
            "history_json": np.asarray(json.dumps(history, ensure_ascii=False)),
        }

    def _summary_rows(self, payload: dict[str, np.ndarray]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        motor_ids = np.asarray(payload["motor_ids"], dtype=np.int64)
        motor_names = np.asarray(payload["motor_names"]).astype(str)
        for index, motor_id in enumerate(motor_ids.tolist()):
            rows.append(
                {
                    "motor_id": int(motor_id),
                    "motor_name": str(motor_names[index]),
                    "round_count": int(payload["round_count"][index]),
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
                }
            )
        return rows

    def _write_summary_csv(self, path: Path, payload: dict[str, np.ndarray]) -> None:
        rows = self._summary_rows(payload)
        fieldnames = list(rows[0].keys()) if rows else [
            "motor_id",
            "motor_name",
            "round_count",
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
        ]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_summary_report(self, path: Path, payload: dict[str, np.ndarray]) -> None:
        rows = self._summary_rows(payload)
        lines = [
            "# Hardware Identification Summary",
            "",
            "| Motor | tau_static | tau_c | viscous | inertia | friction RMSE | inertia RMSE | repeat consistency | recommended |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{int(row['motor_id']):02d} {row['motor_name']}",
                        f"{float(row['tau_static']):.6f}",
                        f"{float(row['tau_c']):.6f}",
                        f"{float(row['viscous']):.6f}",
                        f"{float(row['inertia']):.6f}",
                        f"{float(row['friction_validation_rmse']):.6f}",
                        f"{float(row['inertia_validation_rmse']):.6f}",
                        f"{float(row['repeat_consistency_score']):.6f}",
                        "yes" if bool(row["recommended_for_compensation"]) else "no",
                    ]
                )
                + " |"
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
