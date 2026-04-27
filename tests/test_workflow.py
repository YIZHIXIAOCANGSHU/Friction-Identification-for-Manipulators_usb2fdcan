from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from friction_identification_core.core import (
    BreakawayIdentificationResult,
    FrictionIdentificationResult,
    InertiaIdentificationResult,
    MotorIdentificationResult,
    RoundCapture,
    ValidationResult,
)
from friction_identification_core.results import ResultStore, RoundArtifact
from friction_identification_core.runtime_config import DEFAULT_CONFIG_PATH, load_config
from friction_identification_core.workflow import run_breakaway, run_compensation, run_identify_all
from friction_identification_core.io import RECV_FRAME_HEAD, RECV_FRAME_STRUCT


class ClosedLoopFakeTransport:
    def __init__(
        self,
        motor_ids: tuple[int, ...],
        *,
        dt: float = 0.005,
        static_threshold: float = 0.12,
        tau_c: float = 0.18,
        tau_bias: float = 0.01,
        viscous: float = 0.04,
        inertia: float = 0.08,
        velocity_gain: float = 1.6,
        trip_motor_id: int | None = None,
        trip_command_threshold: float = 0.05,
        trip_velocity: float | None = None,
        initial_velocity_by_motor: dict[int, float] | None = None,
        torque_limit: float = 2.5,
    ) -> None:
        self._motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
        self._dt = float(dt)
        self._static_threshold = float(static_threshold)
        self._tau_c = float(tau_c)
        self._tau_bias = float(tau_bias)
        self._viscous = float(viscous)
        self._inertia = float(inertia)
        self._velocity_gain = float(velocity_gain)
        self._trip_motor_id = None if trip_motor_id is None else int(trip_motor_id)
        self._trip_command_threshold = float(trip_command_threshold)
        self._trip_velocity = None if trip_velocity is None else float(trip_velocity)
        self._initial_velocity_by_motor = {
            int(motor_id): float(velocity)
            for motor_id, velocity in (initial_velocity_by_motor or {}).items()
        }
        self._torque_limit = float(torque_limit)
        self._pending = bytearray()
        self._state = {
            motor_id: {
                "enabled": False,
                "mode": "mit_torque",
                "position": 0.0,
                "velocity": float(self._initial_velocity_by_motor.get(int(motor_id), 0.0)),
                "torque_feedback": 0.0,
                "torque_cmd": 0.0,
                "velocity_cmd": 0.0,
                "kd": 0.0,
            }
            for motor_id in self._motor_ids
        }
        self.writes: list[tuple[str, int, float]] = []
        self.zero_command_count = 0
        self.disable_count = 0
        self.closed = False

    def _advance_motor(self, motor_id: int) -> tuple[int, float, float, float, float]:
        item = self._state[int(motor_id)]
        velocity = float(item["velocity"])
        position = float(item["position"])
        if not bool(item["enabled"]):
            velocity *= 0.7
            torque_feedback = 0.0
            state = 0
        else:
            state = 1
            if str(item["mode"]) == "mit_torque":
                applied_torque = float(item["torque_cmd"])
            else:
                gain = self._velocity_gain * (1.0 + float(item["kd"]))
                applied_torque = float(np.clip(gain * (float(item["velocity_cmd"]) - velocity), -2.5, 2.5))
            torque_feedback = float(applied_torque)
            if int(motor_id) == int(self._trip_motor_id or -1) and abs(applied_torque) >= self._trip_command_threshold and self._trip_velocity is not None:
                velocity = float(self._trip_velocity)
            else:
                direction = np.sign(velocity) if abs(velocity) > 1.0e-4 else np.sign(applied_torque)
                friction = self._tau_c * direction + self._viscous * velocity + self._tau_bias
                if str(item["mode"]) == "mit_torque" and abs(applied_torque) <= self._static_threshold and abs(velocity) < 0.05:
                    velocity *= 0.8
                else:
                    acceleration = (applied_torque - friction) / self._inertia
                    velocity += self._dt * acceleration
            position += self._dt * velocity
        item["position"] = float(position)
        item["velocity"] = float(velocity)
        item["torque_feedback"] = float(torque_feedback)
        return state, float(position), float(velocity), float(torque_feedback), 30.0 + float(motor_id)

    def _build_cycle_bytes(self) -> bytes:
        frames = bytearray()
        for motor_id in self._motor_ids:
            state, position, velocity, torque_feedback, mos_temperature = self._advance_motor(int(motor_id))
            frames.extend(
                RECV_FRAME_STRUCT.pack(
                    RECV_FRAME_HEAD,
                    int(motor_id),
                    int(state),
                    float(position),
                    float(velocity),
                    float(torque_feedback),
                    float(mos_temperature),
                )
            )
        return bytes(frames)

    def read(self, size: int) -> bytes:
        while len(self._pending) < int(size):
            self._pending.extend(self._build_cycle_bytes())
        chunk = bytes(self._pending[: int(size)])
        del self._pending[: int(size)]
        return chunk

    def send_mit_torque(self, motor_id: int, torque: float) -> bytes:
        item = self._state[int(motor_id)]
        item["mode"] = "mit_torque"
        item["torque_cmd"] = float(torque)
        item["velocity_cmd"] = 0.0
        packet = f"mit_torque:{int(motor_id)}:{float(torque):+.6f}".encode("ascii")
        self.writes.append(("mit_torque", int(motor_id), float(torque)))
        return packet

    def send_mit_velocity(
        self,
        motor_id: int,
        velocity: float,
        kd: float,
        *,
        kp: float = 0.0,
        torque_ff: float = 0.0,
        position: float = 0.0,
    ) -> bytes:
        _ = kp, torque_ff, position
        item = self._state[int(motor_id)]
        item["mode"] = "mit_velocity"
        item["velocity_cmd"] = float(velocity)
        item["kd"] = float(kd)
        packet = f"mit_velocity:{int(motor_id)}:{float(velocity):+.6f}:{float(kd):+.6f}".encode("ascii")
        self.writes.append(("mit_velocity", int(motor_id), float(velocity)))
        return packet

    def send_velocity_mode(self, motor_id: int, velocity: float) -> bytes:
        item = self._state[int(motor_id)]
        item["mode"] = "velocity_mode"
        item["velocity_cmd"] = float(velocity)
        packet = f"velocity_mode:{int(motor_id)}:{float(velocity):+.6f}".encode("ascii")
        self.writes.append(("velocity_mode", int(motor_id), float(velocity)))
        return packet

    def send_zero_command(self, motor_id: int, semantic_mode: str) -> bytes:
        self.zero_command_count += 1
        if semantic_mode == "mit_torque":
            return self.send_mit_torque(int(motor_id), 0.0)
        if semantic_mode == "mit_velocity":
            return self.send_mit_velocity(int(motor_id), 0.0, 0.8)
        return self.send_velocity_mode(int(motor_id), 0.0)

    def enable_motor(self, motor_id: int) -> bytes:
        self._state[int(motor_id)]["enabled"] = True
        self.writes.append(("enable", int(motor_id), 0.0))
        return f"enable:{int(motor_id)}".encode("ascii")

    def disable_motor(self, motor_id: int) -> bytes:
        self.disable_count += 1
        self._state[int(motor_id)]["enabled"] = False
        self.writes.append(("disable", int(motor_id), 0.0))
        return f"disable:{int(motor_id)}".encode("ascii")

    def clear_error(self, motor_id: int) -> bytes:
        self.writes.append(("clear_error", int(motor_id), 0.0))
        return f"clear_error:{int(motor_id)}".encode("ascii")

    def reset_input_buffer(self) -> None:
        self._pending.clear()

    def close(self) -> None:
        self.closed = True

    def motor_type_name(self, motor_id: int) -> str:
        _ = motor_id
        return "FAKE"

    def motor_limits(self, motor_id: int):  # noqa: ANN001
        _ = motor_id
        return None

    def limit_torque_command(self, motor_id: int, torque: float) -> float:
        _ = motor_id
        return float(np.clip(float(torque), -self._torque_limit, self._torque_limit))


class CoastingBreakawayFakeTransport(ClosedLoopFakeTransport):
    def __init__(
        self,
        motor_ids: tuple[int, ...],
        *,
        initial_velocity_by_motor: dict[int, float] | None = None,
    ) -> None:
        super().__init__(
            motor_ids,
            dt=0.005,
            static_threshold=0.12,
            tau_c=0.0,
            tau_bias=0.0,
            viscous=0.0,
            inertia=1.0,
            velocity_gain=3.5,
            initial_velocity_by_motor=initial_velocity_by_motor,
        )

    def _advance_motor(self, motor_id: int) -> tuple[int, float, float, float, float]:
        item = self._state[int(motor_id)]
        velocity = float(item["velocity"])
        position = float(item["position"])
        if not bool(item["enabled"]):
            velocity *= 0.7
            torque_feedback = 0.0
            state = 0
        else:
            state = 1
            mode = str(item["mode"])
            if mode == "mit_torque":
                applied_torque = float(item["torque_cmd"])
                if abs(applied_torque) >= self._static_threshold:
                    velocity = 0.7 * float(np.sign(applied_torque))
                else:
                    # Simulate a lightly damped motor that coasts for too long if we only release torque.
                    velocity *= 0.995
            else:
                target_velocity = float(item["velocity_cmd"])
                velocity += 0.65 * (target_velocity - velocity)
                applied_torque = velocity - target_velocity
            torque_feedback = float(applied_torque)
            position += self._dt * velocity
        item["position"] = float(position)
        item["velocity"] = float(velocity)
        item["torque_feedback"] = float(torque_feedback)
        return state, float(position), float(velocity), float(torque_feedback), 30.0 + float(motor_id)


class MissingFeedbackMotorFakeTransport(ClosedLoopFakeTransport):
    def __init__(self, motor_ids: tuple[int, ...], *, missing_motor_ids: tuple[int, ...]) -> None:
        super().__init__(motor_ids)
        self._missing_motor_ids = {int(motor_id) for motor_id in missing_motor_ids}

    def _build_cycle_bytes(self) -> bytes:
        frames = bytearray()
        for motor_id in self._motor_ids:
            state, position, velocity, torque_feedback, mos_temperature = self._advance_motor(int(motor_id))
            if int(motor_id) in self._missing_motor_ids:
                continue
            frames.extend(
                RECV_FRAME_STRUCT.pack(
                    RECV_FRAME_HEAD,
                    int(motor_id),
                    int(state),
                    float(position),
                    float(velocity),
                    float(torque_feedback),
                    float(mos_temperature),
                )
            )
        return bytes(frames)


class CommandTriggeredFeedbackFakeTransport(ClosedLoopFakeTransport):
    def __init__(self, motor_ids: tuple[int, ...], **kwargs) -> None:  # noqa: ANN003
        super().__init__(motor_ids, **kwargs)
        self._feedback_budget = 0

    def _grant_feedback(self) -> None:
        self._feedback_budget += 1

    def read(self, size: int) -> bytes:
        if self._feedback_budget <= 0:
            return b""
        self._feedback_budget -= 1
        return super().read(size)

    def send_mit_torque(self, motor_id: int, torque: float) -> bytes:
        self._grant_feedback()
        return super().send_mit_torque(motor_id, torque)

    def send_mit_velocity(
        self,
        motor_id: int,
        velocity: float,
        kd: float,
        *,
        kp: float = 0.0,
        torque_ff: float = 0.0,
        position: float = 0.0,
    ) -> bytes:
        self._grant_feedback()
        return super().send_mit_velocity(
            motor_id,
            velocity,
            kd,
            kp=kp,
            torque_ff=torque_ff,
            position=position,
        )

    def send_velocity_mode(self, motor_id: int, velocity: float) -> bytes:
        self._grant_feedback()
        return super().send_velocity_mode(motor_id, velocity)


class StaticBreakawayAssistFakeTransport(ClosedLoopFakeTransport):
    def __init__(
        self,
        motor_ids: tuple[int, ...],
        *,
        external_push_torque: float = 0.25,
    ) -> None:
        super().__init__(
            motor_ids,
            dt=0.005,
            static_threshold=0.5,
            tau_c=0.0,
            tau_bias=0.0,
            viscous=0.0,
            inertia=0.2,
            velocity_gain=1.0,
        )
        self._external_push_torque = float(external_push_torque)

    def _advance_motor(self, motor_id: int) -> tuple[int, float, float, float, float]:
        item = self._state[int(motor_id)]
        velocity = float(item["velocity"])
        position = float(item["position"])
        if not bool(item["enabled"]):
            velocity *= 0.7
            torque_feedback = 0.0
            state = 0
        else:
            state = 1
            applied_torque = float(item["torque_cmd"])
            if abs(velocity) < 0.02 and abs(applied_torque) < self._static_threshold:
                velocity = 0.0
                torque_feedback = float(self._external_push_torque)
            else:
                torque_feedback = float(applied_torque)
                acceleration = (applied_torque - 0.05 * np.sign(applied_torque)) / self._inertia
                velocity += self._dt * acceleration
            position += self._dt * velocity
        item["position"] = float(position)
        item["velocity"] = float(velocity)
        item["torque_feedback"] = float(torque_feedback)
        return state, float(position), float(velocity), float(torque_feedback), 30.0 + float(motor_id)


class TrackingLossFakeTransport(ClosedLoopFakeTransport):
    def __init__(
        self,
        motor_ids: tuple[int, ...],
        *,
        low_speed_scale: float = 0.44,
        holdout_speed_scale: float = 0.70,
    ) -> None:
        super().__init__(
            motor_ids,
            dt=0.005,
            static_threshold=0.12,
            tau_c=0.22,
            tau_bias=0.0,
            viscous=0.02,
            inertia=0.06,
            velocity_gain=2.4,
        )
        self._low_speed_scale = float(low_speed_scale)
        self._holdout_speed_scale = float(holdout_speed_scale)

    def _advance_motor(self, motor_id: int) -> tuple[int, float, float, float, float]:
        item = self._state[int(motor_id)]
        velocity = float(item["velocity"])
        position = float(item["position"])
        if not bool(item["enabled"]):
            velocity *= 0.7
            torque_feedback = 0.0
            state = 0
        else:
            state = 1
            mode = str(item["mode"])
            if mode == "mit_torque":
                applied_torque = float(item["torque_cmd"])
            else:
                commanded_velocity = float(item["velocity_cmd"])
                tracking_scale = 1.0
                if abs(commanded_velocity) <= 0.5 + 1.0e-9:
                    tracking_scale = self._low_speed_scale
                elif abs(commanded_velocity) >= 8.0 - 1.0e-9:
                    tracking_scale = self._holdout_speed_scale
                effective_velocity_cmd = tracking_scale * commanded_velocity
                gain = self._velocity_gain * (1.0 + float(item["kd"]))
                applied_torque = float(np.clip(gain * (effective_velocity_cmd - velocity), -2.5, 2.5))
            torque_feedback = float(applied_torque)
            direction = np.sign(velocity) if abs(velocity) > 1.0e-4 else np.sign(applied_torque)
            friction = self._tau_c * direction + self._viscous * velocity + self._tau_bias
            if mode == "mit_torque" and abs(applied_torque) <= self._static_threshold and abs(velocity) < 0.05:
                velocity *= 0.8
            else:
                acceleration = (applied_torque - friction) / self._inertia
                velocity += self._dt * acceleration
            position += self._dt * velocity
        item["position"] = float(position)
        item["velocity"] = float(velocity)
        item["torque_feedback"] = float(torque_feedback)
        return state, float(position), float(velocity), float(torque_feedback), 30.0 + float(motor_id)


class OscillatingCompensationFakeTransport(ClosedLoopFakeTransport):
    def __init__(self, motor_ids: tuple[int, ...]) -> None:
        super().__init__(
            motor_ids,
            dt=0.005,
            static_threshold=0.5,
            tau_c=0.0,
            tau_bias=0.0,
            viscous=0.0,
            inertia=0.2,
            velocity_gain=1.5,
            torque_limit=2.5,
        )
        self._oscillation_active = False
        self._oscillation_step = 0

    def send_mit_torque(self, motor_id: int, torque: float) -> bytes:
        if abs(float(torque)) > 0.0:
            self._oscillation_active = True
        return super().send_mit_torque(motor_id, torque)

    def _advance_motor(self, motor_id: int) -> tuple[int, float, float, float, float]:
        state, position, velocity, torque_feedback, mos_temperature = super()._advance_motor(motor_id)
        if self._oscillation_active and int(motor_id) == 1 and state == 1:
            self._oscillation_step += 1
            velocity = 7.2 + 0.6 * np.sin(0.9 * float(self._oscillation_step))
            position += self._dt * velocity
            self._state[int(motor_id)]["position"] = float(position)
            self._state[int(motor_id)]["velocity"] = float(velocity)
            self._state[int(motor_id)]["torque_feedback"] = float(torque_feedback)
        return state, float(position), float(velocity), float(torque_feedback), float(mos_temperature)


class CompensationSoftAbortFakeTransport(ClosedLoopFakeTransport):
    def __init__(self, motor_ids: tuple[int, ...], *, trip_velocity: float = 8.7) -> None:
        super().__init__(
            motor_ids,
            trip_motor_id=1,
            trip_command_threshold=0.05,
            trip_velocity=trip_velocity,
            torque_limit=2.5,
        )


class WorkflowTests(unittest.TestCase):
    def _base_config(self):
        return load_config(DEFAULT_CONFIG_PATH)

    def _synthetic_artifact(
        self,
        *,
        motor_id: int,
        group_index: int,
        tau_static: float,
        tau_c: float,
        viscous: float,
        inertia: float,
        friction_rmse: float,
        inertia_rmse: float,
        recommended: bool,
        tmpdir: str,
    ) -> RoundArtifact:
        capture = RoundCapture(
            group_index=int(group_index),
            round_index=int(group_index),
            target_motor_id=int(motor_id),
            motor_name=f"motor_{int(motor_id):02d}",
            time=np.asarray([0.0], dtype=np.float64),
            motor_id=np.asarray([int(motor_id)], dtype=np.int64),
            position=np.asarray([0.0], dtype=np.float64),
            velocity=np.asarray([0.0], dtype=np.float64),
            torque_feedback=np.asarray([0.0], dtype=np.float64),
            command_raw=np.asarray([0.0], dtype=np.float64),
            command=np.asarray([0.0], dtype=np.float64),
            position_cmd=np.asarray([0.0], dtype=np.float64),
            velocity_cmd=np.asarray([0.0], dtype=np.float64),
            acceleration_cmd=np.asarray([0.0], dtype=np.float64),
            phase_name=np.asarray(["synthetic"], dtype=str),
            state=np.asarray([1], dtype=np.uint8),
            mos_temperature=np.asarray([30.0], dtype=np.float64),
            id_match_ok=np.asarray([True], dtype=bool),
            filtered_velocity=np.asarray([0.0], dtype=np.float64),
            estimated_acceleration=np.asarray([0.0], dtype=np.float64),
            friction_term=np.asarray([0.0], dtype=np.float64),
            inertia_term=np.asarray([0.0], dtype=np.float64),
            guard_scale=np.asarray([1.0], dtype=np.float64),
            metadata={"mode": "identify-all"},
        )
        identification = MotorIdentificationResult(
            motor_id=int(motor_id),
            motor_name=f"motor_{int(motor_id):02d}",
            breakaway=BreakawayIdentificationResult(
                torque_positive=float(tau_static),
                torque_negative=-float(tau_static),
                tau_static=float(tau_static),
                tau_bias=0.0,
                metadata={},
            ),
            friction=FrictionIdentificationResult(
                tau_c=float(tau_c),
                viscous=float(viscous),
                tau_bias=0.0,
                train_rmse=float(friction_rmse),
                valid_rmse=float(friction_rmse),
                train_mask=np.asarray([True], dtype=bool),
                valid_mask=np.asarray([True], dtype=bool),
                torque_pred=np.asarray([0.0], dtype=np.float64),
                torque_target=np.asarray([0.0], dtype=np.float64),
                metadata={"status": "accepted" if recommended else "rejected"},
            ),
            inertia=InertiaIdentificationResult(
                inertia=float(inertia),
                train_rmse=float(inertia_rmse),
                valid_rmse=float(inertia_rmse),
                train_mask=np.asarray([True], dtype=bool),
                valid_mask=np.asarray([True], dtype=bool),
                torque_pred=np.asarray([0.0], dtype=np.float64),
                torque_target=np.asarray([0.0], dtype=np.float64),
                filtered_velocity=np.asarray([0.0], dtype=np.float64),
                acceleration=np.asarray([0.0], dtype=np.float64),
                metadata={"status": "accepted" if recommended else "rejected"},
            ),
            validation=ValidationResult(
                friction_rmse=float(friction_rmse),
                inertia_rmse=float(inertia_rmse),
                recommended_for_compensation=bool(recommended),
                detail="accepted" if recommended else "rejected",
                metadata={"status": "accepted" if recommended else "rejected"},
            ),
            metadata={"mode": "identify-all"},
        )
        artifact_dir = Path(tmpdir)
        return RoundArtifact(
            capture=capture,
            identification=identification,
            capture_path=artifact_dir / f"capture_{int(motor_id)}_{int(group_index)}.npz",
            identification_path=artifact_dir / f"identification_{int(motor_id)}_{int(group_index)}.npz",
        )

    def test_identify_all_generates_capture_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.05,
                ),
                safety=replace(
                    base_config.safety,
                    moving_hold_ms=5,
                    post_abort_disable_delay_ms=10,
                ),
                breakaway=replace(
                    base_config.breakaway,
                    torque_step=0.02,
                    hold_duration=0.02,
                    scan_max_torque=np.asarray([0.24, 0.80, 0.60, 0.60, 0.40, 0.40, 0.40], dtype=np.float64),
                ),
                mit_velocity=replace(
                    base_config.mit_velocity,
                    kd_speed=np.asarray([0.8, 1.0, 0.8, 0.8, 0.6, 0.6, 0.6], dtype=np.float64),
                    ramp_acceleration=40.0,
                    steady_hold_duration=0.03,
                    steady_window_ratio=0.5,
                ),
                identification=replace(
                    base_config.identification,
                    steady_speed_points=(0.5, 1.0, 2.0),
                    repeat_count=1,
                    savgol_window=9,
                    savgol_polyorder=2,
                    min_publishable_rounds=1,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = ClosedLoopFakeTransport(motor_ids=base_config.motor_ids)
            result = run_identify_all(config, transport_factory=lambda: transport, show_rerun_viewer=False)
            self.assertTrue(transport.closed)
            self.assertEqual(len(result.artifacts), 1)
            artifact = result.artifacts[0]
            self.assertIsNotNone(result.summary_paths)
            assert result.summary_paths is not None
            self.assertTrue(result.summary_paths.run_summary_path.exists())
            self.assertTrue(result.summary_paths.run_summary_csv_path.exists())
            self.assertTrue(result.summary_paths.run_summary_report_path.exists())
            self.assertEqual(artifact.capture.metadata["mode"], "identify-all")
            phase_names = set(artifact.capture.phase_name.astype(str).tolist())
            self.assertTrue(any(name.startswith("breakaway_") for name in phase_names))
            self.assertTrue(any(name.startswith("speed_hold_") for name in phase_names))
            self.assertTrue(any(name.startswith("inertia_") for name in phase_names))
            self.assertGreater(float(artifact.identification.breakaway.torque_positive), 0.0)
            self.assertLess(float(artifact.identification.breakaway.torque_negative), 0.0)
            self.assertTrue(np.isfinite(float(artifact.identification.friction.tau_c)))
            self.assertTrue(np.isfinite(float(artifact.identification.inertia.inertia)))

            with np.load(result.summary_paths.run_summary_path, allow_pickle=False) as summary:
                self.assertIn("tau_c", summary.files)
                self.assertIn("inertia", summary.files)
                self.assertIn("recommended_for_compensation", summary.files)

            latest_path = Path(tmpdir) / "latest_motor_parameters.json"
            self.assertTrue(latest_path.exists())
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["speed_limit_rad_s"], 10.0)
            self.assertEqual(payload["results_dir"], str(Path(tmpdir)))
            self.assertIn("1", payload["motors"])
            self.assertEqual(payload["motors"]["1"]["motor_id"], 1)
            self.assertEqual(payload["motors"]["1"]["source_run_label"], Path(result.manifest_path).parent.name)

    def test_identify_all_merges_latest_parameters_by_motor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()

            def build_config(enabled_ids: tuple[int, ...]):
                return replace(
                    base_config,
                    motors=replace(base_config.motors, enabled_ids=enabled_ids),
                    transport=replace(
                        base_config.transport,
                        read_timeout=0.001,
                        read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                        sync_timeout=0.05,
                    ),
                    safety=replace(
                        base_config.safety,
                        moving_hold_ms=5,
                        post_abort_disable_delay_ms=10,
                    ),
                    breakaway=replace(
                        base_config.breakaway,
                        torque_step=0.02,
                        hold_duration=0.02,
                        scan_max_torque=np.asarray([0.24, 0.80, 0.60, 0.60, 0.40, 0.40, 0.40], dtype=np.float64),
                    ),
                    mit_velocity=replace(
                        base_config.mit_velocity,
                        kd_speed=np.asarray([0.8, 1.0, 0.8, 0.8, 0.6, 0.6, 0.6], dtype=np.float64),
                        ramp_acceleration=40.0,
                        steady_hold_duration=0.03,
                        steady_window_ratio=0.5,
                    ),
                    identification=replace(
                        base_config.identification,
                        steady_speed_points=(0.5, 1.0, 2.0),
                        repeat_count=1,
                        savgol_window=9,
                        savgol_polyorder=2,
                        min_publishable_rounds=1,
                    ),
                    output=replace(base_config.output, results_dir=Path(tmpdir)),
                )

            result_1 = run_identify_all(
                build_config((1,)),
                transport_factory=lambda: ClosedLoopFakeTransport(motor_ids=base_config.motor_ids),
                show_rerun_viewer=False,
            )
            latest_path = Path(tmpdir) / "latest_motor_parameters.json"
            payload_after_first = json.loads(latest_path.read_text(encoding="utf-8"))
            motor_1_run_label = payload_after_first["motors"]["1"]["source_run_label"]
            self.assertEqual(motor_1_run_label, Path(result_1.manifest_path).parent.name)

            result_2 = run_identify_all(
                build_config((2,)),
                transport_factory=lambda: ClosedLoopFakeTransport(motor_ids=base_config.motor_ids),
                show_rerun_viewer=False,
            )
            payload_after_second = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertIn("1", payload_after_second["motors"])
            self.assertIn("2", payload_after_second["motors"])
            self.assertEqual(payload_after_second["motors"]["1"]["source_run_label"], motor_1_run_label)
            self.assertEqual(
                payload_after_second["motors"]["2"]["source_run_label"],
                Path(result_2.manifest_path).parent.name,
            )

    def test_identify_all_rejects_undertracked_platforms_from_publishable_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(3,)),
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.05,
                ),
                safety=replace(
                    base_config.safety,
                    moving_hold_ms=5,
                    post_abort_disable_delay_ms=10,
                ),
                breakaway=replace(
                    base_config.breakaway,
                    torque_step=0.02,
                    hold_duration=0.02,
                    scan_max_torque=np.asarray([0.24, 0.80, 0.60, 0.60, 0.40, 0.40, 0.40], dtype=np.float64),
                ),
                mit_velocity=replace(
                    base_config.mit_velocity,
                    kd_speed=np.asarray([0.8, 1.0, 0.8, 0.8, 0.6, 0.6, 0.6], dtype=np.float64),
                    ramp_acceleration=40.0,
                    steady_hold_duration=0.03,
                    steady_window_ratio=0.5,
                ),
                identification=replace(
                    base_config.identification,
                    steady_speed_points=(0.5, 1.0, 2.0, 4.0, 8.0),
                    repeat_count=1,
                    savgol_window=9,
                    savgol_polyorder=2,
                    min_publishable_rounds=1,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            result = run_identify_all(
                config,
                transport_factory=lambda: TrackingLossFakeTransport(motor_ids=base_config.motor_ids),
                show_rerun_viewer=False,
            )

            identification = result.artifacts[0].identification
            self.assertFalse(identification.validation.recommended_for_compensation)
            self.assertEqual(identification.validation.metadata["status"], "rejected")
            self.assertGreaterEqual(int(identification.friction.metadata["rejected_platform_count"]), 2)
            self.assertEqual(int(identification.friction.metadata["accepted_valid_platform_count"]), 0)

            latest_path = Path(tmpdir) / "latest_motor_parameters.json"
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertNotIn("3", payload["motors"])

    def test_save_latest_parameters_uses_only_accepted_rounds_and_preserves_previous_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1, 2)),
                identification=replace(
                    base_config.identification,
                    repeat_count=3,
                    min_publishable_rounds=2,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            latest_path = Path(tmpdir) / "latest_motor_parameters.json"
            latest_path.write_text(
                json.dumps(
                    {
                        "updated_at": "2026-04-24T00:00:00+00:00",
                        "results_dir": str(Path(tmpdir)),
                        "speed_limit_rad_s": 10.0,
                        "motors": {
                            "2": {
                                "motor_id": 2,
                                "motor_name": "motor_02",
                                "identified_at": "2026-04-24T00:00:00+00:00",
                                "source_run_label": "seed_run",
                                "tau_static": 0.5,
                                "tau_bias": 0.0,
                                "tau_c": 0.21,
                                "viscous": 0.03,
                                "inertia": 0.02,
                                "friction_validation_rmse": 0.03,
                                "inertia_validation_rmse": 0.04,
                                "repeat_consistency_score": 0.02,
                                "recommended_for_compensation": True,
                                "model_kind": "static_v1",
                                "publish_status": "published",
                                "publish_detail": "seed model",
                                "accepted_round_count": 2,
                                "selected_rounds": [1, 2],
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            artifacts = [
                self._synthetic_artifact(
                    motor_id=1,
                    group_index=1,
                    tau_static=0.20,
                    tau_c=0.10,
                    viscous=0.01,
                    inertia=0.02,
                    friction_rmse=0.02,
                    inertia_rmse=0.03,
                    recommended=True,
                    tmpdir=tmpdir,
                ),
                self._synthetic_artifact(
                    motor_id=1,
                    group_index=2,
                    tau_static=0.24,
                    tau_c=0.14,
                    viscous=0.02,
                    inertia=0.03,
                    friction_rmse=0.02,
                    inertia_rmse=0.03,
                    recommended=True,
                    tmpdir=tmpdir,
                ),
                self._synthetic_artifact(
                    motor_id=1,
                    group_index=3,
                    tau_static=0.90,
                    tau_c=0.60,
                    viscous=0.20,
                    inertia=0.30,
                    friction_rmse=0.40,
                    inertia_rmse=0.35,
                    recommended=False,
                    tmpdir=tmpdir,
                ),
                self._synthetic_artifact(
                    motor_id=2,
                    group_index=1,
                    tau_static=0.32,
                    tau_c=0.18,
                    viscous=0.02,
                    inertia=0.01,
                    friction_rmse=0.02,
                    inertia_rmse=0.03,
                    recommended=True,
                    tmpdir=tmpdir,
                ),
                self._synthetic_artifact(
                    motor_id=2,
                    group_index=2,
                    tau_static=0.70,
                    tau_c=0.45,
                    viscous=0.12,
                    inertia=0.18,
                    friction_rmse=0.35,
                    inertia_rmse=0.28,
                    recommended=False,
                    tmpdir=tmpdir,
                ),
            ]
            store = ResultStore(config, mode="identify-all")
            store.save_summary(artifacts)
            store.save_latest_parameters(artifacts)

            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            motor_1 = payload["motors"]["1"]
            self.assertAlmostEqual(float(motor_1["tau_static"]), 0.22, places=6)
            self.assertAlmostEqual(float(motor_1["tau_c"]), 0.12, places=6)
            self.assertAlmostEqual(float(motor_1["viscous"]), 0.015, places=6)
            self.assertAlmostEqual(float(motor_1["inertia"]), 0.025, places=6)
            self.assertEqual(motor_1["publish_status"], "published")
            self.assertEqual(motor_1["accepted_round_count"], 2)
            self.assertEqual(motor_1["selected_rounds"], [1, 2])

            motor_2 = payload["motors"]["2"]
            self.assertEqual(motor_2["source_run_label"], "seed_run")
            self.assertAlmostEqual(float(motor_2["tau_c"]), 0.21, places=6)

    def test_compensation_requires_latest_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            with self.assertRaisesRegex(ValueError, "latest motor parameters"):
                run_compensation(config, transport_factory=lambda: ClosedLoopFakeTransport(motor_ids=base_config.motor_ids), show_rerun_viewer=False, max_runtime_s=0.01)

    def test_compensation_requires_published_model_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            latest_path = Path(tmpdir) / "latest_motor_parameters.json"
            latest_path.write_text(
                json.dumps(
                    {
                        "updated_at": "2026-04-25T00:00:00+00:00",
                        "results_dir": str(Path(tmpdir)),
                        "speed_limit_rad_s": 10.0,
                        "motors": {
                            "1": {
                                "motor_id": 1,
                                "motor_name": "motor_01",
                                "identified_at": "2026-04-25T00:00:00+00:00",
                                "source_run_label": "seed_run",
                                "tau_static": 0.12,
                                "tau_bias": 0.01,
                                "tau_c": 0.18,
                                "viscous": 0.04,
                                "inertia": 0.08,
                                "friction_validation_rmse": 0.25,
                                "inertia_validation_rmse": 0.30,
                                "repeat_consistency_score": 0.40,
                                "recommended_for_compensation": False,
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            with self.assertRaisesRegex(ValueError, "published model"):
                run_compensation(
                    config,
                    transport_factory=lambda: ClosedLoopFakeTransport(motor_ids=base_config.motor_ids),
                    show_rerun_viewer=False,
                    max_runtime_s=0.01,
                )

    def test_compensation_uses_latest_parameters_and_saves_capture_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            latest_path = Path(tmpdir) / "latest_motor_parameters.json"
            latest_path.write_text(
                json.dumps(
                    {
                        "updated_at": "2026-04-25T00:00:00+00:00",
                        "results_dir": str(Path(tmpdir)),
                        "speed_limit_rad_s": 10.0,
                        "motors": {
                            "1": {
                                "motor_id": 1,
                                "motor_name": "motor_01",
                                "identified_at": "2026-04-25T00:00:00+00:00",
                                "source_run_label": "seed_run",
                                "tau_static": 0.12,
                                "tau_bias": 0.01,
                                "tau_c": 0.18,
                                "viscous": 0.04,
                                "inertia": 0.08,
                                "friction_validation_rmse": 0.01,
                                "inertia_validation_rmse": 0.02,
                                "repeat_consistency_score": 0.03,
                                "recommended_for_compensation": False,
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.05,
                ),
                safety=replace(
                    base_config.safety,
                    moving_hold_ms=5,
                    post_abort_disable_delay_ms=10,
                ),
                identification=replace(
                    base_config.identification,
                    savgol_window=9,
                    savgol_polyorder=2,
                ),
                compensation=replace(base_config.compensation, require_published_model=False),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = ClosedLoopFakeTransport(
                motor_ids=base_config.motor_ids,
                initial_velocity_by_motor={1: 1.0},
            )

            result = run_compensation(
                config,
                transport_factory=lambda: transport,
                show_rerun_viewer=False,
                max_runtime_s=0.05,
            )

            self.assertTrue(transport.closed)
            self.assertEqual(result.summary_paths, None)
            capture_files = sorted(Path(tmpdir).glob("runs/*_compensation/group_01/motor_01/capture.npz"))
            self.assertEqual(len(capture_files), 1)
            identification_files = list(Path(tmpdir).glob("runs/*_compensation/group_01/motor_01/identification.npz"))
            self.assertEqual(identification_files, [])
            self.assertTrue(any(kind == "mit_torque" and abs(value) > 0.0 for kind, _, value in transport.writes))

    def test_compensation_sends_heartbeat_commands_when_feedback_requires_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            latest_path = Path(tmpdir) / "latest_motor_parameters.json"
            latest_path.write_text(
                json.dumps(
                    {
                        "updated_at": "2026-04-25T00:00:00+00:00",
                        "results_dir": str(Path(tmpdir)),
                        "speed_limit_rad_s": 10.0,
                        "motors": {
                            "1": {
                                "motor_id": 1,
                                "motor_name": "motor_01",
                                "identified_at": "2026-04-25T00:00:00+00:00",
                                "source_run_label": "seed_run",
                                "tau_static": 0.12,
                                "tau_bias": 0.01,
                                "tau_c": 0.18,
                                "viscous": 0.04,
                                "inertia": 0.08,
                                "friction_validation_rmse": 0.01,
                                "inertia_validation_rmse": 0.02,
                                "repeat_consistency_score": 0.03,
                                "recommended_for_compensation": False,
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.05,
                ),
                safety=replace(
                    base_config.safety,
                    moving_hold_ms=5,
                    post_abort_disable_delay_ms=10,
                ),
                identification=replace(
                    base_config.identification,
                    savgol_window=9,
                    savgol_polyorder=2,
                ),
                compensation=replace(base_config.compensation, require_published_model=False),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = CommandTriggeredFeedbackFakeTransport(
                motor_ids=base_config.motor_ids,
                initial_velocity_by_motor={1: 1.0},
            )

            result = run_compensation(
                config,
                transport_factory=lambda: transport,
                show_rerun_viewer=False,
                max_runtime_s=0.05,
            )

            with np.load(result.artifacts[0], allow_pickle=False) as capture:
                self.assertGreater(int(capture["time"].size), 0)
                self.assertTrue(np.any(np.abs(capture["command"]) > 0.0))

    def test_compensation_uses_tau_static_assist_from_feedback_torque_near_zero_speed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            latest_path = Path(tmpdir) / "latest_motor_parameters.json"
            latest_path.write_text(
                json.dumps(
                    {
                        "updated_at": "2026-04-25T00:00:00+00:00",
                        "results_dir": str(Path(tmpdir)),
                        "speed_limit_rad_s": 10.0,
                        "motors": {
                            "1": {
                                "motor_id": 1,
                                "motor_name": "motor_01",
                                "identified_at": "2026-04-25T00:00:00+00:00",
                                "source_run_label": "seed_run",
                                "tau_static": 0.60,
                                "tau_bias": 0.0,
                                "tau_c": 0.12,
                                "viscous": 0.0,
                                "inertia": 0.0,
                                "friction_validation_rmse": 0.01,
                                "inertia_validation_rmse": 0.02,
                                "repeat_consistency_score": 0.03,
                                "recommended_for_compensation": True,
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.05,
                ),
                safety=replace(
                    base_config.safety,
                    moving_hold_ms=5,
                    post_abort_disable_delay_ms=10,
                    moving_velocity_threshold=0.2,
                ),
                identification=replace(
                    base_config.identification,
                    savgol_window=9,
                    savgol_polyorder=2,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = StaticBreakawayAssistFakeTransport(motor_ids=base_config.motor_ids)

            result = run_compensation(
                config,
                transport_factory=lambda: transport,
                show_rerun_viewer=False,
                max_runtime_s=0.05,
            )

            with np.load(result.artifacts[0], allow_pickle=False) as capture:
                self.assertGreater(float(np.max(capture["command"])), 0.15)
                self.assertLessEqual(float(np.max(capture["command"])), 0.18 + 1.0e-6)
                self.assertGreater(float(np.max(capture["friction_term"])), 0.15)

    def test_compensation_limits_static_friction_inertia_and_total_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            latest_path = Path(tmpdir) / "latest_motor_parameters.json"
            latest_path.write_text(
                json.dumps(
                    {
                        "updated_at": "2026-04-25T00:00:00+00:00",
                        "results_dir": str(Path(tmpdir)),
                        "speed_limit_rad_s": 10.0,
                        "motors": {
                            "1": {
                                "motor_id": 1,
                                "motor_name": "motor_01",
                                "identified_at": "2026-04-25T00:00:00+00:00",
                                "source_run_label": "seed_run",
                                "tau_static": 2.0,
                                "tau_bias": 0.0,
                                "tau_c": 0.2,
                                "viscous": 0.0,
                                "inertia": 1.5,
                                "friction_validation_rmse": 0.01,
                                "inertia_validation_rmse": 0.02,
                                "repeat_consistency_score": 0.03,
                                "recommended_for_compensation": True,
                                "publish_status": "published",
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.05,
                ),
                safety=replace(
                    base_config.safety,
                    moving_hold_ms=5,
                    post_abort_disable_delay_ms=10,
                ),
                identification=replace(
                    base_config.identification,
                    savgol_window=9,
                    savgol_polyorder=2,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = OscillatingCompensationFakeTransport(motor_ids=base_config.motor_ids)

            result = run_compensation(
                config,
                transport_factory=lambda: transport,
                show_rerun_viewer=False,
                max_runtime_s=0.05,
            )

            with np.load(result.artifacts[0], allow_pickle=False) as capture:
                self.assertIn("filtered_velocity", capture.files)
                self.assertIn("estimated_acceleration", capture.files)
                self.assertIn("friction_term", capture.files)
                self.assertIn("inertia_term", capture.files)
                self.assertIn("guard_scale", capture.files)
                self.assertLessEqual(float(np.max(np.abs(capture["friction_term"]))), 0.300001)
                self.assertLessEqual(float(np.max(np.abs(capture["inertia_term"]))), 0.218751)
                self.assertLessEqual(float(np.max(np.abs(capture["command"]))), 0.875001)
                self.assertTrue(np.all(np.isfinite(capture["guard_scale"])))
                self.assertTrue(np.all(capture["guard_scale"] <= 1.0))

    def test_compensation_soft_abort_triggers_before_hard_speed_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            latest_path = Path(tmpdir) / "latest_motor_parameters.json"
            latest_path.write_text(
                json.dumps(
                    {
                        "updated_at": "2026-04-25T00:00:00+00:00",
                        "results_dir": str(Path(tmpdir)),
                        "speed_limit_rad_s": 10.0,
                        "motors": {
                            "1": {
                                "motor_id": 1,
                                "motor_name": "motor_01",
                                "identified_at": "2026-04-25T00:00:00+00:00",
                                "source_run_label": "seed_run",
                                "tau_static": 0.60,
                                "tau_bias": 0.0,
                                "tau_c": 0.18,
                                "viscous": 0.0,
                                "inertia": 0.0,
                                "friction_validation_rmse": 0.01,
                                "inertia_validation_rmse": 0.02,
                                "repeat_consistency_score": 0.03,
                                "recommended_for_compensation": True,
                                "publish_status": "published",
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.05,
                ),
                safety=replace(
                    base_config.safety,
                    moving_hold_ms=5,
                    post_abort_disable_delay_ms=10,
                ),
                identification=replace(
                    base_config.identification,
                    savgol_window=9,
                    savgol_polyorder=2,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = CompensationSoftAbortFakeTransport(motor_ids=base_config.motor_ids)

            with self.assertRaisesRegex(RuntimeError, r"reason=soft_speed_abort"):
                run_compensation(
                    config,
                    transport_factory=lambda: transport,
                    show_rerun_viewer=False,
                    max_runtime_s=0.05,
                )

    def test_breakaway_hard_abort_sends_zero_then_disable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.05,
                ),
                safety=replace(
                    base_config.safety,
                    moving_hold_ms=5,
                    post_abort_disable_delay_ms=10,
                ),
                breakaway=replace(
                    base_config.breakaway,
                    torque_step=0.02,
                    hold_duration=0.02,
                    scan_max_torque=np.asarray([0.24, 0.80, 0.60, 0.60, 0.40, 0.40, 0.40], dtype=np.float64),
                ),
                identification=replace(base_config.identification, repeat_count=1),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = ClosedLoopFakeTransport(
                motor_ids=base_config.motor_ids,
                trip_motor_id=1,
                trip_command_threshold=0.02,
                trip_velocity=12.0,
            )
            with self.assertRaises(RuntimeError):
                run_breakaway(config, transport_factory=lambda: transport, show_rerun_viewer=False)

        self.assertTrue(transport.closed)
        self.assertGreaterEqual(transport.zero_command_count, 5)
        self.assertGreaterEqual(transport.disable_count, 1)

    def test_breakaway_uses_active_velocity_settle_between_directions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.04,
                ),
                safety=replace(
                    base_config.safety,
                    moving_hold_ms=5,
                    post_abort_disable_delay_ms=10,
                ),
                breakaway=replace(
                    base_config.breakaway,
                    torque_step=0.04,
                    hold_duration=0.01,
                    scan_max_torque=np.asarray([0.20, 0.80, 0.60, 0.60, 0.40, 0.40, 0.40], dtype=np.float64),
                ),
                identification=replace(base_config.identification, repeat_count=1),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = CoastingBreakawayFakeTransport(motor_ids=base_config.motor_ids)

            result = run_breakaway(config, transport_factory=lambda: transport, show_rerun_viewer=False)

            self.assertTrue(transport.closed)
            self.assertEqual(len(result.artifacts), 1)
            self.assertTrue(any(kind == "mit_velocity" for kind, _, _ in transport.writes))

    def test_precheck_uses_active_velocity_settle_for_coasting_motor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.04,
                ),
                safety=replace(
                    base_config.safety,
                    moving_hold_ms=5,
                    post_abort_disable_delay_ms=10,
                ),
                breakaway=replace(
                    base_config.breakaway,
                    torque_step=0.04,
                    hold_duration=0.01,
                    scan_max_torque=np.asarray([0.20, 0.80, 0.60, 0.60, 0.40, 0.40, 0.40], dtype=np.float64),
                ),
                identification=replace(base_config.identification, repeat_count=1),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = CoastingBreakawayFakeTransport(
                motor_ids=base_config.motor_ids,
                initial_velocity_by_motor={1: 1.0},
            )

            result = run_breakaway(config, transport_factory=lambda: transport, show_rerun_viewer=False)

            self.assertTrue(transport.closed)
            self.assertEqual(len(result.artifacts), 1)
            first_enable_index = next(index for index, (kind, _, _) in enumerate(transport.writes) if kind == "enable")
            self.assertEqual(transport.writes[first_enable_index + 1][0], "mit_velocity")

    def test_precheck_reports_feedback_timeout_when_target_motor_is_silent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(6,)),
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.04,
                ),
                safety=replace(
                    base_config.safety,
                    moving_hold_ms=5,
                    post_abort_disable_delay_ms=10,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = MissingFeedbackMotorFakeTransport(
                motor_ids=base_config.motor_ids,
                missing_motor_ids=(6,),
            )

            with self.assertRaisesRegex(RuntimeError, r"reason=feedback_timeout"):
                run_breakaway(config, transport_factory=lambda: transport, show_rerun_viewer=False)


if __name__ == "__main__":
    unittest.main()
