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
    PIECEWISE_STATIC_LINEAR_KIND,
    RoundCapture,
    ValidationResult,
)
from friction_identification_core.capture import CaptureBuffer
from friction_identification_core.results import ResultStore, RoundArtifact
from friction_identification_core.runtime_config import DEFAULT_CONFIG_PATH, load_config
from friction_identification_core.workflow import _identify_round, run_breakaway, run_compensation, run_dynamic_mit, run_identify_all
from friction_identification_core.phases.breakaway import breakaway_torque_scan_values
from friction_identification_core.phases.dynamic_mit import build_dynamic_mit_trajectory
from friction_identification_core.phases.low_speed import _run_low_speed_micro_motion, run_low_speed_characterization_phase
from friction_identification_core.io import FeedbackFrameParser, RECV_FRAME_HEAD, RECV_FRAME_STRUCT
from friction_identification_core.safety import RuntimeAbortError
from friction_identification_core.visualization import RerunRecorder


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
        initial_position_by_motor: dict[int, float] | None = None,
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
        self._initial_position_by_motor = {
            int(motor_id): float(position)
            for motor_id, position in (initial_position_by_motor or {}).items()
        }
        self._torque_limit = float(torque_limit)
        self._pending = bytearray()
        self._state = {
            motor_id: {
                "enabled": False,
                "mode": "mit_torque",
                "position": float(self._initial_position_by_motor.get(int(motor_id), 0.0)),
                "velocity": float(self._initial_velocity_by_motor.get(int(motor_id), 0.0)),
                "torque_feedback": 0.0,
                "torque_cmd": 0.0,
                "velocity_cmd": 0.0,
                "kd": 0.0,
            }
            for motor_id in self._motor_ids
        }
        self.writes: list[tuple[str, int, float]] = []
        self.mit_velocity_commands: list[dict[str, float]] = []
        self.mit_state_commands: list[dict[str, float]] = []
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
        item = self._state[int(motor_id)]
        item["mode"] = "mit_velocity"
        item["velocity_cmd"] = float(velocity)
        item["kd"] = float(kd)
        packet = f"mit_velocity:{int(motor_id)}:{float(velocity):+.6f}:{float(kd):+.6f}".encode("ascii")
        self.writes.append(("mit_velocity", int(motor_id), float(velocity)))
        self.mit_velocity_commands.append(
            {
                "motor_id": float(motor_id),
                "velocity": float(velocity),
                "kd": float(kd),
                "kp": float(kp),
                "torque_ff": float(torque_ff),
                "position": float(position),
            }
        )
        return packet

    def send_mit_state(
        self,
        motor_id: int,
        position: float,
        velocity: float,
        kp: float,
        kd: float,
        torque_ff: float = 0.0,
    ) -> bytes:
        item = self._state[int(motor_id)]
        item["mode"] = "mit_state"
        item["position_cmd"] = float(position)
        item["velocity_cmd"] = float(velocity)
        item["kp"] = float(kp)
        item["kd"] = float(kd)
        item["torque_cmd"] = float(torque_ff)
        packet = (
            f"mit_state:{int(motor_id)}:{float(position):+.6f}:"
            f"{float(velocity):+.6f}:{float(kp):+.6f}:{float(kd):+.6f}:{float(torque_ff):+.6f}"
        ).encode("ascii")
        self.writes.append(("mit_state", int(motor_id), float(velocity)))
        self.mit_state_commands.append(
            {
                "motor_id": float(motor_id),
                "position": float(position),
                "velocity": float(velocity),
                "kp": float(kp),
                "kd": float(kd),
                "torque_ff": float(torque_ff),
            }
        )
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
        if semantic_mode == "mit_state":
            return self.send_mit_state(int(motor_id), 0.0, 0.0, 0.0, 0.8)
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

    def send_mit_state(
        self,
        motor_id: int,
        position: float,
        velocity: float,
        kp: float,
        kd: float,
        torque_ff: float = 0.0,
    ) -> bytes:
        self._grant_feedback()
        return super().send_mit_state(
            motor_id,
            position,
            velocity,
            kp,
            kd,
            torque_ff=torque_ff,
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
            stiction_evidence=np.asarray([False], dtype=bool),
            metadata={"mode": "identify-all"},
        )
        friction_model = {
            "kind": PIECEWISE_STATIC_LINEAR_KIND,
            "parameters": {
                "tau_static": float(tau_static),
                "tau_c": float(tau_c),
                "viscous": float(viscous),
                "tau_bias": 0.0,
                "inertia": float(inertia),
                "static_velocity_threshold_rad_s": 0.20,
                "static_transition_velocity_rad_s": 0.50,
                "breakaway_positive": float(tau_static),
                "breakaway_negative": -float(tau_static),
            },
            "train_rmse": float(friction_rmse),
            "valid_rmse": float(friction_rmse),
        }
        export_models = {
            "embedded_piecewise_linear_friction": {
                "kind": PIECEWISE_STATIC_LINEAR_KIND,
                "tau_static": float(tau_static),
                "tau_c": float(tau_c),
                "viscous": float(viscous),
                "tau_bias": 0.0,
                "inertia": float(inertia),
                "static_velocity_threshold_rad_s": 0.20,
                "static_transition_velocity_rad_s": 0.50,
            }
        }
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
        identification = MotorIdentificationResult(
            motor_id=identification.motor_id,
            motor_name=identification.motor_name,
            breakaway=identification.breakaway,
            friction=identification.friction,
            inertia=identification.inertia,
            validation=identification.validation,
            metadata={
                "mode": "identify-all",
                "model_kind": PIECEWISE_STATIC_LINEAR_KIND,
                "friction_model": friction_model,
                "export_models": export_models,
            },
        )
        artifact_dir = Path(tmpdir)
        return RoundArtifact(
            capture=capture,
            identification=identification,
            capture_path=artifact_dir / f"capture_{int(motor_id)}_{int(group_index)}.npz",
            identification_path=artifact_dir / f"identification_{int(motor_id)}_{int(group_index)}.npz",
        )

    def _synthetic_inertia_capture(self) -> RoundCapture:
        tau_c = 0.30
        viscous = 0.02
        tau_bias = 0.01
        inertia = 0.04

        platform_values = np.asarray([1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0], dtype=np.float64)
        platform_phase_names = [
            "speed_hold_train_+1.00",
            "speed_hold_train_-1.00",
            "speed_hold_train_+2.00",
            "speed_hold_train_-2.00",
            "speed_hold_train_+4.00",
            "speed_hold_train_-4.00",
            "speed_hold_valid_+6.00",
            "speed_hold_valid_-6.00",
        ]
        speed_values = np.repeat(platform_values, 8)
        speed_phase_names = [phase_name for phase_name in platform_phase_names for _ in range(8)]
        speed_time = np.linspace(0.0, 1.5, speed_values.size, dtype=np.float64)
        speed_torque = tau_c * np.sign(speed_values) + viscous * speed_values + tau_bias

        inertia_time = np.linspace(1.6, 9.6, 800, dtype=np.float64)
        local_time = inertia_time - float(inertia_time[0])
        clean_velocity = (
            2.5 * np.sin(2.0 * np.pi * 0.35 * local_time)
            + 0.8 * np.sin(2.0 * np.pi * 0.90 * local_time)
        )
        measured_velocity = clean_velocity + (
            0.12 * np.sin(2.0 * np.pi * 13.0 * local_time)
            + 0.04 * np.sin(2.0 * np.pi * 27.0 * local_time)
        )
        clean_acceleration = np.gradient(clean_velocity, inertia_time, edge_order=1)
        inertia_torque = (
            tau_c * np.sign(clean_velocity)
            + viscous * clean_velocity
            + tau_bias
            + inertia * clean_acceleration
        )
        inertia_phase_names = np.where(local_time < 5.0, "inertia_train_01", "inertia_valid_01")

        time = np.concatenate([speed_time, inertia_time])
        velocity = np.concatenate([speed_values, measured_velocity])
        torque = np.concatenate([speed_torque, inertia_torque])
        phase_name = np.concatenate([np.asarray(speed_phase_names, dtype=str), inertia_phase_names.astype(str)])

        return RoundCapture(
            group_index=1,
            round_index=1,
            target_motor_id=2,
            motor_name="motor_02",
            time=time,
            motor_id=np.full(time.size, 2, dtype=np.int64),
            position=np.zeros(time.size, dtype=np.float64),
            velocity=velocity,
            torque_feedback=torque,
            command_raw=np.zeros(time.size, dtype=np.float64),
            command=np.zeros(time.size, dtype=np.float64),
            position_cmd=np.zeros(time.size, dtype=np.float64),
            velocity_cmd=velocity,
            acceleration_cmd=np.zeros(time.size, dtype=np.float64),
            phase_name=phase_name,
            state=np.ones(time.size, dtype=np.uint8),
            mos_temperature=np.full(time.size, 30.0, dtype=np.float64),
            id_match_ok=np.ones(time.size, dtype=bool),
            filtered_velocity=np.zeros(time.size, dtype=np.float64),
            estimated_acceleration=np.zeros(time.size, dtype=np.float64),
            friction_term=np.zeros(time.size, dtype=np.float64),
            inertia_term=np.zeros(time.size, dtype=np.float64),
            guard_scale=np.ones(time.size, dtype=np.float64),
            stiction_evidence=np.zeros(time.size, dtype=bool),
            metadata={"mode": "identify-all"},
        )

    def _synthetic_breakaway(
        self,
        *,
        positive_limit: bool = False,
        negative_limit: bool = False,
    ) -> BreakawayIdentificationResult:
        scan_limit = 0.50
        return BreakawayIdentificationResult(
            torque_positive=scan_limit if positive_limit else 0.42,
            torque_negative=-scan_limit if negative_limit else -0.41,
            tau_static=scan_limit if positive_limit and negative_limit else 0.415,
            tau_bias=0.0,
            metadata={
                "scan_max_torque": scan_limit,
                "torque_step": 0.01,
                "hold_duration": 0.25,
                "positive_scan_limit_reached": bool(positive_limit),
                "negative_scan_limit_reached": bool(negative_limit),
                "both_scan_limits_reached": bool(positive_limit and negative_limit),
            },
        )

    def test_low_speed_characterization_records_stiction_without_exceeding_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            config = replace(
                base_config,
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.05,
                ),
                low_speed=replace(
                    base_config.low_speed,
                    speed_points=(0.05, 0.10),
                    ramp_acceleration=8.0,
                    hold_duration=0.02,
                    micro_motion_record_duration=0.03,
                    micro_motion_velocity_limit=0.20,
                    micro_motion_frequency_hz=1.0,
                ),
            )
            transport = TrackingLossFakeTransport(motor_ids=base_config.motor_ids, low_speed_scale=0.05)
            transport.enable_motor(1)
            parser = FeedbackFrameParser(max_motor_id=max(base_config.motor_ids))
            recorder = RerunRecorder(
                Path(tmpdir) / "low_speed.rrd",
                motor_ids=base_config.motor_ids,
                motor_names={motor_id: base_config.motors.name_for(motor_id) for motor_id in base_config.motor_ids},
                mode="identify-all",
                show_viewer=False,
            )
            capture_buffer = CaptureBuffer(target_motor_id=1, motor_name="motor_01")

            try:
                run_low_speed_characterization_phase(
                    config=config,
                    transport=transport,
                    parser=parser,
                    rerun_recorder=recorder,
                    capture_buffer=capture_buffer,
                    target_motor_id=1,
                    group_index=1,
                    round_index=1,
                )
            finally:
                recorder.close()
                transport.close()

            capture = capture_buffer.build(group_index=1, round_index=1, metadata={"mode": "identify-all"})
            phase_names = capture.phase_name.astype(str)
            self.assertTrue(any(name.startswith("low_speed_hold_") for name in phase_names))
            self.assertTrue(any(name.startswith("low_speed_micro_") for name in phase_names))
            self.assertTrue(np.any(capture.stiction_evidence))
            commanded_velocities = [value for kind, _, value in transport.writes if kind in {"mit_velocity", "mit_state"}]
            self.assertLessEqual(max(abs(float(item)) for item in commanded_velocities), 0.200001)

    def test_low_speed_micro_motion_sends_mit_velocity_targets_without_torque_ff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            config = replace(
                base_config,
                transport=replace(
                    base_config.transport,
                    read_timeout=0.001,
                    read_chunk_size=RECV_FRAME_STRUCT.size * len(base_config.motor_ids),
                    sync_timeout=0.05,
                ),
                low_speed=replace(
                    base_config.low_speed,
                    speed_points=(0.05,),
                    ramp_acceleration=8.0,
                    hold_duration=0.01,
                    micro_motion_record_duration=0.04,
                    micro_motion_velocity_limit=0.12,
                    micro_motion_frequency_hz=0.5,
                    micro_motion_kp=1.5,
                    micro_motion_kd=0.25,
                ),
                dynamic_mit=replace(
                    base_config.dynamic_mit,
                    kp=9.0,
                    kd=1.5,
                ),
            )
            transport = ClosedLoopFakeTransport(motor_ids=base_config.motor_ids)
            transport.enable_motor(1)
            parser = FeedbackFrameParser(max_motor_id=max(base_config.motor_ids))
            recorder = RerunRecorder(
                Path(tmpdir) / "low_speed_gains.rrd",
                motor_ids=base_config.motor_ids,
                motor_names={motor_id: base_config.motors.name_for(motor_id) for motor_id in base_config.motor_ids},
                mode="identify-all",
                show_viewer=False,
            )
            capture_buffer = CaptureBuffer(target_motor_id=1, motor_name="motor_01")

            try:
                _run_low_speed_micro_motion(
                    config=config,
                    transport=transport,
                    parser=parser,
                    rerun_recorder=recorder,
                    capture_buffer=capture_buffer,
                    target_motor_id=1,
                    group_index=1,
                    round_index=1,
                )
            finally:
                recorder.close()
                transport.close()

            self.assertFalse(transport.mit_state_commands)
            self.assertTrue(transport.mit_velocity_commands)
            self.assertTrue(
                all(
                    abs(float(command["velocity"])) <= float(config.low_speed.micro_motion_velocity_limit) + 1.0e-9
                    for command in transport.mit_velocity_commands
                )
            )
            self.assertTrue(all(np.isclose(command["kp"], 0.0) for command in transport.mit_velocity_commands))
            self.assertTrue(
                all(np.isclose(command["kd"], config.low_speed.micro_motion_kd) for command in transport.mit_velocity_commands)
            )
            self.assertTrue(all(np.isclose(command["torque_ff"], 0.0) for command in transport.mit_velocity_commands))
            self.assertTrue(all(np.isclose(command["position"], 0.0) for command in transport.mit_velocity_commands))

            capture = capture_buffer.build(group_index=1, round_index=1, metadata={"mode": "identify-all"})
            micro_mask = np.asarray(
                [name.startswith("low_speed_micro_") for name in capture.phase_name.astype(str)],
                dtype=bool,
            )
            self.assertTrue(np.any(micro_mask))
            self.assertTrue(np.allclose(capture.command[micro_mask], capture.velocity_cmd[micro_mask]))
            self.assertTrue(np.allclose(capture.position_cmd[micro_mask], 0.0))
            self.assertTrue(np.allclose(capture.kp_cmd[micro_mask], 0.0))
            self.assertTrue(np.allclose(capture.kd_cmd[micro_mask], config.low_speed.micro_motion_kd))
            self.assertTrue(np.allclose(capture.torque_ff_cmd[micro_mask], 0.0))

    def test_breakaway_scan_values_never_exceed_scan_limit(self) -> None:
        values = breakaway_torque_scan_values(torque_step=0.06, scan_limit=0.20)

        self.assertEqual(values.tolist(), [0.06, 0.12, 0.18, 0.20])
        self.assertLessEqual(float(np.max(values)), 0.20)

    def test_dynamic_mit_trajectory_rejects_velocity_budget_overrun(self) -> None:
        base_config = self._base_config()
        config = replace(
            base_config,
            safety=replace(base_config.safety, hard_speed_abort_abs=10.0),
            identification=replace(base_config.identification, generation_safety_margin_ratio=0.80),
            dynamic_mit=replace(
                base_config.dynamic_mit,
                trajectory_type="sine",
                position_amplitude=1.0,
                frequency_hz=2.0,
                velocity_limit=2.0,
            ),
        )

        with self.assertRaisesRegex(ValueError, "lower position_amplitude or frequency_hz"):
            build_dynamic_mit_trajectory(config, target_motor_id=1)

    def test_dynamic_mit_trajectory_stays_within_velocity_budgets(self) -> None:
        base_config = self._base_config()
        trajectory = build_dynamic_mit_trajectory(base_config, target_motor_id=1)

        self.assertLessEqual(float(np.max(np.abs(trajectory.velocity))), float(base_config.dynamic_mit.velocity_limit) + 1.0e-9)
        self.assertLess(
            float(np.max(np.abs(trajectory.velocity))),
            float(base_config.safety.hard_speed_abort_abs) * float(base_config.identification.generation_safety_margin_ratio),
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
                low_speed=replace(
                    base_config.low_speed,
                    speed_points=(0.05, 0.10),
                    ramp_acceleration=8.0,
                    hold_duration=0.02,
                    micro_motion_record_duration=0.03,
                    micro_motion_velocity_limit=0.20,
                    micro_motion_frequency_hz=1.0,
                ),
                identification=replace(
                    base_config.identification,
                    steady_speed_points=(0.5, 1.0, 2.0),
                    repeat_count=1,
                    savgol_window=9,
                    savgol_polyorder=2,
                    min_publishable_rounds=1,
                    min_joint_fit_sample_count=3,
                    friction_rmse_publish_threshold=10.0,
                    inertia_rmse_publish_threshold=10.0,
                ),
                dynamic_mit=replace(
                    base_config.dynamic_mit,
                    enabled=True,
                    position_amplitude=0.15,
                    velocity_limit=2.0,
                    frequency_hz=1.0,
                    warmup_duration=0.0,
                    record_duration=0.04,
                    kp=4.0,
                    kd=0.5,
                    min_fit_sample_count=3,
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
            self.assertTrue(any(name.startswith("low_speed_") for name in phase_names))
            self.assertTrue(any(name.startswith("speed_hold_") for name in phase_names))
            self.assertTrue(any(name.startswith("inertia_") for name in phase_names))
            self.assertTrue(any(name.startswith("dynamic_mit_") for name in phase_names))
            self.assertIn("stiction_evidence", artifact.capture.__dataclass_fields__)
            phase_names_array = artifact.capture.phase_name.astype(str)
            breakaway_torque_mask = np.asarray(
                [
                    name.startswith("breakaway_pos_step_") or name.startswith("breakaway_neg_step_")
                    for name in phase_names_array
                ],
                dtype=bool,
            )
            self.assertTrue(np.any(np.abs(artifact.capture.torque_ff_cmd[breakaway_torque_mask]) > 0.0))
            self.assertTrue(np.allclose(artifact.capture.torque_ff_cmd[~breakaway_torque_mask], 0.0))
            self.assertGreater(float(artifact.identification.breakaway.torque_positive), 0.0)
            self.assertLess(float(artifact.identification.breakaway.torque_negative), 0.0)
            self.assertTrue(np.isfinite(float(artifact.identification.friction.tau_c)))
            self.assertTrue(np.isfinite(float(artifact.identification.inertia.inertia)))
            self.assertEqual(artifact.identification.metadata["model_kind"], PIECEWISE_STATIC_LINEAR_KIND)
            self.assertIn(artifact.identification.metadata["fit_model_kind"], ("static_v1", "joint_static_dynamic_v1"))
            self.assertIn("friction_model", artifact.identification.metadata)

            with np.load(result.summary_paths.run_summary_path, allow_pickle=False) as summary:
                self.assertIn("tau_c", summary.files)
                self.assertIn("inertia", summary.files)
                self.assertIn("recommended_for_compensation", summary.files)

            latest_path = Path(tmpdir) / "latest_motor_parameters.json"
            self.assertTrue(latest_path.exists())
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            latest_text = latest_path.read_text(encoding="utf-8")
            for legacy_token in ("ta" + "nh", "open" + "arm", "e" + "xp("):
                self.assertNotIn(legacy_token, latest_text)
            self.assertEqual(payload["speed_limit_rad_s"], 10.0)
            self.assertEqual(payload["results_dir"], str(Path(tmpdir)))
            self.assertIn("1", payload["motors"])
            self.assertEqual(payload["motors"]["1"]["motor_id"], 1)
            self.assertEqual(payload["motors"]["1"]["source_run_label"], Path(result.manifest_path).parent.name)
            self.assertEqual(payload["motors"]["1"]["model_version"], "1.0")
            self.assertEqual(payload["motors"]["1"]["model_kind"], PIECEWISE_STATIC_LINEAR_KIND)
            self.assertIn("fit_method", payload["motors"]["1"])
            self.assertIn("source_phases", payload["motors"]["1"])
            self.assertIn("confidence", payload["motors"]["1"])
            self.assertIn("quality_flags", payload["motors"]["1"])
            self.assertEqual(payload["motors"]["1"]["friction_model"]["kind"], PIECEWISE_STATIC_LINEAR_KIND)
            self.assertIn("embedded_piecewise_linear_friction", payload["motors"]["1"]["export_models"])

    def test_identify_round_selects_best_inertia_savgol_candidate(self) -> None:
        base_config = self._base_config()
        config = replace(
            base_config,
            identification=replace(
                base_config.identification,
                inertia_savgol_window_candidates=(21, 61),
                friction_rmse_publish_threshold=1.0,
                inertia_rmse_publish_threshold=1.0,
            ),
            dynamic_mit=replace(base_config.dynamic_mit, enabled=False),
        )

        identification = _identify_round(
            config=config,
            capture=self._synthetic_inertia_capture(),
            mode="identify-all",
            breakaway_result=self._synthetic_breakaway(),
        )

        inertia_metadata = identification.inertia.metadata
        candidate_results = inertia_metadata["savgol_window_candidates"]
        by_window = {int(item["window"]): item for item in candidate_results}
        self.assertEqual(inertia_metadata["selected_savgol_window"], 61)
        self.assertLess(float(by_window[61]["valid_rmse"]), float(by_window[21]["valid_rmse"]))
        self.assertTrue(identification.validation.recommended_for_compensation)

    def test_identify_round_rejects_publish_when_both_breakaway_directions_hit_scan_limit(self) -> None:
        base_config = self._base_config()
        config = replace(
            base_config,
            identification=replace(
                base_config.identification,
                inertia_savgol_window_candidates=(21, 61),
                friction_rmse_publish_threshold=1.0,
                inertia_rmse_publish_threshold=1.0,
            ),
            dynamic_mit=replace(base_config.dynamic_mit, enabled=False),
        )

        identification = _identify_round(
            config=config,
            capture=self._synthetic_inertia_capture(),
            mode="identify-all",
            breakaway_result=self._synthetic_breakaway(positive_limit=True, negative_limit=True),
        )

        self.assertFalse(identification.validation.recommended_for_compensation)
        self.assertEqual(identification.validation.metadata["status"], "rejected")
        self.assertIn("breakaway_scan_limit_reached=both", identification.validation.metadata["reasons"])

    def test_identify_round_records_single_breakaway_scan_limit_without_rejecting(self) -> None:
        base_config = self._base_config()
        config = replace(
            base_config,
            identification=replace(
                base_config.identification,
                inertia_savgol_window_candidates=(21, 61),
                friction_rmse_publish_threshold=1.0,
                inertia_rmse_publish_threshold=1.0,
            ),
            dynamic_mit=replace(base_config.dynamic_mit, enabled=False),
        )

        identification = _identify_round(
            config=config,
            capture=self._synthetic_inertia_capture(),
            mode="identify-all",
            breakaway_result=self._synthetic_breakaway(positive_limit=True, negative_limit=False),
        )

        self.assertTrue(identification.validation.recommended_for_compensation)
        self.assertTrue(identification.validation.metadata["breakaway_scan_limit"]["positive"])
        self.assertFalse(identification.validation.metadata["breakaway_scan_limit"]["negative"])

    def test_dynamic_mit_generates_command_rich_capture(self) -> None:
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
                dynamic_mit=replace(
                    base_config.dynamic_mit,
                    enabled=True,
                    trajectory_type="sine",
                    position_amplitude=0.2,
                    velocity_limit=2.0,
                    frequency_hz=1.0,
                    warmup_duration=0.0,
                    record_duration=0.05,
                    kp=4.0,
                    kd=0.5,
                    min_fit_sample_count=3,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = CommandTriggeredFeedbackFakeTransport(motor_ids=base_config.motor_ids)

            result = run_dynamic_mit(config, transport_factory=lambda: transport, show_rerun_viewer=False)

            self.assertTrue(transport.closed)
            self.assertTrue(any(kind == "mit_state" for kind, _, _ in transport.writes))
            capture_path = result.artifacts[0].capture_path
            with np.load(capture_path, allow_pickle=False) as capture:
                phase_names = capture["phase_name"].astype(str)
                self.assertTrue(any(name.startswith("dynamic_mit_") for name in phase_names))
                self.assertIn("kp_cmd", capture.files)
                self.assertIn("kd_cmd", capture.files)
                self.assertIn("torque_ff_cmd", capture.files)
                self.assertIn("position_error", capture.files)
                self.assertIn("velocity_error", capture.files)
                self.assertIn("used_for_fit", capture.files)
                self.assertTrue(np.any(capture["used_for_fit"]))
                self.assertTrue(np.allclose(capture["torque_ff_cmd"], 0.0))

    def test_dynamic_mit_position_commands_are_relative_to_current_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            initial_position = -2.25
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
                dynamic_mit=replace(
                    base_config.dynamic_mit,
                    enabled=True,
                    trajectory_type="sine",
                    position_amplitude=0.2,
                    velocity_limit=2.0,
                    frequency_hz=1.0,
                    warmup_duration=0.0,
                    record_duration=0.05,
                    kp=4.0,
                    kd=0.5,
                    min_fit_sample_count=3,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = CommandTriggeredFeedbackFakeTransport(
                motor_ids=base_config.motor_ids,
                initial_position_by_motor={1: initial_position},
            )

            result = run_dynamic_mit(config, transport_factory=lambda: transport, show_rerun_viewer=False)

            capture_path = result.artifacts[0].capture_path
            with np.load(capture_path, allow_pickle=False) as capture:
                phase_names = capture["phase_name"].astype(str)
                anchor_mask = phase_names == "dynamic_mit_anchor"
                train_mask = np.asarray([name.startswith("dynamic_mit_train") for name in phase_names], dtype=bool)
                self.assertTrue(np.any(anchor_mask))
                self.assertTrue(np.any(train_mask))
                anchor_position = float(capture["position_cmd"][anchor_mask][0])
                train_positions = capture["position_cmd"][train_mask]
                self.assertAlmostEqual(anchor_position, initial_position, delta=0.05)
                self.assertLess(abs(float(train_positions[0]) - anchor_position), 0.05)
                self.assertLess(float(np.nanmax(np.abs(train_positions - anchor_position))), 0.25)

    def test_dynamic_mit_aborts_at_runtime_velocity_limit(self) -> None:
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
                dynamic_mit=replace(
                    base_config.dynamic_mit,
                    enabled=True,
                    trajectory_type="sine",
                    position_amplitude=0.14,
                    velocity_limit=0.5,
                    frequency_hz=1.0,
                    warmup_duration=0.0,
                    record_duration=0.20,
                    kp=0.0,
                    kd=0.25,
                    min_fit_sample_count=3,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            transport = ClosedLoopFakeTransport(
                motor_ids=base_config.motor_ids,
                trip_motor_id=1,
                trip_command_threshold=0.05,
                trip_velocity=0.7,
            )

            with self.assertRaises(RuntimeAbortError) as context:
                run_dynamic_mit(config, transport_factory=lambda: transport, show_rerun_viewer=False)

            self.assertEqual(context.exception.event.reason, "dynamic_mit_velocity_abort")
            self.assertEqual(context.exception.event.velocity_limit, 0.5)
            self.assertGreaterEqual(transport.zero_command_count, 1)
            self.assertTrue(transport.closed)

    def test_identify_all_keeps_static_model_when_joint_fit_has_higher_rmse(self) -> None:
        base_config = self._base_config()
        config = replace(
            base_config,
            identification=replace(
                base_config.identification,
                savgol_window=5,
                savgol_polyorder=2,
                min_platform_sample_count=3,
                min_joint_fit_sample_count=8,
                joint_dynamic_mit_weight=8.0,
                friction_rmse_publish_threshold=0.08,
                inertia_rmse_publish_threshold=0.08,
            ),
            dynamic_mit=replace(base_config.dynamic_mit, min_fit_sample_count=8),
        )
        true_j = 0.05
        true_viscous = 0.04
        true_tau_c = 0.18
        true_tau_bias = 0.01

        time_values: list[float] = []
        velocity_values: list[float] = []
        torque_values: list[float] = []
        phase_values: list[str] = []
        used_for_fit_values: list[bool] = []
        t = 0.0

        def append_segment(phase_name: str, velocity: np.ndarray, torque: np.ndarray, *, used_for_fit: bool) -> None:
            nonlocal t
            for vel, tau in zip(velocity, torque):
                time_values.append(float(t))
                velocity_values.append(float(vel))
                torque_values.append(float(tau))
                phase_values.append(str(phase_name))
                used_for_fit_values.append(bool(used_for_fit))
                t += 0.01

        for phase_name, commanded_velocity in (
            ("speed_hold_train_-2.0", -2.0),
            ("speed_hold_train_2.0", 2.0),
            ("speed_hold_valid_-2.0", -2.0),
            ("speed_hold_valid_2.0", 2.0),
        ):
            velocity = np.full(8, float(commanded_velocity), dtype=np.float64)
            torque = true_viscous * velocity + true_tau_c * np.sign(velocity) + true_tau_bias
            append_segment(phase_name, velocity, torque, used_for_fit=False)

        inertia_time = np.linspace(0.0, 1.0, 80, dtype=np.float64)
        inertia_velocity = 1.4 * np.sin(2.0 * np.pi * inertia_time)
        inertia_acceleration = np.gradient(inertia_velocity, 0.01, edge_order=1)
        inertia_torque = (
            true_j * inertia_acceleration
            + true_viscous * inertia_velocity
            + true_tau_c * np.sign(inertia_velocity)
            + true_tau_bias
        )
        append_segment("inertia_train_wave", inertia_velocity[:48], inertia_torque[:48], used_for_fit=False)
        append_segment("inertia_valid_wave", inertia_velocity[48:], inertia_torque[48:], used_for_fit=False)

        dynamic_velocity = np.linspace(-1.5, 1.5, 40, dtype=np.float64)
        dynamic_acceleration = np.gradient(dynamic_velocity, 0.01, edge_order=1)
        dynamic_torque = (
            true_j * dynamic_acceleration
            + true_viscous * dynamic_velocity
            + true_tau_c * np.sign(dynamic_velocity)
            + true_tau_bias
            + 1.0
        )
        append_segment("dynamic_mit_train_bad", dynamic_velocity[:28], dynamic_torque[:28], used_for_fit=True)
        append_segment("dynamic_mit_valid_bad", dynamic_velocity[28:], dynamic_torque[28:], used_for_fit=True)

        time = np.asarray(time_values, dtype=np.float64)
        velocity = np.asarray(velocity_values, dtype=np.float64)
        torque = np.asarray(torque_values, dtype=np.float64)
        capture = RoundCapture(
            group_index=1,
            round_index=1,
            target_motor_id=1,
            motor_name="motor_01",
            time=time,
            motor_id=np.full(time.size, 1, dtype=np.int64),
            position=np.zeros(time.size, dtype=np.float64),
            velocity=velocity,
            torque_feedback=torque,
            command_raw=np.zeros(time.size, dtype=np.float64),
            command=np.zeros(time.size, dtype=np.float64),
            position_cmd=np.zeros(time.size, dtype=np.float64),
            velocity_cmd=velocity,
            acceleration_cmd=np.zeros(time.size, dtype=np.float64),
            phase_name=np.asarray(phase_values, dtype=str),
            state=np.ones(time.size, dtype=np.uint8),
            mos_temperature=np.full(time.size, 30.0, dtype=np.float64),
            id_match_ok=np.ones(time.size, dtype=bool),
            filtered_velocity=np.zeros(time.size, dtype=np.float64),
            estimated_acceleration=np.zeros(time.size, dtype=np.float64),
            friction_term=np.zeros(time.size, dtype=np.float64),
            inertia_term=np.zeros(time.size, dtype=np.float64),
            guard_scale=np.ones(time.size, dtype=np.float64),
            used_for_fit=np.asarray(used_for_fit_values, dtype=bool),
            tau_mit_est=torque,
            metadata={"mode": "identify-all"},
        )
        breakaway_result = BreakawayIdentificationResult(
            torque_positive=0.2,
            torque_negative=-0.18,
            tau_static=0.19,
            tau_bias=0.01,
            metadata={},
        )

        identification = _identify_round(
            config=config,
            capture=capture,
            mode="identify-all",
            breakaway_result=breakaway_result,
        )

        self.assertEqual(identification.metadata["model_kind"], PIECEWISE_STATIC_LINEAR_KIND)
        self.assertEqual(identification.metadata["fit_model_kind"], "static_v1")
        self.assertEqual(identification.validation.metadata["model_selection"], "static_selected_over_joint")
        joint_candidate = identification.validation.metadata["joint_candidate"]
        self.assertLess(float(identification.validation.friction_rmse), float(joint_candidate["friction_rmse"]))
        self.assertLess(float(identification.validation.inertia_rmse), float(joint_candidate["inertia_rmse"]))

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
                    low_speed=replace(
                        base_config.low_speed,
                        speed_points=(0.05,),
                        ramp_acceleration=8.0,
                        hold_duration=0.01,
                        micro_motion_record_duration=0.0,
                    ),
                    identification=replace(
                        base_config.identification,
                        steady_speed_points=(0.5, 1.0, 2.0),
                        repeat_count=1,
                        savgol_window=9,
                        savgol_polyorder=2,
                        min_publishable_rounds=1,
                        friction_rmse_publish_threshold=10.0,
                        inertia_rmse_publish_threshold=10.0,
                    ),
                    dynamic_mit=replace(base_config.dynamic_mit, enabled=False),
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
                low_speed=replace(
                    base_config.low_speed,
                    speed_points=(0.05,),
                    ramp_acceleration=8.0,
                    hold_duration=0.01,
                    micro_motion_record_duration=0.0,
                ),
                identification=replace(
                    base_config.identification,
                    steady_speed_points=(0.5, 1.0, 2.0, 4.0, 8.0),
                    generation_safety_margin_ratio=0.85,
                    repeat_count=1,
                    savgol_window=9,
                    savgol_polyorder=2,
                    min_publishable_rounds=1,
                ),
                dynamic_mit=replace(base_config.dynamic_mit, enabled=False),
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
            self.assertIn("3", payload["motors"])
            self.assertEqual(payload["motors"]["3"]["publish_status"], "rejected")
            self.assertFalse(payload["motors"]["3"]["recommended_for_compensation"])

    def test_save_latest_parameters_uses_only_accepted_rounds_and_records_unpublished_latest_attempt(self) -> None:
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
            self.assertEqual(motor_1["friction_model"]["kind"], PIECEWISE_STATIC_LINEAR_KIND)
            self.assertAlmostEqual(float(motor_1["friction_model"]["parameters"]["tau_static"]), 0.22, places=6)
            self.assertIn("embedded_piecewise_linear_friction", motor_1["export_models"])
            self.assertAlmostEqual(float(motor_1["export_models"]["embedded_piecewise_linear_friction"]["tau_c"]), 0.12, places=6)

            motor_2 = payload["motors"]["2"]
            self.assertEqual(motor_2["source_run_label"], store.run_label)
            self.assertEqual(motor_2["publish_status"], "not_published")
            self.assertEqual(motor_2["accepted_round_count"], 1)
            self.assertEqual(motor_2["selected_rounds"], [1])
            self.assertAlmostEqual(float(motor_2["tau_c"]), 0.18, places=6)
            self.assertEqual(motor_2["previous_published_model"]["source_run_label"], "seed_run")
            self.assertAlmostEqual(float(motor_2["previous_published_model"]["tau_c"]), 0.21, places=6)

            store_again = ResultStore(config, mode="identify-all")
            store_again.save_latest_parameters(artifacts)
            payload_again = json.loads(latest_path.read_text(encoding="utf-8"))
            motor_2_again = payload_again["motors"]["2"]
            self.assertEqual(motor_2_again["source_run_label"], store_again.run_label)
            self.assertEqual(motor_2_again["publish_status"], "not_published")
            self.assertEqual(motor_2_again["previous_published_model"]["source_run_label"], "seed_run")

    def test_save_latest_parameters_rejects_model_that_exceeds_compensation_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(5,)),
                identification=replace(
                    base_config.identification,
                    repeat_count=2,
                    min_publishable_rounds=2,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )
            artifacts = [
                self._synthetic_artifact(
                    motor_id=5,
                    group_index=1,
                    tau_static=3.0,
                    tau_c=0.50,
                    viscous=0.0,
                    inertia=0.0,
                    friction_rmse=0.01,
                    inertia_rmse=0.01,
                    recommended=True,
                    tmpdir=tmpdir,
                ),
                self._synthetic_artifact(
                    motor_id=5,
                    group_index=2,
                    tau_static=3.0,
                    tau_c=0.50,
                    viscous=0.0,
                    inertia=0.0,
                    friction_rmse=0.01,
                    inertia_rmse=0.01,
                    recommended=True,
                    tmpdir=tmpdir,
                ),
            ]
            store = ResultStore(config, mode="identify-all")
            store.save_latest_parameters(artifacts)

            payload = json.loads((Path(tmpdir) / "latest_motor_parameters.json").read_text(encoding="utf-8"))
            motor_5 = payload["motors"]["5"]
            self.assertEqual(motor_5["publish_status"], "rejected")
            self.assertFalse(motor_5["recommended_for_compensation"])
            self.assertIn("model_exceeds_compensation_budget", motor_5["publish_detail"])
            self.assertAlmostEqual(float(motor_5["friction_model"]["parameters"]["tau_static"]), 3.0, places=6)

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

    def test_compensation_uses_piecewise_static_level_near_zero_speed(self) -> None:
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
                self.assertGreater(float(np.max(capture["command"])), 0.0)
                self.assertLessEqual(float(np.max(capture["command"])), 0.60 + 1.0e-6)
                self.assertGreater(float(np.max(capture["friction_term"])), 0.55)

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
                                "tau_bias": 0.4,
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
                self.assertTrue(np.all(np.isfinite(capture["friction_term"])))
                self.assertGreater(float(np.max(np.abs(capture["inertia_term"]))), 0.218751)
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

            capture_files = sorted(Path(tmpdir).glob("runs/*_compensation/group_01/motor_01/capture.npz"))
            self.assertEqual(len(capture_files), 1)
            with np.load(capture_files[0], allow_pickle=False) as capture:
                self.assertGreater(int(capture["time"].size), 0)

            manifest_files = sorted(Path(tmpdir).glob("runs/*_compensation/run_manifest.json"))
            self.assertEqual(len(manifest_files), 1)
            manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["abort_event"]["reason"], "soft_speed_abort")
            self.assertEqual(manifest["capture_files"], [str(capture_files[0])])

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
