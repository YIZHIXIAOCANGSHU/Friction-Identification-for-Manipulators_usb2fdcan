from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from friction_identification_core.config import DEFAULT_CONFIG_PATH, apply_overrides, load_config
from friction_identification_core.io import (
    COMMAND_PAYLOAD_STRUCT,
    RECV_FRAME_HEAD,
    RECV_FRAME_SIZE,
    RECV_FRAME_STRUCT,
)
from friction_identification_core.workflow import run_step_torque


class ClosedLoopFakeTransport:
    def __init__(
        self,
        motor_ids: tuple[int, ...],
        *,
        step_dt: float = 0.005,
        initial_position: float = 0.0,
        command_gain: float = 6.0,
        velocity_damping: float = 1.4,
        position_stiffness: float = 1.2,
        trip_motor_id: int | None = None,
        trip_after_target_frames: int = 1,
        trip_velocity: float | None = None,
    ) -> None:
        self._motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
        self._step_dt = float(step_dt)
        self._pending = bytearray()
        self._last_commands = np.zeros(7, dtype=np.float64)
        self._command_gain = float(command_gain)
        self._velocity_damping = float(velocity_damping)
        self._position_stiffness = float(position_stiffness)
        self._states = {
            int(motor_id): {
                "position": float(initial_position),
                "velocity": 0.0,
                "torque": 0.0,
                "temperature": 30.0 + float(motor_id),
            }
            for motor_id in self._motor_ids
        }
        self._target_frame_count = 0
        self._trip_motor_id = None if trip_motor_id is None else int(trip_motor_id)
        self._trip_after_target_frames = max(int(trip_after_target_frames), 1)
        self._trip_velocity = None if trip_velocity is None else float(trip_velocity)
        self.writes: list[bytes] = []
        self.closed = False

    def _advance_state(self, motor_id: int) -> tuple[float, float, float, float]:
        state = self._states[int(motor_id)]
        command = float(self._last_commands[int(motor_id) - 1])
        acceleration = (
            self._command_gain * command
            - self._velocity_damping * float(state["velocity"])
            - self._position_stiffness * float(state["position"])
        )
        state["velocity"] = float(state["velocity"]) + self._step_dt * acceleration
        state["position"] = float(state["position"]) + self._step_dt * float(state["velocity"])
        state["torque"] = command
        if int(motor_id) == int(self._trip_motor_id or -1):
            self._target_frame_count += 1
            if self._target_frame_count >= self._trip_after_target_frames and self._trip_velocity is not None:
                state["velocity"] = float(self._trip_velocity)
        return (
            float(state["position"]),
            float(state["velocity"]),
            float(state["torque"]),
            float(state["temperature"]),
        )

    def _build_cycle_bytes(self) -> bytes:
        frames = bytearray()
        for motor_id in self._motor_ids:
            position, velocity, torque, temperature = self._advance_state(int(motor_id))
            frames.extend(
                RECV_FRAME_STRUCT.pack(
                    RECV_FRAME_HEAD,
                    int(motor_id),
                    1,
                    position,
                    velocity,
                    torque,
                    temperature,
                )
            )
        return bytes(frames)

    def read(self, size: int) -> bytes:
        while len(self._pending) < int(size):
            self._pending.extend(self._build_cycle_bytes())
        chunk = bytes(self._pending[:size])
        del self._pending[:size]
        return chunk

    def write(self, payload: bytes) -> int:
        self.writes.append(bytes(payload))
        values = COMMAND_PAYLOAD_STRUCT.unpack(payload[2:30])
        self._last_commands[:] = np.asarray(values, dtype=np.float64)
        return len(payload)

    def reset_input_buffer(self) -> None:
        self._pending.clear()

    def close(self) -> None:
        self.closed = True


class OneFrameReadTransport(ClosedLoopFakeTransport):
    def __init__(self, *args, read_sleep_s: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._read_sleep_s = max(float(read_sleep_s), 0.0)

    def read(self, size: int) -> bytes:
        _ = size
        if self._read_sleep_s > 0.0:
            time.sleep(self._read_sleep_s)
        while len(self._pending) < RECV_FRAME_SIZE:
            self._pending.extend(self._build_cycle_bytes())
        chunk = bytes(self._pending[:RECV_FRAME_SIZE])
        del self._pending[:RECV_FRAME_SIZE]
        return chunk


class StepTorqueWorkflowTests(unittest.TestCase):
    def _base_config(self):
        return load_config(DEFAULT_CONFIG_PATH)

    def test_default_config_exposes_step_torque_defaults(self) -> None:
        config = self._base_config()

        self.assertEqual(config.enabled_motor_ids, (1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(config.step_torque.initial_torque, 0.0)
        self.assertEqual(config.step_torque.torque_step, 0.1)
        self.assertEqual(config.step_torque.hold_duration, 1.0)
        self.assertEqual(config.step_torque.velocity_limit, 10.0)

    def test_motor_override_selects_requested_subset(self) -> None:
        config = self._base_config()

        self.assertEqual(apply_overrides(config, motors="all").enabled_motor_ids, (1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(apply_overrides(config, motors="2,4").enabled_motor_ids, (2, 4))

    def test_step_torque_run_stops_after_reaching_motor_torque_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                self._base_config(),
                motors=replace(self._base_config().motors, enabled_ids=(1,)),
                serial=replace(
                    self._base_config().serial,
                    read_chunk_size=RECV_FRAME_SIZE,
                    read_timeout=0.001,
                    sync_cycles_required=1,
                    sync_timeout=0.2,
                ),
                step_torque=replace(
                    self._base_config().step_torque,
                    hold_duration=0.05,
                ),
                control=replace(
                    self._base_config().control,
                    max_torque=np.asarray([0.25, 40.0, 27.0, 27.0, 7.0, 7.0, 9.0], dtype=np.float64),
                ),
                output=replace(self._base_config().output, results_dir=Path(tmpdir)),
            )

            transport = OneFrameReadTransport(
                motor_ids=(1, 2, 3, 4, 5, 6, 7),
                read_sleep_s=0.005,
                command_gain=2.0,
            )
            result = run_step_torque(config, transport_factory=lambda: transport, show_rerun_viewer=False)

        self.assertEqual(len(result.artifacts), 1)
        capture = result.artifacts[0].capture
        self.assertEqual(capture.metadata["stop_reason"], "max_torque_reached")
        self.assertFalse(capture.metadata["velocity_limit_reached"])
        self.assertTrue(np.isnan(float(capture.metadata["velocity_limit_trigger_command"])))
        self.assertTrue(np.isnan(float(capture.metadata["velocity_limit_trigger_feedback_torque"])))
        self.assertTrue(np.isnan(float(capture.metadata["velocity_limit_trigger_velocity"])))
        self.assertTrue(transport.closed)
        self.assertTrue(np.isclose(capture.command[0], 0.0, atol=1.0e-6))
        self.assertTrue(np.any(np.isclose(capture.command, 0.1, atol=1.0e-6)))
        self.assertTrue(np.any(np.isclose(capture.command, 0.2, atol=1.0e-6)))
        self.assertTrue(np.any(np.isclose(capture.command, 0.25, atol=1.0e-6)))
        self.assertLessEqual(float(np.max(capture.command)), 0.25 + 1.0e-9)

    def test_step_torque_run_switches_to_next_motor_after_overspeed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._base_config()
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1, 2)),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=RECV_FRAME_SIZE,
                    read_timeout=0.001,
                    sync_cycles_required=1,
                    sync_timeout=0.2,
                ),
                step_torque=replace(
                    base_config.step_torque,
                    hold_duration=0.05,
                ),
                control=replace(
                    base_config.control,
                    max_torque=np.asarray([0.3, 0.1, 27.0, 27.0, 7.0, 7.0, 9.0], dtype=np.float64),
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            transport = OneFrameReadTransport(
                motor_ids=(1, 2, 3, 4, 5, 6, 7),
                read_sleep_s=0.005,
                command_gain=2.0,
                trip_motor_id=1,
                trip_after_target_frames=3,
                trip_velocity=12.0,
            )
            result = run_step_torque(config, transport_factory=lambda: transport, show_rerun_viewer=False)

        self.assertEqual(len(result.artifacts), 2)
        first_capture = result.artifacts[0].capture
        second_capture = result.artifacts[1].capture
        self.assertEqual(first_capture.target_motor_id, 1)
        self.assertEqual(first_capture.metadata["stop_reason"], "velocity_limit_exceeded")
        self.assertTrue(first_capture.metadata["velocity_limit_reached"])
        self.assertAlmostEqual(
            float(first_capture.metadata["velocity_limit_trigger_command"]),
            float(first_capture.command[-1]),
            places=6,
        )
        self.assertAlmostEqual(
            float(first_capture.metadata["velocity_limit_trigger_feedback_torque"]),
            float(first_capture.torque_feedback[-1]),
            places=6,
        )
        self.assertAlmostEqual(
            float(first_capture.metadata["velocity_limit_trigger_velocity"]),
            float(first_capture.velocity[-1]),
            places=6,
        )
        self.assertEqual(second_capture.target_motor_id, 2)
        self.assertEqual(second_capture.metadata["stop_reason"], "max_torque_reached")


if __name__ == "__main__":
    unittest.main()
