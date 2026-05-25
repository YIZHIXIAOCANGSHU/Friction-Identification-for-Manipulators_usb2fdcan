from __future__ import annotations

import unittest

from mit_sender.commands import SelectedMotorCommand, build_uniform_commands
from mit_sender.damiao import (
    CLEAR_ERROR_CMD,
    CONTROL_REPEAT,
    DISABLE_CMD,
    ENABLE_CMD,
    DM_Motor_Type,
    DamiaoMitController,
    MitCommand,
    MotorSpec,
    build_control_cmd_frame,
    build_mit_frame,
    build_param_write_frame,
    decode_feedback_frame,
    default_motor_specs,
)


class FakeCanTransport:
    def __init__(self) -> None:
        self.sent: list[tuple[int, bytes]] = []
        self.closed = False

    def send(self, can_id: int, payload: bytes) -> None:
        self.sent.append((int(can_id), bytes(payload)))

    def close(self) -> None:
        self.closed = True


class DamiaoMitSenderTests(unittest.TestCase):
    def test_zero_mit_frame_is_eight_byte_mit_payload(self) -> None:
        can_id, payload = build_mit_frame(
            0x01,
            DM_Motor_Type.DM8009,
            kp=0.0,
            kd=0.0,
            position=0.0,
            velocity=0.0,
            torque=0.0,
        )

        self.assertEqual(can_id, 0x01)
        self.assertEqual(len(payload), 8)

    def test_prepare_motor_clears_error_switches_mit_and_enables(self) -> None:
        transport = FakeCanTransport()
        controller = DamiaoMitController(
            transport,
            [MotorSpec(motor_id=1, can_id=0x01, mst_id=0x11, motor_type=DM_Motor_Type.DM8009)],
        )

        controller.prepare_motor(1)

        self.assertEqual(
            transport.sent[:CONTROL_REPEAT],
            [build_control_cmd_frame(0x01, CLEAR_ERROR_CMD)] * CONTROL_REPEAT,
        )
        self.assertEqual(
            transport.sent[CONTROL_REPEAT],
            build_param_write_frame(0x01, 10, bytes([1, 0, 0, 0])),
        )
        self.assertEqual(
            transport.sent[CONTROL_REPEAT + 1 : CONTROL_REPEAT * 2 + 1],
            [build_control_cmd_frame(0x01, ENABLE_CMD)] * CONTROL_REPEAT,
        )

    def test_disable_motor_sends_repeated_disable_frames(self) -> None:
        transport = FakeCanTransport()
        controller = DamiaoMitController(
            transport,
            [MotorSpec(motor_id=1, can_id=0x01, mst_id=0x11, motor_type=DM_Motor_Type.DM8009)],
        )

        controller.disable_motor(1)

        self.assertEqual(
            transport.sent,
            [build_control_cmd_frame(0x01, DISABLE_CMD)] * CONTROL_REPEAT,
        )

    def test_prepare_and_send_mit_adds_command_after_enable_sequence(self) -> None:
        transport = FakeCanTransport()
        spec = MotorSpec(motor_id=1, can_id=0x01, mst_id=0x11, motor_type=DM_Motor_Type.DM8009)
        controller = DamiaoMitController(transport, [spec])
        command = MitCommand(position=0.1, velocity=0.2, kp=1.0, kd=0.3, torque_ff=0.4)

        controller.prepare_and_send_mit(1, command)

        self.assertEqual(
            transport.sent[-1],
            build_mit_frame(0x01, spec.motor_type, 1.0, 0.3, 0.1, 0.2, 0.4),
        )

    def test_build_uniform_commands_reuses_one_command_for_each_motor(self) -> None:
        command = SelectedMotorCommand(
            motor_id=0,
            position=0.1,
            velocity=0.2,
            kp=1.0,
            kd=0.3,
            torque_ff=0.4,
        )

        commands = build_uniform_commands([1, 3, 7], command)

        self.assertEqual(
            commands,
            [
                SelectedMotorCommand(1, 0.1, 0.2, 1.0, 0.3, 0.4),
                SelectedMotorCommand(3, 0.1, 0.2, 1.0, 0.3, 0.4),
                SelectedMotorCommand(7, 0.1, 0.2, 1.0, 0.3, 0.4),
            ],
        )

    def test_decode_feedback_frame_maps_feedback_id_to_motor(self) -> None:
        frame = decode_feedback_frame(
            0x11,
            bytes([0x11, 0x7F, 0xFF, 0x7F, 0xF7, 0xFF, 40, 50]),
            default_motor_specs(),
        )

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.motor_id, 1)
        self.assertEqual(frame.can_id, 0x11)
        self.assertEqual(frame.state, 1)
        self.assertEqual(frame.controller_id, 1)
        self.assertAlmostEqual(frame.position, 0.0, delta=5e-4)
        self.assertAlmostEqual(frame.velocity, 0.0, delta=2e-2)
        self.assertAlmostEqual(frame.torque, 0.0, delta=2e-2)
        self.assertAlmostEqual(frame.mos_temperature, 40.0, places=6)
        self.assertAlmostEqual(frame.rotor_temperature, 50.0, places=6)


if __name__ == "__main__":
    unittest.main()
