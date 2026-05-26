from __future__ import annotations

import unittest
from unittest.mock import patch

from mit_sender.commands import SelectedMotorCommand, build_uniform_commands
from mit_sender.damiao import (
    CLEAR_ERROR_CMD,
    CONTROL_REPEAT,
    DISABLE_CMD,
    ENABLE_CMD,
    MIT_MODE,
    MIT_MODE_CODE,
    PARAM_ESC_ID_RID,
    PARAM_MST_ID_RID,
    PARAM_CTRL_MODE_RID,
    ZERO_POSITION_CMD,
    DM_Motor_Type,
    DamiaoMitController,
    MitCommand,
    MotorSpec,
    build_control_cmd_frame,
    build_mit_frame,
    build_param_store_frame,
    build_param_write_frame,
    build_param_write_uint32_frame,
    build_zero_position_frame,
    decode_feedback_frame,
    default_motor_specs,
    ensure_interface_ready,
    format_can_setup_commands,
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
            build_param_write_frame(0x01, PARAM_CTRL_MODE_RID, bytes([MIT_MODE_CODE, 0, 0, 0])),
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

    def test_zero_position_frame_uses_control_command_payload(self) -> None:
        self.assertEqual(
            build_zero_position_frame(0x03, 0x100),
            (0x103, bytes([0xFF] * 7 + [ZERO_POSITION_CMD])),
        )

    def test_param_write_uint32_frame_uses_little_endian_value(self) -> None:
        self.assertEqual(
            build_param_write_uint32_frame(0x01, PARAM_ESC_ID_RID, 0x03),
            (0x7FF, bytes([0x01, 0x00, 0x55, PARAM_ESC_ID_RID, 0x03, 0x00, 0x00, 0x00])),
        )

    def test_build_param_store_frame_uses_flash_store_command(self) -> None:
        self.assertEqual(
            build_param_store_frame(0x20),
            (0x7FF, bytes([0x20, 0x00, 0xAA, 0x01])),
        )

    def test_set_mit_mode_persistent_disables_writes_mode_then_stores(self) -> None:
        transport = FakeCanTransport()
        controller = DamiaoMitController(
            transport,
            [MotorSpec(motor_id=1, can_id=0x01, mst_id=0x11, motor_type=DM_Motor_Type.DM8009)],
        )

        controller.set_mit_mode_persistent_raw(0x01, MIT_MODE)

        self.assertEqual(
            transport.sent[:CONTROL_REPEAT],
            [build_control_cmd_frame(0x01, DISABLE_CMD)] * CONTROL_REPEAT,
        )
        self.assertEqual(
            transport.sent[CONTROL_REPEAT],
            build_param_write_uint32_frame(0x01, PARAM_CTRL_MODE_RID, MIT_MODE_CODE),
        )
        self.assertEqual(
            transport.sent[CONTROL_REPEAT + 1 : CONTROL_REPEAT * 2 + 1],
            [build_control_cmd_frame(0x01, DISABLE_CMD)] * CONTROL_REPEAT,
        )
        self.assertEqual(transport.sent[CONTROL_REPEAT * 2 + 1], build_param_store_frame(0x01))

    def test_set_motor_ids_persistent_disables_writes_ids_then_stores_new_id(self) -> None:
        transport = FakeCanTransport()
        controller = DamiaoMitController(
            transport,
            [MotorSpec(motor_id=1, can_id=0x01, mst_id=0x11, motor_type=DM_Motor_Type.DM8009)],
        )

        controller.set_motor_ids_persistent_raw(
            current_can_id=0x01,
            current_mode_offset=MIT_MODE,
            new_can_id=0x20,
            new_mst_id=0x30,
        )

        self.assertEqual(
            transport.sent[:CONTROL_REPEAT],
            [build_control_cmd_frame(0x01, DISABLE_CMD)] * CONTROL_REPEAT,
        )
        self.assertEqual(
            transport.sent[CONTROL_REPEAT],
            build_param_write_uint32_frame(0x01, PARAM_MST_ID_RID, 0x30),
        )
        self.assertEqual(
            transport.sent[CONTROL_REPEAT + 1],
            build_param_write_uint32_frame(0x01, PARAM_ESC_ID_RID, 0x20),
        )
        self.assertEqual(
            transport.sent[CONTROL_REPEAT + 2 : CONTROL_REPEAT * 2 + 2],
            [build_control_cmd_frame(0x20, DISABLE_CMD)] * CONTROL_REPEAT,
        )
        self.assertEqual(transport.sent[CONTROL_REPEAT * 2 + 2], build_param_store_frame(0x20))

    def test_debug_configure_sequence_disables_then_sets_zero_mit_and_ids(self) -> None:
        transport = FakeCanTransport()
        controller = DamiaoMitController(
            transport,
            [MotorSpec(motor_id=1, can_id=0x01, mst_id=0x11, motor_type=DM_Motor_Type.DM8009)],
        )

        controller.configure_single_motor_raw(
            current_can_id=0x01,
            current_mode_offset=MIT_MODE,
            new_can_id=0x03,
            new_mst_id=0x13,
        )

        self.assertEqual(
            transport.sent[:CONTROL_REPEAT],
            [build_control_cmd_frame(0x01, DISABLE_CMD)] * CONTROL_REPEAT,
        )
        self.assertEqual(transport.sent[CONTROL_REPEAT], build_zero_position_frame(0x01, MIT_MODE))
        self.assertEqual(
            transport.sent[CONTROL_REPEAT + 1],
            build_param_write_uint32_frame(0x01, PARAM_CTRL_MODE_RID, MIT_MODE_CODE),
        )
        self.assertEqual(
            transport.sent[CONTROL_REPEAT + 2],
            build_param_write_uint32_frame(0x01, PARAM_MST_ID_RID, 0x13),
        )
        self.assertEqual(
            transport.sent[CONTROL_REPEAT + 3],
            build_param_write_uint32_frame(0x01, PARAM_ESC_ID_RID, 0x03),
        )
        self.assertEqual(
            transport.sent[CONTROL_REPEAT + 4 : CONTROL_REPEAT * 2 + 4],
            [build_control_cmd_frame(0x03, DISABLE_CMD)] * CONTROL_REPEAT,
        )
        self.assertEqual(transport.sent[CONTROL_REPEAT * 2 + 4], build_param_store_frame(0x03))

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

    def test_format_can_setup_commands_returns_three_manual_commands(self) -> None:
        self.assertEqual(
            format_can_setup_commands("can0", 1_000_000, 5_000_000),
            "\n".join(
                [
                    "sudo ip link set can0 down",
                    "sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on",
                    "sudo ip link set can0 up",
                ]
            ),
        )

    def test_missing_can_interface_error_lists_available_interfaces(self) -> None:
        with (
            patch("mit_sender.damiao.get_can_interface_state", return_value=None),
            patch("mit_sender.damiao.list_available_can_interfaces", return_value=("can0", "can1")),
        ):
            with self.assertRaises(RuntimeError) as error:
                ensure_interface_ready("can2", 1_000_000, 5_000_000)

        message = str(error.exception)
        self.assertIn("CAN interface can2 does not exist", message)
        self.assertIn("检测到: can0, can1", message)
        self.assertIn("sudo ip link set can2 down", message)


if __name__ == "__main__":
    unittest.main()
