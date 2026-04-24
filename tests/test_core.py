from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from friction_identification_core.config import DEFAULT_CONFIG_PATH, load_config
from friction_identification_core.io import (
    COMMAND_PAYLOAD_STRUCT,
    RECV_FRAME_HEAD,
    RECV_FRAME_STRUCT,
    SerialFrameParser,
    SingleMotorCommandAdapter,
    calculate_xor_checksum,
)


class StepTorqueCoreTests(unittest.TestCase):
    def test_parser_and_command_adapter(self) -> None:
        parser = SerialFrameParser(max_motor_id=7)
        payload = RECV_FRAME_STRUCT.pack(RECV_FRAME_HEAD, 3, 2, 1.0, -2.0, 0.5, 40.0)
        parser.feed(b"\x00\x01" + payload)
        frame = parser.pop_frame()
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.motor_id, 3)
        self.assertAlmostEqual(frame.velocity, -2.0, places=6)

        config = load_config(DEFAULT_CONFIG_PATH)
        adapter = SingleMotorCommandAdapter(motor_count=7, torque_limits=config.control.max_torque)
        command = adapter.pack(5, 99.0)
        payload_values = COMMAND_PAYLOAD_STRUCT.unpack(command[2:30])
        self.assertAlmostEqual(adapter.limit_command(5, 99.0), 7.0, places=6)
        self.assertAlmostEqual(payload_values[4], 7.0, places=6)
        self.assertTrue(all(abs(value) < 1.0e-9 for index, value in enumerate(payload_values) if index != 4))
        self.assertEqual(command[30], calculate_xor_checksum(command[:30]))

    def test_default_config_reads_step_torque_section(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)

        self.assertEqual(config.step_torque.initial_torque, 0.0)
        self.assertEqual(config.step_torque.torque_step, 0.1)
        self.assertEqual(config.step_torque.hold_duration, 1.0)
        self.assertEqual(config.step_torque.velocity_limit, 10.0)

    def test_invalid_step_torque_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "step_torque:",
                        "  torque_step: 0.0",
                        "  hold_duration: -1.0",
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
