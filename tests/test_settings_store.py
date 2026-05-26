from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PySide6.QtCore import QSettings

from mit_sender.commands import SingleMotorDebugCommand, TransportSettings
from mit_sender.damiao import DEFAULT_DATA_BITRATE, DEFAULT_INTERFACE, DEFAULT_NOMINAL_BITRATE, default_motor_specs
from mit_sender.settings_store import SettingsStore


class SettingsStoreTests(unittest.TestCase):
    def test_saves_and_restores_last_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.ini"
            store = SettingsStore(QSettings(str(path), QSettings.IniFormat))

            store.save(
                transport=TransportSettings("can1", 500000, 2000000, True),
                selected_motor_ids={1, 3},
                motor_commands={
                    1: {"position": "1.1", "velocity": "2.2", "kp": "3.3", "kd": "0.4", "torque_ff": "5.5"},
                    3: {"position": "6.6", "velocity": "7.7", "kp": "8.8", "kd": "0.9", "torque_ff": "1.0"},
                },
                uniform_command={"position": "0.1", "velocity": "0.2", "kp": "1.0", "kd": "0.3", "torque_ff": "0.4"},
                window_geometry=b"window",
                feedback_geometry=b"feedback",
                debug_command=SingleMotorDebugCommand(
                    current_can_id=0x02,
                    current_mode_offset=0x100,
                    new_can_id=0x05,
                    new_mst_id=0x21,
                ),
            )

            restored = SettingsStore(QSettings(str(path), QSettings.IniFormat)).load(default_motor_specs())

        self.assertEqual(restored.transport, TransportSettings("can1", 500000, 2000000, True))
        self.assertEqual(restored.selected_motor_ids, {1, 3})
        self.assertEqual(restored.motor_commands[1]["position"], "1.1")
        self.assertEqual(restored.motor_commands[3]["torque_ff"], "1.0")
        self.assertEqual(restored.uniform_command["kp"], "1.0")
        self.assertEqual(restored.debug_command, SingleMotorDebugCommand(0x02, 0x100, 0x05, 0x21))

    def test_invalid_values_fall_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.ini"
            settings = QSettings(str(path), QSettings.IniFormat)
            settings.setValue("transport/interface", "")
            settings.setValue("transport/nominal_bitrate", "bad")
            settings.setValue("transport/data_bitrate", "-1")
            settings.setValue("uniform_command/position", "not-a-float")
            settings.setValue("debug/current_can_id", "0x7FF")
            settings.setValue("debug/current_mode_offset", "123")
            settings.setValue("debug/new_can_id", "bad")
            settings.setValue("debug/new_mst_id", "0")
            settings.sync()

            restored = SettingsStore(QSettings(str(path), QSettings.IniFormat)).load(default_motor_specs())

        self.assertEqual(restored.transport.interface, DEFAULT_INTERFACE)
        self.assertEqual(restored.transport.nominal_bitrate, DEFAULT_NOMINAL_BITRATE)
        self.assertEqual(restored.transport.data_bitrate, DEFAULT_DATA_BITRATE)
        self.assertEqual(restored.uniform_command["position"], "0.0")
        self.assertEqual(restored.debug_command, SingleMotorDebugCommand(0x01, 0x000, 0x01, 0x11))

    def test_debug_feedback_id_defaults_to_target_can_plus_offset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.ini"
            settings = QSettings(str(path), QSettings.IniFormat)
            settings.setValue("debug/current_can_id", "02")
            settings.setValue("debug/new_can_id", "0x03")
            settings.sync()

            restored = SettingsStore(QSettings(str(path), QSettings.IniFormat)).load(default_motor_specs())

        self.assertEqual(restored.debug_command.current_can_id, 0x02)
        self.assertEqual(restored.debug_command.new_can_id, 0x03)
        self.assertEqual(restored.debug_command.new_mst_id, 0x13)

    def test_invalid_transport_interface_falls_back_to_can0(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.ini"
            settings = QSettings(str(path), QSettings.IniFormat)
            settings.setValue("transport/interface", "can2")
            settings.sync()

            restored = SettingsStore(QSettings(str(path), QSettings.IniFormat)).load(default_motor_specs())

        self.assertEqual(restored.transport.interface, "can0")

    def test_restores_manual_debug_feedback_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.ini"
            settings = QSettings(str(path), QSettings.IniFormat)
            settings.setValue("debug/new_can_id", "0x03")
            settings.setValue("debug/new_mst_id", "0x21")
            settings.sync()

            restored = SettingsStore(QSettings(str(path), QSettings.IniFormat)).load(default_motor_specs())

        self.assertEqual(restored.debug_command.new_can_id, 0x03)
        self.assertEqual(restored.debug_command.new_mst_id, 0x21)


if __name__ == "__main__":
    unittest.main()
