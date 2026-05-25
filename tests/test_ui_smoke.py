from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402

from mit_sender.settings_store import SettingsStore  # noqa: E402
from mit_sender.ui import MitSenderWindow, create_app  # noqa: E402


class UiSmokeTests(unittest.TestCase):
    def test_main_window_builds_with_default_motor_rows(self) -> None:
        app = create_app()
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsStore(QSettings(str(Path(directory) / "settings.ini"), QSettings.IniFormat))
            window = MitSenderWindow(settings)

            self.assertEqual(len(window.motor_specs), 7)
            self.assertEqual(window.send_button.text(), "一键发送")
            self.assertEqual(window.uniform_send_button.text(), "统一发送")
            self.assertEqual(window.status_label.text(), "就绪")
            self.assertEqual(set(window.checkboxes), {1, 2, 3, 4, 5, 6, 7})

            window.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
