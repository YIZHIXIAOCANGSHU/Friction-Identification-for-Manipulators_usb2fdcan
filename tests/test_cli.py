from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from friction_identification_core.__main__ import build_parser, main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SH_PATH = PROJECT_ROOT / "run.sh"
README_PATH = PROJECT_ROOT / "README.md"


class MainCliTests(unittest.TestCase):
    def test_parser_exposes_new_modes(self) -> None:
        parser = build_parser()
        mode_action = next(action for action in parser._actions if action.dest == "mode")
        self.assertEqual(tuple(mode_action.choices), ("identify-all", "compensation", "breakaway", "speed-hold", "inertia"))

    def test_main_routes_modes_to_runners(self) -> None:
        loaded_config = object()
        configured = object()

        with (
            mock.patch("friction_identification_core.__main__.load_config", return_value=loaded_config),
            mock.patch("friction_identification_core.__main__.apply_overrides", return_value=configured),
            mock.patch("friction_identification_core.__main__.run_identify_all") as identify_all_mock,
            mock.patch("friction_identification_core.__main__.run_compensation") as compensation_mock,
            mock.patch("friction_identification_core.__main__.run_breakaway") as breakaway_mock,
            mock.patch("friction_identification_core.__main__.run_speed_hold") as speed_hold_mock,
            mock.patch("friction_identification_core.__main__.run_inertia") as inertia_mock,
        ):
            main([])
            main(["--mode", "compensation"])
            main(["--mode", "breakaway"])
            main(["--mode", "speed-hold"])
            main(["--mode", "inertia"])

        identify_all_mock.assert_called_once_with(configured, show_rerun_viewer=True)
        compensation_mock.assert_called_once_with(configured, show_rerun_viewer=True)
        breakaway_mock.assert_called_once_with(configured, show_rerun_viewer=True)
        speed_hold_mock.assert_called_once_with(configured, show_rerun_viewer=True)
        inertia_mock.assert_called_once_with(configured, show_rerun_viewer=True)

    def test_readme_mentions_identify_all_runtime(self) -> None:
        readme = README_PATH.read_text(encoding="utf-8")
        self.assertIn("identify-all", readme)
        self.assertIn("compensation", readme)
        self.assertIn("0.01 Nm", readme)
        self.assertIn("10 rad/s", readme)
        self.assertNotIn("soft_speed_limit", readme)


class RunShInteractionTests(unittest.TestCase):
    def _make_env_with_fake_python(self) -> dict[str, str]:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        fake_dir = Path(tempdir.name)
        fake_python = fake_dir / "python3"
        fake_python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = f"{fake_dir}:{env.get('PATH', '')}"
        return env

    def _run_script(self, *args: str, input_text: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(RUN_SH_PATH), *args],
            input=input_text,
            text=True,
            capture_output=True,
            cwd=PROJECT_ROOT,
            env=self._make_env_with_fake_python(),
            check=False,
        )

    def _fake_python_args(self, completed: subprocess.CompletedProcess[str]) -> list[str]:
        lines = completed.stdout.strip().splitlines()
        try:
            start = len(lines) - 1 - lines[::-1].index("-m")
        except ValueError:
            return []
        return lines[start:]

    def test_help_prints_usage(self) -> None:
        completed = self._run_script("--help")
        self.assertEqual(completed.returncode, 0)
        self.assertIn("Usage:", completed.stdout)

    def test_interactive_menu_builds_identify_all_command(self) -> None:
        completed = self._run_script(
            input_text="\n".join(
                [
                    "1",
                    "2",
                    "1,3,4",
                    "",
                ]
            ),
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("最终命令", completed.stdout)
        self.assertEqual(
            self._fake_python_args(completed),
            [
                "-m",
                "friction_identification_core",
                "--mode",
                "identify-all",
                "--config",
                "friction_identification_core/default.yaml",
                "--motors",
                "1,3,4",
                "--output",
                "results",
            ],
        )

    def test_interactive_menu_builds_compensation_command(self) -> None:
        completed = self._run_script(
            input_text="\n".join(
                [
                    "2",
                    "3",
                    "",
                ]
            ),
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("最终命令", completed.stdout)
        self.assertEqual(
            self._fake_python_args(completed),
            [
                "-m",
                "friction_identification_core",
                "--mode",
                "compensation",
                "--config",
                "friction_identification_core/default.yaml",
                "--motors",
                "3",
                "--output",
                "results",
            ],
        )

    def test_mode_zero_exits_without_running_python(self) -> None:
        completed = self._run_script(input_text="0\n")
        self.assertEqual(completed.returncode, 0)
        self.assertIn("已退出", completed.stdout)
        self.assertEqual(self._fake_python_args(completed), [])

    def test_legacy_non_interactive_invocation_is_rejected(self) -> None:
        completed = self._run_script("identify-all")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("python3 -m friction_identification_core", completed.stderr)


if __name__ == "__main__":
    unittest.main()
