from __future__ import annotations

import unittest

from mit_sender.rerun_feedback import _set_rerun_time_seconds


class FakeRecordingWithSetTime:
    def __init__(self) -> None:
        self.calls = []

    def set_time(self, timeline: str, *, duration: float) -> None:
        self.calls.append((timeline, duration))


class FakeRecordingWithSetTimeSeconds:
    def __init__(self) -> None:
        self.calls = []

    def set_time_seconds(self, timeline: str, seconds: float) -> None:
        self.calls.append((timeline, seconds))


class RerunFeedbackTests(unittest.TestCase):
    def test_set_rerun_time_uses_current_set_time_api(self) -> None:
        recording = FakeRecordingWithSetTime()

        _set_rerun_time_seconds(recording, "feedback_time", 1.25)

        self.assertEqual(recording.calls, [("feedback_time", 1.25)])

    def test_set_rerun_time_falls_back_to_old_set_time_seconds_api(self) -> None:
        recording = FakeRecordingWithSetTimeSeconds()

        _set_rerun_time_seconds(recording, "feedback_time", 2.5)

        self.assertEqual(recording.calls, [("feedback_time", 2.5)])


if __name__ == "__main__":
    unittest.main()
