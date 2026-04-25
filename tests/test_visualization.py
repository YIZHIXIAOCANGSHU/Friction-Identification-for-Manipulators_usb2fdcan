from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import friction_identification_core.visualization as visualization


class _FakeBlueprintNode:
    def __init__(self, kind: str, *children, **kwargs) -> None:
        self.kind = str(kind)
        self.children = list(children)
        self.kwargs = dict(kwargs)


class _FakeTextDocument:
    def __init__(self, text: str, *, media_type: str) -> None:
        self.text = str(text)
        self.media_type = str(media_type)


class _FakeTextLog:
    def __init__(self, text: str) -> None:
        self.text = str(text)


class _FakeScalars:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)


class _FakeRecordingStream:
    instances: list["_FakeRecordingStream"] = []

    def __init__(self, application_id: str) -> None:
        self.application_id = str(application_id)
        self.saved_path: Path | None = None
        self.logs: list[tuple[str, object, bool]] = []
        self.blueprints: list[tuple[object, bool, bool]] = []
        self.times: list[tuple[str, int]] = []
        self.disconnected = False
        type(self).instances.append(self)

    def save(self, path: Path) -> None:
        self.saved_path = Path(path)

    def log(self, path: str, payload: object, static: bool = False) -> None:
        self.logs.append((str(path), payload, bool(static)))

    def send_blueprint(self, blueprint: object, make_active: bool = False, make_default: bool = False) -> None:
        self.blueprints.append((blueprint, bool(make_active), bool(make_default)))

    def set_time(self, timeline: str, *, sequence: int) -> None:
        self.times.append((str(timeline), int(sequence)))

    def disconnect(self) -> None:
        self.disconnected = True


class _FakeBlueprintNamespace:
    @staticmethod
    def Blueprint(*children, **kwargs) -> _FakeBlueprintNode:
        return _FakeBlueprintNode("Blueprint", *children, **kwargs)

    @staticmethod
    def Tabs(*children, **kwargs) -> _FakeBlueprintNode:
        return _FakeBlueprintNode("Tabs", *children, **kwargs)

    @staticmethod
    def Vertical(*children, **kwargs) -> _FakeBlueprintNode:
        return _FakeBlueprintNode("Vertical", *children, **kwargs)

    @staticmethod
    def Horizontal(*children, **kwargs) -> _FakeBlueprintNode:
        return _FakeBlueprintNode("Horizontal", *children, **kwargs)

    @staticmethod
    def TextDocumentView(*children, **kwargs) -> _FakeBlueprintNode:
        return _FakeBlueprintNode("TextDocumentView", *children, **kwargs)

    @staticmethod
    def TextLogView(*children, **kwargs) -> _FakeBlueprintNode:
        return _FakeBlueprintNode("TextLogView", *children, **kwargs)

    @staticmethod
    def TimeSeriesView(*children, **kwargs) -> _FakeBlueprintNode:
        return _FakeBlueprintNode("TimeSeriesView", *children, **kwargs)


def _fake_rr_module() -> SimpleNamespace:
    _FakeRecordingStream.instances.clear()
    return SimpleNamespace(
        RecordingStream=_FakeRecordingStream,
        Scalars=_FakeScalars,
        TextDocument=_FakeTextDocument,
        TextLog=_FakeTextLog,
        blueprint=_FakeBlueprintNamespace(),
    )


def _contains_node_with_origin(node: object, *, kind: str, origin: str) -> bool:
    if isinstance(node, _FakeBlueprintNode):
        if node.kind == kind and node.kwargs.get("origin") == origin:
            return True
        return any(_contains_node_with_origin(child, kind=kind, origin=origin) for child in node.children)
    return False


def _top_level_tab_names(blueprint: object) -> list[str]:
    if not isinstance(blueprint, _FakeBlueprintNode) or not blueprint.children:
        return []
    tabs = blueprint.children[0]
    if not isinstance(tabs, _FakeBlueprintNode) or tabs.kind != "Tabs":
        return []
    return [str(child.kwargs.get("name", "")) for child in tabs.children if isinstance(child, _FakeBlueprintNode)]


def _contains_time_series_content(node: object, *, content: str) -> bool:
    if isinstance(node, _FakeBlueprintNode):
        values = node.kwargs.get("contents", [])
        if node.kind == "TimeSeriesView" and content in values:
            return True
        return any(_contains_time_series_content(child, content=content) for child in node.children)
    return False


class VisualizationTests(unittest.TestCase):
    def test_rerun_feedback_frames_view_and_logging_are_wired(self) -> None:
        fake_rr = _fake_rr_module()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(visualization, "rr", fake_rr):
            recorder = visualization.RerunRecorder(
                Path(tmpdir) / "feedback.rrd",
                motor_ids=(1, 2),
                motor_names={1: "motor_01", 2: "motor_02"},
                mode="step_torque",
                show_viewer=False,
            )
            recording = _FakeRecordingStream.instances[-1]

            self.assertTrue(recording.blueprints)
            blueprint, make_active, make_default = recording.blueprints[-1]
            self.assertTrue(make_active)
            self.assertTrue(make_default)
            self.assertEqual(_top_level_tab_names(blueprint)[0], "Feedback Frames")
            self.assertTrue(
                _contains_node_with_origin(
                    blueprint,
                    kind="TextLogView",
                    origin="/live/feedback_frames",
                )
            )
            self.assertTrue(
                _contains_time_series_content(
                    blueprint,
                    content="/live/motors/motor_01/position",
                )
            )

            recorder.log_live_feedback_frame(
                group_index=1,
                round_index=2,
                active_motor_id=1,
                motor_id=2,
                state=3,
                position=1.25,
                velocity=-0.5,
                feedback_torque=0.75,
                mos_temperature=42.0,
                phase_name="sync_wait",
                stage="step_torque",
            )

            feedback_logs = [entry for entry in recording.logs if entry[0] == "live/feedback_frames"]
            self.assertEqual(len(feedback_logs), 1)
            payload = feedback_logs[0][1]
            self.assertIsInstance(payload, _FakeTextLog)
            self.assertIn("frame 00001", payload.text)
            self.assertIn("stage step_torque", payload.text)
            self.assertIn("phase sync_wait", payload.text)
            self.assertIn("active 01", payload.text)
            self.assertIn("rx 02", payload.text)
            self.assertIn("match OTHER", payload.text)
            self.assertIn("state   3", payload.text)
            self.assertIn("\n", payload.text)
            self.assertIn("pos   +1.250000 rad", payload.text)
            self.assertIn("vel   -0.500000 rad/s", payload.text)
            self.assertIn("torque   +0.750000 Nm", payload.text)

            scalar_logs = [entry for entry in recording.logs if entry[0].startswith("live/motors/motor_02/")]
            logged_paths = {entry[0] for entry in scalar_logs}
            self.assertIn("live/motors/motor_02/position", logged_paths)
            self.assertIn("live/motors/motor_02/velocity", logged_paths)
            self.assertIn("live/motors/motor_02/torque", logged_paths)


if __name__ == "__main__":
    unittest.main()
