from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

try:
    import rerun as rr
except ImportError:  # pragma: no cover
    rr = None


class RerunRecorder:
    def __init__(
        self,
        recording_path: Path,
        *,
        motor_ids: tuple[int, ...],
        motor_names: Mapping[int, str],
        mode: str = "identify-all",
        show_viewer: bool = False,
    ) -> None:
        self.recording_path = Path(recording_path)
        self._motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
        self._motor_names = {int(motor_id): str(motor_names[motor_id]) for motor_id in self._motor_ids}
        self._mode = str(mode)
        self._sequence = 0
        self._feedback_frame_count = 0
        self._latest_raw_command_packet = b""
        self._recording = None if rr is None else rr.RecordingStream("friction_identification")
        if self._recording is None:
            return
        self._recording.save(self.recording_path)
        if show_viewer:
            spawn = getattr(self._recording, "spawn", None)
            if callable(spawn):
                try:
                    spawn(connect=True, detach_process=True)
                except Exception:
                    pass
        self._send_default_blueprint()
        self._log_text("live/overview/current_state", f"mode={self._mode}")

    def _send_default_blueprint(self) -> None:
        if self._recording is None:
            return
        rrb = rr.blueprint
        motor_views = [
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin="/",
                    contents=[f"{self._motor_series_path(motor_id)}/position"],
                    name=f"{self._motor_names[motor_id]} Position",
                ),
                rrb.TimeSeriesView(
                    origin="/",
                    contents=[f"{self._motor_series_path(motor_id)}/velocity"],
                    name=f"{self._motor_names[motor_id]} Velocity",
                ),
                rrb.TimeSeriesView(
                    origin="/",
                    contents=[f"{self._motor_series_path(motor_id)}/torque"],
                    name=f"{self._motor_names[motor_id]} Torque",
                ),
                name=f"{self._motor_names[motor_id]} Signals",
            )
            for motor_id in self._motor_ids
        ]
        blueprint = rrb.Blueprint(
            rrb.Tabs(
                rrb.Vertical(
                    rrb.TextLogView(origin="/live/feedback_frames", name="Feedback Frames"),
                    name="Feedback Frames",
                ),
                rrb.Vertical(
                    rrb.TextDocumentView(origin="/live/overview/current_state", name="Overview"),
                    rrb.TextDocumentView(origin="/live/overview/raw_command_packet", name="Raw Packet"),
                    name="Overview",
                ),
                rrb.Vertical(
                    *motor_views,
                    name="Motor Signals",
                ),
            ),
            auto_views=False,
        )
        self._recording.send_blueprint(blueprint, make_active=True, make_default=True)

    def _set_time(self) -> None:
        if self._recording is None:
            return
        self._recording.set_time("sample", sequence=int(self._sequence))
        self._sequence += 1

    def _log_text(self, path: str, text: str) -> None:
        if self._recording is None:
            return
        self._recording.log(path, rr.TextDocument(str(text), media_type="text/plain"))

    @staticmethod
    def _motor_series_path(motor_id: int) -> str:
        return f"/live/motors/motor_{int(motor_id):02d}"

    def log_round_timing(
        self,
        *,
        group_index: int,
        round_index: int,
        active_motor_id: int,
        planned_duration_s: float,
        actual_capture_duration_s: float,
        sync_wait_duration_s: float,
        round_total_duration_s: float,
    ) -> None:
        self._log_text(
            "live/overview/current_state",
            "\n".join(
                [
                    f"mode={self._mode}",
                    f"group_index={int(group_index)}",
                    f"round_index={int(round_index)}",
                    f"active_motor_id={int(active_motor_id)}",
                    f"planned_duration_s={float(planned_duration_s):.6f}",
                    f"actual_capture_duration_s={float(actual_capture_duration_s):.6f}",
                    f"sync_wait_duration_s={float(sync_wait_duration_s):.6f}",
                    f"round_total_duration_s={float(round_total_duration_s):.6f}",
                ]
            ),
        )

    def log_live_command_packet(
        self,
        *,
        sent_commands: np.ndarray,
        expected_positions: np.ndarray,
        expected_velocities: np.ndarray,
        raw_packet: bytes | None = None,
    ) -> None:
        _ = sent_commands, expected_positions, expected_velocities
        if raw_packet is not None:
            self._latest_raw_command_packet = bytes(raw_packet)
        packet = self._latest_raw_command_packet
        text = packet.hex(" ").upper() if packet else "-"
        self._log_text("live/overview/raw_command_packet", text)

    def _feedback_frame_log_text(
        self,
        *,
        group_index: int,
        round_index: int,
        active_motor_id: int,
        motor_id: int,
        state: int,
        position: float,
        velocity: float,
        feedback_torque: float,
        mos_temperature: float,
        phase_name: str,
        stage: str,
    ) -> str:
        target_marker = "TARGET" if int(motor_id) == int(active_motor_id) else "OTHER "
        return "\n".join(
            [
                (
                    f"frame {self._feedback_frame_count:05d}    "
                    f"stage {str(stage):<12}    "
                    f"phase {str(phase_name):<16}"
                ),
                (
                    f"group {int(group_index):02d}       "
                    f"round {int(round_index):02d}           "
                    f"active {int(active_motor_id):02d}       "
                    f"rx {int(motor_id):02d}       "
                    f"match {target_marker}    "
                    f"state {int(state):3d}"
                ),
                (
                    f"pos {float(position):+11.6f} rad    "
                    f"vel {float(velocity):+11.6f} rad/s"
                ),
                (
                    f"torque {float(feedback_torque):+11.6f} Nm     "
                    f"temp {float(mos_temperature):8.3f} C"
                ),
            ]
        )

    def log_live_feedback_frame(
        self,
        *,
        group_index: int,
        round_index: int,
        active_motor_id: int,
        motor_id: int,
        state: int,
        position: float,
        velocity: float,
        feedback_torque: float,
        mos_temperature: float,
        phase_name: str,
        stage: str,
    ) -> None:
        if self._recording is None:
            return
        self._feedback_frame_count += 1
        self._set_time()
        self._recording.log(
            "live/feedback_frames",
            rr.TextLog(
                self._feedback_frame_log_text(
                    group_index=int(group_index),
                    round_index=int(round_index),
                    active_motor_id=int(active_motor_id),
                    motor_id=int(motor_id),
                    state=int(state),
                    position=float(position),
                    velocity=float(velocity),
                    feedback_torque=float(feedback_torque),
                    mos_temperature=float(mos_temperature),
                    phase_name=str(phase_name),
                    stage=str(stage),
                )
            ),
        )
        self._recording.log(f"live/motors/motor_{int(motor_id):02d}/position", rr.Scalars([float(position)]))
        self._recording.log(f"live/motors/motor_{int(motor_id):02d}/velocity", rr.Scalars([float(velocity)]))
        self._recording.log(f"live/motors/motor_{int(motor_id):02d}/torque", rr.Scalars([float(feedback_torque)]))

    def log_live_motor_sample(
        self,
        *,
        group_index: int,
        round_index: int,
        active_motor_id: int,
        motor_id: int,
        position: float,
        velocity: float,
        feedback_torque: float,
        command_raw: float,
        command: float,
        reference_position: float,
        reference_velocity: float,
        reference_acceleration: float,
        velocity_limit: float,
        torque_limit: float,
        position_limit: float,
        phase_name: str,
        stage: str,
        safety_margin_text: str,
    ) -> None:
        _ = (
            group_index,
            round_index,
            active_motor_id,
            motor_id,
            position,
            velocity,
            feedback_torque,
            command_raw,
            command,
            reference_position,
            reference_velocity,
            reference_acceleration,
            velocity_limit,
            torque_limit,
            position_limit,
            phase_name,
            stage,
            safety_margin_text,
        )

    def log_phase_event(self, *, motor_id: int, phase_name: str, detail: str = "") -> None:
        if self._recording is None:
            return
        self._recording.log(
            f"live/events/motor_{int(motor_id):02d}",
            rr.TextLog(f"{phase_name}: {detail}".strip()),
        )

    def log_abort_event(self, payload: dict[str, object]) -> None:
        self._log_text("live/overview/abort_event", str(payload))

    def log_round_stop(
        self,
        *,
        group_index: int,
        round_index: int,
        motor_id: int,
        phase_name: str,
        stage: str,
    ) -> None:
        self._log_text(
            "live/overview/current_state",
            f"group={int(group_index)} round={int(round_index)} motor={int(motor_id)} stage={stage} phase={phase_name}",
        )

    def log_summary(self, *, summary_path: Path, report_path: Path) -> None:
        self._log_text("summary/static", f"summary={summary_path}\nreport={report_path}")

    def close(self) -> None:
        if self._recording is None:
            return
        disconnect = getattr(self._recording, "disconnect", None)
        if callable(disconnect):
            disconnect()


__all__ = ["RerunRecorder"]
