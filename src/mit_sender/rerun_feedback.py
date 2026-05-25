from __future__ import annotations

from mit_sender.damiao import MotorFeedback, MotorSpec


def feedback_rerun_path(motor_id: int) -> str:
    return f"/feedback/motors/motor_{int(motor_id):02d}"


def build_feedback_rerun_blueprint(motor_specs: list[MotorSpec]) -> object | None:
    try:
        import rerun.blueprint as rrb
    except ImportError:
        return None

    motor_views = []
    for spec in motor_specs:
        base_path = feedback_rerun_path(spec.motor_id)
        motor_views.append(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/position"], name="Position"),
                    rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/velocity"], name="Velocity"),
                    name="Motion",
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/torque"], name="Torque"),
                    rrb.TimeSeriesView(
                        origin="/",
                        contents=[f"{base_path}/mos_temperature", f"{base_path}/rotor_temperature"],
                        name="Temperature",
                    ),
                    name="Load",
                ),
                rrb.TextLogView(origin=f"{base_path}/events", name="Frames"),
                name=f"Motor {spec.motor_id}",
            )
        )

    return rrb.Blueprint(
        rrb.Tabs(
            rrb.Vertical(
                rrb.TextDocumentView(origin="/feedback/overview", name="Overview"),
                rrb.TextLogView(origin="/feedback/frames", name="All Frames"),
                name="Overview",
            ),
            rrb.Vertical(rrb.Tabs(*motor_views), name="Motors"),
        ),
        auto_views=False,
    )


class FeedbackRerunLogger:
    def __init__(self, motor_specs: list[MotorSpec]) -> None:
        try:
            import rerun as rr
        except ImportError as exc:
            raise RuntimeError("未安装 rerun-sdk，请先运行: .venv/bin/python -m pip install -e .") from exc

        self._rr = rr
        self._recording = rr.RecordingStream("mit_sender_feedback")
        self._frame_count = 0
        blueprint = build_feedback_rerun_blueprint(motor_specs)
        self._recording.spawn(connect=True, detach_process=True, default_blueprint=blueprint)
        if blueprint is not None:
            self._recording.send_blueprint(blueprint, make_active=True, make_default=True)
        self._recording.log(
            "/feedback/overview",
            rr.TextDocument("等待反馈帧...", media_type="text/plain"),
        )
        for spec in motor_specs:
            base_path = feedback_rerun_path(spec.motor_id)
            self._recording.log(
                f"{base_path}/position",
                rr.SeriesLines(names=["position"], widths=[2.0]),
                static=True,
            )
            self._recording.log(
                f"{base_path}/velocity",
                rr.SeriesLines(names=["velocity"], widths=[2.0]),
                static=True,
            )
            self._recording.log(
                f"{base_path}/torque",
                rr.SeriesLines(names=["torque"], widths=[2.0]),
                static=True,
            )

    def log_feedback(self, feedback: MotorFeedback, elapsed_seconds: float) -> int:
        self._frame_count += 1
        rr = self._rr
        base_path = feedback_rerun_path(feedback.motor_id)
        self._recording.set_time_seconds("feedback_time", float(elapsed_seconds))
        self._recording.log(f"{base_path}/position", rr.Scalars([float(feedback.position)]))
        self._recording.log(f"{base_path}/velocity", rr.Scalars([float(feedback.velocity)]))
        self._recording.log(f"{base_path}/torque", rr.Scalars([float(feedback.torque)]))
        self._recording.log(f"{base_path}/state", rr.Scalars([int(feedback.state)]))
        self._recording.log(f"{base_path}/mos_temperature", rr.Scalars([float(feedback.mos_temperature)]))
        self._recording.log(f"{base_path}/rotor_temperature", rr.Scalars([float(feedback.rotor_temperature)]))
        text = (
            f"#{self._frame_count:06d} motor={feedback.motor_id} can_id=0x{feedback.can_id:03X} "
            f"state={feedback.state} controller={feedback.controller_id} "
            f"pos={feedback.position:+.6f} vel={feedback.velocity:+.6f} "
            f"tau={feedback.torque:+.6f} mos={feedback.mos_temperature:.1f} "
            f"rotor={feedback.rotor_temperature:.1f}"
        )
        self._recording.log("/feedback/frames", rr.TextLog(text, level="INFO"))
        self._recording.log(f"{base_path}/events", rr.TextLog(text, level="INFO"))
        self._recording.log(
            "/feedback/overview",
            rr.TextDocument(
                "\n".join(
                    [
                        "MIT 电机反馈读取中",
                        f"frames={self._frame_count}",
                        f"latest_motor={feedback.motor_id}",
                        f"elapsed_s={float(elapsed_seconds):.3f}",
                    ]
                ),
                media_type="text/plain",
            ),
        )
        return self._frame_count

    def close(self) -> None:
        disconnect = getattr(self._recording, "disconnect", None)
        if callable(disconnect):
            disconnect()
