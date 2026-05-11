from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from send.damiao import (
    DEFAULT_DATA_BITRATE,
    DEFAULT_INTERFACE,
    DEFAULT_NOMINAL_BITRATE,
    DamiaoMitController,
    MitCommand,
    MotorFeedback,
    MotorSpec,
    SocketCanTransport,
    configure_can_interface,
    decode_feedback_frame,
    default_motor_specs,
    ensure_interface_ready,
)


MIT_COMMAND_DEFAULTS = {
    "position": "0.0",
    "velocity": "0.0",
    "kp": "0.0",
    "kd": "0.0",
    "torque_ff": "0.0",
}


@dataclass(frozen=True)
class SelectedMotorCommand:
    motor_id: int
    position: float
    velocity: float
    kp: float
    kd: float
    torque_ff: float

    def as_mit_command(self) -> MitCommand:
        return MitCommand(
            position=self.position,
            velocity=self.velocity,
            kp=self.kp,
            kd=self.kd,
            torque_ff=self.torque_ff,
        )


def build_uniform_commands(
    motor_ids: list[int],
    command: SelectedMotorCommand,
) -> list[SelectedMotorCommand]:
    return [
        SelectedMotorCommand(
            motor_id=motor_id,
            position=command.position,
            velocity=command.velocity,
            kp=command.kp,
            kd=command.kd,
            torque_ff=command.torque_ff,
        )
        for motor_id in motor_ids
    ]


@dataclass(frozen=True)
class TransportSettings:
    interface: str
    nominal_bitrate: int
    data_bitrate: int
    configure_interface: bool


def _feedback_rerun_path(motor_id: int) -> str:
    return f"/feedback/motors/motor_{int(motor_id):02d}"


def build_feedback_rerun_blueprint(motor_specs: list[MotorSpec]) -> object | None:
    try:
        import rerun.blueprint as rrb
    except ImportError:
        return None

    motor_views = []
    for spec in motor_specs:
        base_path = _feedback_rerun_path(spec.motor_id)
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
            raise RuntimeError("未安装 rerun-sdk，请先运行: python3 -m pip install rerun-sdk") from exc

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
            base_path = _feedback_rerun_path(spec.motor_id)
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
        base_path = _feedback_rerun_path(feedback.motor_id)
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


class FeedbackMonitorWorker(QObject):
    frame_received = pyqtSignal(object, float, int)
    status_changed = pyqtSignal(str)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, settings: TransportSettings, motor_specs: list[MotorSpec]) -> None:
        super().__init__()
        self.settings = settings
        self.motor_specs = list(motor_specs)
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        transport = None
        logger = None
        try:
            if self.settings.configure_interface:
                configure_can_interface(
                    self.settings.interface,
                    self.settings.nominal_bitrate,
                    self.settings.data_bitrate,
                )
            ensure_interface_ready(
                self.settings.interface,
                self.settings.nominal_bitrate,
                self.settings.data_bitrate,
            )
            logger = FeedbackRerunLogger(self.motor_specs)
            transport = SocketCanTransport(self.settings.interface)
            start_time = time.monotonic()
            self.status_changed.emit("正在读取反馈帧，Rerun 窗口已弹出。")
            while self._running:
                packet = transport.recv(timeout=0.05)
                if packet is None:
                    continue
                feedback = decode_feedback_frame(packet[0], packet[1], self.motor_specs)
                if feedback is None:
                    continue
                elapsed_seconds = time.monotonic() - start_time
                frame_count = logger.log_feedback(feedback, elapsed_seconds)
                self.frame_received.emit(feedback, elapsed_seconds, frame_count)
            self.status_changed.emit("反馈读取已停止。")
        except Exception as exc:  # noqa: BLE001 - show hardware/viewer errors directly.
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail)
        finally:
            if transport is not None:
                transport.close()
            if logger is not None:
                logger.close()
            self.finished.emit()


class FeedbackMonitorDialog(QDialog):
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, motor_specs: list[MotorSpec], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.motor_specs = list(motor_specs)
        self._running = False
        self._frame_counts = {spec.motor_id: 0 for spec in self.motor_specs}
        self.setWindowTitle("反馈帧 Rerun 监视")
        self.resize(860, 360)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        top_layout = QHBoxLayout()
        self.status_label = QLabel("未开始")
        top_layout.addWidget(self.status_label)
        top_layout.addStretch(1)

        self.start_button = QPushButton("开始读取")
        self.start_button.clicked.connect(self.start_requested.emit)
        top_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.stop_button.setEnabled(False)
        top_layout.addWidget(self.stop_button)
        layout.addLayout(top_layout)

        columns = ("电机", "反馈 ID", "状态", "控制器", "position", "velocity", "torque", "MOS", "Rotor", "time", "帧数")
        self.table = QTableWidget(len(self.motor_specs), len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setAlternatingRowColors(True)
        self._row_by_motor_id: dict[int, int] = {}
        for row, spec in enumerate(self.motor_specs):
            self._row_by_motor_id[spec.motor_id] = row
            values = (str(spec.motor_id), f"0x{spec.mst_id:02X}", "-", "-", "-", "-", "-", "-", "-", "-", "0")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, item)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table, stretch=1)

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.start_button.setEnabled(not self._running)
        self.stop_button.setEnabled(self._running)

    def set_status(self, status: str) -> None:
        self.status_label.setText(str(status))

    def update_feedback(self, feedback: MotorFeedback, elapsed_seconds: float, frame_count: int) -> None:
        row = self._row_by_motor_id.get(feedback.motor_id)
        if row is None:
            return
        self._frame_counts[feedback.motor_id] += 1
        values = (
            str(feedback.motor_id),
            f"0x{feedback.can_id:02X}",
            str(feedback.state),
            str(feedback.controller_id),
            f"{feedback.position:+.6f}",
            f"{feedback.velocity:+.6f}",
            f"{feedback.torque:+.6f}",
            f"{feedback.mos_temperature:.1f}",
            f"{feedback.rotor_temperature:.1f}",
            f"{elapsed_seconds:.3f}",
            str(self._frame_counts[feedback.motor_id]),
        )
        for column, value in enumerate(values):
            item = self.table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, item)
            item.setText(value)
        self.set_status(f"正在读取反馈帧，总帧数 {int(frame_count)}")

    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt supplies QCloseEvent at runtime.
        if self._running:
            self.stop_requested.emit()
        super().closeEvent(event)


class MotorActionWorker(QObject):
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        action: str,
        settings: TransportSettings,
        commands: list[SelectedMotorCommand],
    ) -> None:
        super().__init__()
        self.action = action
        self.settings = settings
        self.commands = commands

    def run(self) -> None:
        transport = None
        try:
            if self.settings.configure_interface:
                configure_can_interface(
                    self.settings.interface,
                    self.settings.nominal_bitrate,
                    self.settings.data_bitrate,
                )
            ensure_interface_ready(
                self.settings.interface,
                self.settings.nominal_bitrate,
                self.settings.data_bitrate,
            )
            transport = SocketCanTransport(self.settings.interface)
            controller = DamiaoMitController(transport, default_motor_specs())
            if self.action == "enable":
                for command in self.commands:
                    controller.prepare_motor(command.motor_id)
                message = f"已使能 {len(self.commands)} 个电机。"
            elif self.action == "disable":
                for command in self.commands:
                    controller.disable_motor(command.motor_id)
                message = f"已失能 {len(self.commands)} 个电机。"
            elif self.action in {"send", "send_uniform"}:
                for command in self.commands:
                    controller.prepare_and_send_mit(command.motor_id, command.as_mit_command())
                message = f"已发送 {len(self.commands)} 个 MIT 命令。"
            else:
                raise ValueError(f"unknown action: {self.action}")
            self.finished.emit(True, message)
        except Exception as exc:  # noqa: BLE001 - show hardware/config errors directly.
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.finished.emit(False, detail)
        finally:
            if transport is not None:
                transport.close()


class MitSenderWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.motor_specs = default_motor_specs()
        self.enabled_motor_ids: set[int] = set()
        self.worker_thread: QThread | None = None
        self.worker: MotorActionWorker | None = None
        self.feedback_thread: QThread | None = None
        self.feedback_worker: FeedbackMonitorWorker | None = None
        self.feedback_dialog: FeedbackMonitorDialog | None = None
        self.checkboxes: dict[int, QCheckBox] = {}
        self.inputs: dict[int, dict[str, QLineEdit]] = {}
        self.uniform_inputs: dict[str, QLineEdit] = {}

        self.setWindowTitle("MIT 电机一键发送")
        self.resize(980, 560)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        layout.addWidget(self._build_transport_group())
        layout.addWidget(self._build_uniform_command_group())
        layout.addWidget(self._build_motor_table(), stretch=1)
        layout.addLayout(self._build_actions())

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(200)
        self.log_view.setPlaceholderText("状态输出")
        layout.addWidget(self.log_view)

        self.setCentralWidget(root)

    def _build_transport_group(self) -> QGroupBox:
        group = QGroupBox("CAN 设置")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(10)

        self.interface_input = QLineEdit(DEFAULT_INTERFACE)
        self.interface_input.setMinimumWidth(120)
        self.nominal_input = QLineEdit(str(DEFAULT_NOMINAL_BITRATE))
        self.nominal_input.setValidator(QIntValidator(1, 10_000_000, self))
        self.data_input = QLineEdit(str(DEFAULT_DATA_BITRATE))
        self.data_input.setValidator(QIntValidator(1, 20_000_000, self))
        self.configure_checkbox = QCheckBox("启动前自动配置 can0")

        grid.addWidget(QLabel("接口"), 0, 0)
        grid.addWidget(self.interface_input, 0, 1)
        grid.addWidget(QLabel("仲裁波特率"), 0, 2)
        grid.addWidget(self.nominal_input, 0, 3)
        grid.addWidget(QLabel("数据波特率"), 0, 4)
        grid.addWidget(self.data_input, 0, 5)
        grid.addWidget(self.configure_checkbox, 0, 6)
        grid.setColumnStretch(7, 1)
        return group

    def _build_uniform_command_group(self) -> QGroupBox:
        group = QGroupBox("统一 MIT 指令")
        layout = QHBoxLayout(group)
        layout.setSpacing(8)

        validator = self._mit_value_validator()
        for field, default in MIT_COMMAND_DEFAULTS.items():
            layout.addWidget(QLabel(field))
            line_edit = QLineEdit(default)
            line_edit.setValidator(validator)
            line_edit.setAlignment(Qt.AlignRight)
            line_edit.setFixedWidth(86)
            layout.addWidget(line_edit)
            self.uniform_inputs[field] = line_edit

        layout.addStretch(1)
        self.uniform_send_button = QPushButton("一键统一发送")
        self.uniform_send_button.clicked.connect(self._send_uniform_commands)
        layout.addWidget(self.uniform_send_button)
        return group

    def _build_motor_table(self) -> QTableWidget:
        columns = ("选择", "电机", "CAN ID", "反馈 ID", "型号", "position", "velocity", "kp", "kd", "torque_ff")
        table = QTableWidget(len(self.motor_specs), len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        validator = self._mit_value_validator()

        for row, spec in enumerate(self.motor_specs):
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self._sync_enable_button)
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.addWidget(checkbox)
            table.setCellWidget(row, 0, checkbox_widget)
            self.checkboxes[spec.motor_id] = checkbox

            for column, text in enumerate(
                (
                    f"{spec.motor_id}",
                    f"0x{spec.can_id:02X}",
                    f"0x{spec.mst_id:02X}",
                    spec.motor_type.name,
                ),
                start=1,
            ):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, column, item)

            row_inputs: dict[str, QLineEdit] = {}
            for column, field in enumerate(MIT_COMMAND_DEFAULTS, start=5):
                line_edit = QLineEdit(MIT_COMMAND_DEFAULTS[field])
                line_edit.setValidator(validator)
                line_edit.setAlignment(Qt.AlignRight)
                table.setCellWidget(row, column, line_edit)
                row_inputs[field] = line_edit
            self.inputs[spec.motor_id] = row_inputs

        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _mit_value_validator(self) -> QDoubleValidator:
        validator = QDoubleValidator(-100000.0, 100000.0, 6, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        return validator

    def _build_actions(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addStretch(1)

        self.feedback_button = QPushButton("反馈 Rerun")
        self.feedback_button.clicked.connect(self._start_feedback_monitor)
        layout.addWidget(self.feedback_button)

        self.toggle_enable_button = QPushButton("全部使能")
        self.toggle_enable_button.clicked.connect(self._toggle_enable)
        layout.addWidget(self.toggle_enable_button)

        self.send_button = QPushButton("一键发送")
        self.send_button.clicked.connect(self._send_commands)
        self.send_button.setDefault(True)
        layout.addWidget(self.send_button)
        return layout

    def _settings(self) -> TransportSettings:
        interface = self.interface_input.text().strip()
        if not interface:
            raise ValueError("CAN 接口不能为空")
        return TransportSettings(
            interface=interface,
            nominal_bitrate=int(self.nominal_input.text()),
            data_bitrate=int(self.data_input.text()),
            configure_interface=self.configure_checkbox.isChecked(),
        )

    def _selected_commands(self) -> list[SelectedMotorCommand]:
        commands: list[SelectedMotorCommand] = []
        for spec in self.motor_specs:
            if not self.checkboxes[spec.motor_id].isChecked():
                continue
            fields = self.inputs[spec.motor_id]
            commands.append(
                SelectedMotorCommand(
                    motor_id=spec.motor_id,
                    position=self._read_float(fields["position"], f"motor {spec.motor_id} position"),
                    velocity=self._read_float(fields["velocity"], f"motor {spec.motor_id} velocity"),
                    kp=self._read_float(fields["kp"], f"motor {spec.motor_id} kp"),
                    kd=self._read_float(fields["kd"], f"motor {spec.motor_id} kd"),
                    torque_ff=self._read_float(fields["torque_ff"], f"motor {spec.motor_id} torque_ff"),
                )
            )
        if not commands:
            raise ValueError("请至少勾选一个电机")
        return commands

    def _selected_motor_ids(self) -> list[int]:
        return [
            spec.motor_id
            for spec in self.motor_specs
            if self.checkboxes[spec.motor_id].isChecked()
        ]

    def _uniform_commands(self) -> list[SelectedMotorCommand]:
        selected_ids = self._selected_motor_ids()
        if not selected_ids:
            raise ValueError("请至少勾选一个电机")
        command = SelectedMotorCommand(
            motor_id=0,
            position=self._read_float(self.uniform_inputs["position"], "统一 position"),
            velocity=self._read_float(self.uniform_inputs["velocity"], "统一 velocity"),
            kp=self._read_float(self.uniform_inputs["kp"], "统一 kp"),
            kd=self._read_float(self.uniform_inputs["kd"], "统一 kd"),
            torque_ff=self._read_float(self.uniform_inputs["torque_ff"], "统一 torque_ff"),
        )
        return build_uniform_commands(selected_ids, command)

    def _read_float(self, line_edit: QLineEdit, label: str) -> float:
        text = line_edit.text().strip()
        if not text:
            raise ValueError(f"{label} 不能为空")
        return float(text)

    def _toggle_enable(self) -> None:
        action = "disable" if self._selected_motors_are_enabled() else "enable"
        self._start_action(action)

    def _send_commands(self) -> None:
        self._start_action("send")

    def _send_uniform_commands(self) -> None:
        self._start_action("send_uniform")

    def _show_feedback_monitor(self) -> None:
        if self.feedback_dialog is None:
            self.feedback_dialog = FeedbackMonitorDialog(self.motor_specs, self)
            self.feedback_dialog.start_requested.connect(self._start_feedback_monitor)
            self.feedback_dialog.stop_requested.connect(self._stop_feedback_monitor)
            self.feedback_dialog.destroyed.connect(self._feedback_dialog_destroyed)
        self.feedback_dialog.show()
        self.feedback_dialog.raise_()
        self.feedback_dialog.activateWindow()

    def _start_feedback_monitor(self) -> None:
        if self.feedback_thread is not None:
            return
        try:
            settings = self._settings()
        except Exception as exc:  # noqa: BLE001 - input errors should be visible.
            QMessageBox.warning(self, "输入错误", str(exc))
            return
        self._show_feedback_monitor()
        assert self.feedback_dialog is not None
        self.feedback_dialog.set_running(True)
        self.feedback_dialog.set_status("正在启动 Rerun 反馈读取...")
        self.feedback_button.setEnabled(False)

        self.feedback_thread = QThread(self)
        self.feedback_worker = FeedbackMonitorWorker(settings, self.motor_specs)
        self.feedback_worker.moveToThread(self.feedback_thread)
        self.feedback_thread.started.connect(self.feedback_worker.run)
        self.feedback_worker.frame_received.connect(self._handle_feedback_frame)
        self.feedback_worker.status_changed.connect(self._set_feedback_status)
        self.feedback_worker.failed.connect(self._fail_feedback_monitor)
        self.feedback_worker.finished.connect(self.feedback_thread.quit)
        self.feedback_worker.finished.connect(self.feedback_worker.deleteLater)
        self.feedback_thread.finished.connect(self.feedback_thread.deleteLater)
        self.feedback_thread.finished.connect(self._clear_feedback_worker)
        self.feedback_thread.start()

    def _stop_feedback_monitor(self) -> None:
        if self.feedback_worker is not None:
            self.feedback_worker.stop()
        if self.feedback_dialog is not None:
            self.feedback_dialog.set_status("正在停止反馈读取...")

    def _handle_feedback_frame(self, feedback: MotorFeedback, elapsed_seconds: float, frame_count: int) -> None:
        if self.feedback_dialog is not None:
            self.feedback_dialog.update_feedback(feedback, elapsed_seconds, frame_count)

    def _set_feedback_status(self, status: str) -> None:
        if self.feedback_dialog is not None:
            self.feedback_dialog.set_status(status)
        self._append_log(str(status))

    def _fail_feedback_monitor(self, message: str) -> None:
        if self.feedback_dialog is not None:
            self.feedback_dialog.set_status(f"失败: {message}")
            self.feedback_dialog.set_running(False)
        self._append_log(f"反馈读取失败: {message}")
        QMessageBox.critical(self, "反馈读取失败", message)

    def _clear_feedback_worker(self) -> None:
        self.feedback_thread = None
        self.feedback_worker = None
        self.feedback_button.setEnabled(True)
        if self.feedback_dialog is not None:
            self.feedback_dialog.set_running(False)

    def _feedback_dialog_destroyed(self) -> None:
        self.feedback_dialog = None

    def _start_action(self, action: str) -> None:
        if self.worker_thread is not None:
            return
        try:
            settings = self._settings()
            commands = self._uniform_commands() if action == "send_uniform" else self._selected_commands()
        except Exception as exc:  # noqa: BLE001 - input errors should be visible.
            QMessageBox.warning(self, "输入错误", str(exc))
            return

        self._set_busy(True)
        self._append_log(f"开始: {self._action_label(action)}")
        self.worker_thread = QThread(self)
        self.worker = MotorActionWorker(action, settings, commands)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        command_ids = [command.motor_id for command in commands]
        self.worker.finished.connect(
            lambda ok, message, current_action=action, current_ids=command_ids: self._finish_action(
                current_action,
                current_ids,
                ok,
                message,
            )
        )
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker)
        self.worker_thread.start()

    def _finish_action(self, action: str, motor_ids: list[int], ok: bool, message: str) -> None:
        self._set_busy(False)
        if ok:
            if action == "enable":
                self.enabled_motor_ids.update(motor_ids)
            elif action == "disable":
                self.enabled_motor_ids.difference_update(motor_ids)
            elif action in {"send", "send_uniform"}:
                self.enabled_motor_ids.update(motor_ids)
            self._sync_enable_button()
            self._append_log(f"完成: {message}")
            return
        self._append_log(f"失败: {message}")
        QMessageBox.critical(self, "操作失败", message)

    def _clear_worker(self) -> None:
        self.worker_thread = None
        self.worker = None

    def _set_busy(self, busy: bool) -> None:
        self.toggle_enable_button.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.uniform_send_button.setEnabled(not busy)

    def _sync_enable_button(self) -> None:
        self.toggle_enable_button.setText("全部失能" if self._selected_motors_are_enabled() else "全部使能")

    def _selected_motors_are_enabled(self) -> bool:
        selected_ids = [
            spec.motor_id
            for spec in self.motor_specs
            if self.checkboxes[spec.motor_id].isChecked()
        ]
        return bool(selected_ids) and all(motor_id in self.enabled_motor_ids for motor_id in selected_ids)

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def _action_label(self, action: str) -> str:
        labels = {
            "enable": "全部使能",
            "disable": "全部失能",
            "send": "一键发送",
            "send_uniform": "一键统一发送",
        }
        return labels.get(action, action)

    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt supplies QCloseEvent at runtime.
        self._stop_feedback_monitor()
        if self.feedback_thread is not None:
            self.feedback_thread.wait(1000)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MitSenderWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
