from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFrame,
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

from mit_sender.commands import (
    MIT_COMMAND_DEFAULTS,
    MIT_FIELD_LABELS,
    MIT_FIELD_TOOLTIPS,
    SelectedMotorCommand,
    TransportSettings,
    build_uniform_commands,
)
from mit_sender.damiao import (
    DEFAULT_DATA_BITRATE,
    DEFAULT_INTERFACE,
    DEFAULT_NOMINAL_BITRATE,
    MotorFeedback,
    MotorSpec,
    default_motor_specs,
)
from mit_sender.settings_store import SavedAppState, SettingsStore
from mit_sender.workers import FeedbackMonitorWorker, MotorActionWorker


STYLESHEET = """
QMainWindow, QDialog {
    background: #F7F9FC;
    color: #172033;
}
QWidget {
    font-size: 14px;
}
QGroupBox {
    background: #FFFFFF;
    border: 1px solid #D8E0EA;
    border-radius: 8px;
    margin-top: 18px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #344155;
}
QLineEdit {
    background: #FFFFFF;
    border: 1px solid #C9D3E1;
    border-radius: 6px;
    min-height: 30px;
    padding: 2px 8px;
    selection-background-color: #2563EB;
}
QLineEdit:focus {
    border: 1px solid #2563EB;
}
QLineEdit[invalid="true"] {
    border: 1px solid #B91C1C;
    background: #FFF1F2;
}
QPushButton {
    background: #FFFFFF;
    border: 1px solid #C9D3E1;
    border-radius: 6px;
    min-height: 32px;
    padding: 4px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: #EFF6FF;
    border-color: #93B4F5;
}
QPushButton:pressed {
    background: #DBEAFE;
}
QPushButton:disabled {
    color: #8A96A8;
    background: #EDF1F6;
}
QPushButton#primaryButton {
    background: #2563EB;
    border-color: #2563EB;
    color: #FFFFFF;
}
QPushButton#primaryButton:hover {
    background: #1D4ED8;
}
QPushButton#dangerButton {
    color: #B91C1C;
}
QCheckBox {
    spacing: 8px;
}
QTableWidget {
    background: #FFFFFF;
    alternate-background-color: #F3F6FA;
    border: 1px solid #D8E0EA;
    border-radius: 8px;
    gridline-color: #E5EAF1;
    selection-background-color: #DBEAFE;
    selection-color: #172033;
}
QHeaderView::section {
    background: #EDF2F7;
    color: #344155;
    border: 0;
    border-right: 1px solid #D8E0EA;
    border-bottom: 1px solid #D8E0EA;
    min-height: 30px;
    padding: 4px 6px;
    font-weight: 700;
}
QPlainTextEdit {
    background: #FFFFFF;
    border: 1px solid #D8E0EA;
    border-radius: 8px;
    color: #172033;
    font-family: "DejaVu Sans Mono", "Consolas", monospace;
    font-size: 13px;
    padding: 8px;
}
QLabel#titleLabel {
    font-size: 20px;
    font-weight: 800;
    color: #172033;
}
QLabel#subtitleLabel {
    color: #64748B;
}
QLabel#statusPill {
    background: #EAF2FF;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    color: #1E40AF;
    padding: 5px 10px;
    font-weight: 600;
}
QFrame#panel {
    background: #FFFFFF;
    border: 1px solid #D8E0EA;
    border-radius: 8px;
}
"""


class FeedbackMonitorDialog(QDialog):
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(
        self,
        motor_specs: list[MotorSpec],
        settings_store: SettingsStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.motor_specs = list(motor_specs)
        self.settings_store = settings_store
        self._running = False
        self._frame_counts = {spec.motor_id: 0 for spec in self.motor_specs}
        self.setWindowTitle("反馈帧 Rerun 监视")
        self.resize(980, 420)
        self._build_ui()

    def restore_saved_geometry(self, state: SavedAppState) -> None:
        if state.feedback_geometry:
            self.restoreGeometry(state.feedback_geometry)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title_area = QVBoxLayout()
        title = QLabel("反馈监视")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Rerun viewer 会同步显示 0x11 到 0x17 的反馈帧")
        subtitle.setObjectName("subtitleLabel")
        title_area.addWidget(title)
        title_area.addWidget(subtitle)
        header.addLayout(title_area)
        header.addStretch(1)

        self.status_label = QLabel("未开始")
        self.status_label.setObjectName("statusPill")
        header.addWidget(self.status_label)

        self.start_button = QPushButton("开始读取")
        self.start_button.clicked.connect(self.start_requested.emit)
        header.addWidget(self.start_button)

        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.stop_button.setEnabled(False)
        header.addWidget(self.stop_button)
        layout.addLayout(header)

        columns = ("电机", "反馈 ID", "状态", "控制器", "position", "velocity", "torque", "MOS", "Rotor", "time", "帧数")
        self.table = QTableWidget(len(self.motor_specs), len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        self._row_by_motor_id: dict[int, int] = {}
        for row, spec in enumerate(self.motor_specs):
            self.table.setRowHeight(row, 34)
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
        self.set_status(f"读取中 · 总帧数 {int(frame_count)}")

    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt supplies QCloseEvent at runtime.
        self.settings_store.settings.setValue("feedback/geometry", self.saveGeometry())
        if self._running:
            self.stop_requested.emit()
        super().closeEvent(event)


class MitSenderWindow(QMainWindow):
    def __init__(self, settings_store: SettingsStore | None = None) -> None:
        super().__init__()
        self.settings_store = settings_store or SettingsStore()
        self.motor_specs = default_motor_specs()
        self.saved_state = self.settings_store.load(self.motor_specs)
        self.enabled_motor_ids: set[int] = set()
        self.worker_thread: QThread | None = None
        self.worker: MotorActionWorker | None = None
        self.feedback_thread: QThread | None = None
        self.feedback_worker: FeedbackMonitorWorker | None = None
        self.feedback_dialog: FeedbackMonitorDialog | None = None
        self.checkboxes: dict[int, QCheckBox] = {}
        self.inputs: dict[int, dict[str, QLineEdit]] = {}
        self.uniform_inputs: dict[str, QLineEdit] = {}
        self._loading_state = True

        self.setWindowTitle("MIT 电机一键发送")
        self.resize(1120, 720)
        self._build_ui()
        self._apply_saved_state()
        self._loading_state = False

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addLayout(self._build_header())
        layout.addWidget(self._build_transport_group())
        layout.addWidget(self._build_uniform_command_group())
        layout.addWidget(self._build_motor_table(), stretch=1)
        layout.addWidget(self._build_actions_panel())

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setPlaceholderText("状态输出")
        self.log_view.setFixedHeight(128)
        layout.addWidget(self.log_view)

        self.setCentralWidget(root)

    def _build_header(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        title_area = QVBoxLayout()
        title = QLabel("MIT 电机一键发送")
        title.setObjectName("titleLabel")
        subtitle = QLabel("达妙电机 SocketCAN 控制 · 参数编辑 · Rerun 反馈监视")
        subtitle.setObjectName("subtitleLabel")
        title_area.addWidget(title)
        title_area.addWidget(subtitle)
        layout.addLayout(title_area)
        layout.addStretch(1)
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusPill")
        layout.addWidget(self.status_label)
        return layout

    def _build_transport_group(self) -> QGroupBox:
        group = QGroupBox("CAN 设置")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.interface_input = QLineEdit(DEFAULT_INTERFACE)
        self.interface_input.setMinimumWidth(120)
        self.interface_input.setToolTip("SocketCAN 接口，例如 can0")
        self.nominal_input = QLineEdit(str(DEFAULT_NOMINAL_BITRATE))
        self.nominal_input.setValidator(QIntValidator(1, 10_000_000, self))
        self.nominal_input.setToolTip("仲裁波特率，默认 1000000")
        self.data_input = QLineEdit(str(DEFAULT_DATA_BITRATE))
        self.data_input.setValidator(QIntValidator(1, 20_000_000, self))
        self.data_input.setToolTip("CAN FD 数据波特率，默认 5000000")
        self.configure_checkbox = QCheckBox("启动前自动配置接口")
        self.configure_checkbox.setToolTip("执行 ip link set ... fd on；可能需要权限")

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
            label = QLabel(MIT_FIELD_LABELS[field])
            label.setToolTip(MIT_FIELD_TOOLTIPS[field])
            layout.addWidget(label)
            line_edit = QLineEdit(default)
            line_edit.setValidator(validator)
            line_edit.setAlignment(Qt.AlignRight)
            line_edit.setFixedWidth(92)
            line_edit.setToolTip(MIT_FIELD_TOOLTIPS[field])
            line_edit.textChanged.connect(lambda _text, widget=line_edit: self._mark_valid(widget))
            layout.addWidget(line_edit)
            self.uniform_inputs[field] = line_edit

        layout.addStretch(1)
        self.uniform_send_button = QPushButton("统一发送")
        self.uniform_send_button.clicked.connect(self._send_uniform_commands)
        layout.addWidget(self.uniform_send_button)
        return group

    def _build_motor_table(self) -> QTableWidget:
        columns = ("选择", "电机", "CAN ID", "反馈 ID", "型号", "P", "V", "Kp", "Kd", "Tau")
        table = QTableWidget(len(self.motor_specs), len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setShowGrid(True)

        for index, field in enumerate(MIT_COMMAND_DEFAULTS, start=5):
            item = table.horizontalHeaderItem(index)
            if item is not None:
                item.setToolTip(MIT_FIELD_TOOLTIPS[field])

        validator = self._mit_value_validator()

        for row, spec in enumerate(self.motor_specs):
            table.setRowHeight(row, 38)
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self._sync_enable_button)
            checkbox.stateChanged.connect(lambda _state: self._save_state())
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
                line_edit.setToolTip(MIT_FIELD_TOOLTIPS[field])
                line_edit.textChanged.connect(lambda _text, widget=line_edit: self._mark_valid(widget))
                table.setCellWidget(row, column, line_edit)
                row_inputs[field] = line_edit
            self.inputs[spec.motor_id] = row_inputs

        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        return table

    def _build_actions_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self.feedback_button = QPushButton("反馈 Rerun")
        self.feedback_button.clicked.connect(self._start_feedback_monitor)
        layout.addWidget(self.feedback_button)

        layout.addStretch(1)

        self.toggle_enable_button = QPushButton("全部使能")
        self.toggle_enable_button.clicked.connect(self._toggle_enable)
        layout.addWidget(self.toggle_enable_button)

        self.send_button = QPushButton("一键发送")
        self.send_button.setObjectName("primaryButton")
        self.send_button.clicked.connect(self._send_commands)
        self.send_button.setDefault(True)
        layout.addWidget(self.send_button)
        return panel

    def _apply_saved_state(self) -> None:
        state = self.saved_state
        self.interface_input.setText(state.transport.interface)
        self.nominal_input.setText(str(state.transport.nominal_bitrate))
        self.data_input.setText(str(state.transport.data_bitrate))
        self.configure_checkbox.setChecked(state.transport.configure_interface)

        for spec in self.motor_specs:
            self.checkboxes[spec.motor_id].setChecked(spec.motor_id in state.selected_motor_ids)
            command = state.motor_commands.get(spec.motor_id, MIT_COMMAND_DEFAULTS)
            for field, line_edit in self.inputs[spec.motor_id].items():
                line_edit.setText(str(command.get(field, MIT_COMMAND_DEFAULTS[field])))

        for field, line_edit in self.uniform_inputs.items():
            line_edit.setText(str(state.uniform_command.get(field, MIT_COMMAND_DEFAULTS[field])))

        if state.window_geometry:
            self.restoreGeometry(state.window_geometry)
        self._sync_enable_button()
        self._append_log("已恢复上次界面输入。")

    def _mit_value_validator(self) -> QDoubleValidator:
        validator = QDoubleValidator(-100000.0, 100000.0, 6, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        return validator

    def _settings(self) -> TransportSettings:
        self._mark_valid(self.interface_input)
        self._mark_valid(self.nominal_input)
        self._mark_valid(self.data_input)
        interface = self.interface_input.text().strip()
        if not interface:
            self._mark_invalid(self.interface_input)
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
                    position=self._read_float(fields["position"], f"电机 {spec.motor_id} P"),
                    velocity=self._read_float(fields["velocity"], f"电机 {spec.motor_id} V"),
                    kp=self._read_float(fields["kp"], f"电机 {spec.motor_id} Kp"),
                    kd=self._read_float(fields["kd"], f"电机 {spec.motor_id} Kd"),
                    torque_ff=self._read_float(fields["torque_ff"], f"电机 {spec.motor_id} Tau"),
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
            position=self._read_float(self.uniform_inputs["position"], "统一 P"),
            velocity=self._read_float(self.uniform_inputs["velocity"], "统一 V"),
            kp=self._read_float(self.uniform_inputs["kp"], "统一 Kp"),
            kd=self._read_float(self.uniform_inputs["kd"], "统一 Kd"),
            torque_ff=self._read_float(self.uniform_inputs["torque_ff"], "统一 Tau"),
        )
        return build_uniform_commands(selected_ids, command)

    def _read_float(self, line_edit: QLineEdit, label: str) -> float:
        text = line_edit.text().strip()
        if not text:
            self._mark_invalid(line_edit)
            raise ValueError(f"{label} 不能为空")
        try:
            value = float(text)
        except ValueError as exc:
            self._mark_invalid(line_edit)
            raise ValueError(f"{label} 必须是数字") from exc
        self._mark_valid(line_edit)
        return value

    def _toggle_enable(self) -> None:
        action = "disable" if self._selected_motors_are_enabled() else "enable"
        self._start_action(action)

    def _send_commands(self) -> None:
        self._start_action("send")

    def _send_uniform_commands(self) -> None:
        self._start_action("send_uniform")

    def _show_feedback_monitor(self) -> None:
        if self.feedback_dialog is None:
            self.feedback_dialog = FeedbackMonitorDialog(self.motor_specs, self.settings_store, self)
            self.feedback_dialog.restore_saved_geometry(self.saved_state)
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
        self._save_state()
        self._show_feedback_monitor()
        assert self.feedback_dialog is not None
        self.feedback_dialog.set_running(True)
        self.feedback_dialog.set_status("正在启动 Rerun...")
        self.feedback_button.setEnabled(False)
        self._set_status("反馈读取启动中")

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
            self.feedback_dialog.set_status("正在停止...")

    def _handle_feedback_frame(self, feedback: MotorFeedback, elapsed_seconds: float, frame_count: int) -> None:
        if self.feedback_dialog is not None:
            self.feedback_dialog.update_feedback(feedback, elapsed_seconds, frame_count)

    def _set_feedback_status(self, status: str) -> None:
        if self.feedback_dialog is not None:
            self.feedback_dialog.set_status(status)
        self._set_status(status)
        self._append_log(str(status))

    def _fail_feedback_monitor(self, message: str) -> None:
        if self.feedback_dialog is not None:
            self.feedback_dialog.set_status(f"失败: {message}")
            self.feedback_dialog.set_running(False)
        self._set_status("反馈读取失败")
        self._append_log(f"反馈读取失败: {message}")
        QMessageBox.critical(self, "反馈读取失败", message)

    def _clear_feedback_worker(self) -> None:
        self.feedback_thread = None
        self.feedback_worker = None
        self.feedback_button.setEnabled(True)
        self._set_status("就绪")
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

        self._save_state()
        self._set_busy(True)
        self._set_status(f"执行中: {self._action_label(action)}")
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
            self._set_status("完成")
            self._append_log(f"完成: {message}")
            return
        self._set_status("操作失败")
        self._append_log(f"失败: {message}")
        QMessageBox.critical(self, "操作失败", message)

    def _clear_worker(self) -> None:
        self.worker_thread = None
        self.worker = None

    def _set_busy(self, busy: bool) -> None:
        self.toggle_enable_button.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.uniform_send_button.setEnabled(not busy)
        self.feedback_button.setEnabled(not busy and self.feedback_thread is None)

    def _sync_enable_button(self) -> None:
        self.toggle_enable_button.setText("全部失能" if self._selected_motors_are_enabled() else "全部使能")

    def _selected_motors_are_enabled(self) -> bool:
        selected_ids = [
            spec.motor_id
            for spec in self.motor_specs
            if self.checkboxes[spec.motor_id].isChecked()
        ]
        return bool(selected_ids) and all(motor_id in self.enabled_motor_ids for motor_id in selected_ids)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(str(message))

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")

    def _action_label(self, action: str) -> str:
        labels = {
            "enable": "全部使能",
            "disable": "全部失能",
            "send": "一键发送",
            "send_uniform": "统一发送",
        }
        return labels.get(action, action)

    def _mark_invalid(self, line_edit: QLineEdit) -> None:
        line_edit.setProperty("invalid", True)
        line_edit.style().unpolish(line_edit)
        line_edit.style().polish(line_edit)

    def _mark_valid(self, line_edit: QLineEdit) -> None:
        line_edit.setProperty("invalid", False)
        line_edit.style().unpolish(line_edit)
        line_edit.style().polish(line_edit)

    def _save_state(self) -> None:
        if self._loading_state:
            return
        try:
            transport = self._settings()
        except Exception:
            transport = self.saved_state.transport
        self.settings_store.save(
            transport=transport,
            selected_motor_ids=set(self._selected_motor_ids()),
            motor_commands={
                motor_id: {
                    field: line_edit.text().strip()
                    for field, line_edit in fields.items()
                }
                for motor_id, fields in self.inputs.items()
            },
            uniform_command={
                field: line_edit.text().strip()
                for field, line_edit in self.uniform_inputs.items()
            },
            window_geometry=self.saveGeometry(),
            feedback_geometry=self.feedback_dialog.saveGeometry() if self.feedback_dialog is not None else None,
        )

    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt supplies QCloseEvent at runtime.
        self._save_state()
        self._stop_feedback_monitor()
        if self.feedback_thread is not None:
            self.feedback_thread.wait(1000)
        super().closeEvent(event)


def create_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setApplicationName("DamiaoMitSender")
    app.setOrganizationName("MitSender")
    app.setStyleSheet(STYLESHEET)
    return app
