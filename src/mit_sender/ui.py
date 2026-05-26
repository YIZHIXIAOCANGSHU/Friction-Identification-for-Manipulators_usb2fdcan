from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mit_sender.commands import (
    DEBUG_MODE_OPTIONS,
    DEFAULT_DEBUG_FEEDBACK_OFFSET,
    MIT_COMMAND_DEFAULTS,
    MIT_FIELD_LABELS,
    MIT_FIELD_TOOLTIPS,
    SelectedMotorCommand,
    SingleMotorDebugCommand,
    TransportSettings,
    build_uniform_commands,
)
from mit_sender.damiao import (
    ALLOWED_INTERFACES,
    DEFAULT_DATA_BITRATE,
    DEFAULT_INTERFACE,
    DEFAULT_NOMINAL_BITRATE,
    DM_Motor_Type,
    MotorFeedback,
    MotorSpec,
    default_motor_specs,
)
from mit_sender.settings_store import SavedAppState, SettingsStore
from mit_sender.workers import CanSetupWorker, FeedbackMonitorWorker, MotorActionWorker


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
QGroupBox#toolbarGroup {
    margin-top: 8px;
    padding: 10px 10px 10px 10px;
}
QGroupBox#compactGroup {
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
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
QComboBox {
    background: #FFFFFF;
    border: 1px solid #C9D3E1;
    border-radius: 6px;
    min-height: 30px;
    padding: 2px 8px;
}
QComboBox:focus {
    border: 1px solid #2563EB;
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
QTabWidget::pane {
    border: 0;
}
QTabBar::tab {
    background: #E8EEF6;
    border: 1px solid #C9D3E1;
    border-bottom: 0;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #344155;
    font-weight: 700;
    min-height: 30px;
    padding: 5px 16px;
}
QTabBar::tab:selected {
    background: #FFFFFF;
    color: #172033;
}
QTabBar::tab:!selected {
    margin-top: 3px;
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
QLabel#warningText {
    color: #92400E;
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 6px;
    padding: 7px 9px;
}
QLabel#readonlyValue {
    background: #F3F6FA;
    border: 1px solid #D8E0EA;
    border-radius: 6px;
    min-height: 30px;
    padding: 2px 8px;
    color: #172033;
    font-weight: 700;
}
QLabel#sectionLabel {
    color: #344155;
    font-weight: 700;
}
QLabel#pathLabel {
    color: #64748B;
}
QFrame#panel {
    background: #FFFFFF;
    border: 1px solid #D8E0EA;
    border-radius: 8px;
}
QFrame#toolbarPanel {
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
        self._active_action_context: tuple[str, list[int]] | None = None
        self.can_setup_thread: QThread | None = None
        self.can_setup_worker: CanSetupWorker | None = None
        self.feedback_thread: QThread | None = None
        self.feedback_worker: FeedbackMonitorWorker | None = None
        self.feedback_dialog: FeedbackMonitorDialog | None = None
        self.feedback_motor_specs = list(self.motor_specs)
        self._single_motor_discovery_active = False
        self._single_motor_locked_feedback_id: int | None = None
        self.checkboxes: dict[int, QCheckBox] = {}
        self.inputs: dict[int, dict[str, QLineEdit]] = {}
        self.uniform_inputs: dict[str, QLineEdit] = {}
        self._last_auto_feedback_id = "0x011"
        self._manual_feedback_id_override = False
        self._loading_state = True

        self.setWindowTitle("MIT 电机一键发送")
        self.resize(1120, 720)
        self._build_ui()
        self._apply_saved_state()
        self._loading_state = False

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        layout.addLayout(self._build_header())
        layout.addWidget(self._build_transport_group())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_batch_tab(), "批量发送")
        self.tabs.addTab(self._build_single_motor_debug_tab(), "单电机调试")
        layout.addWidget(self.tabs, stretch=1)

        log_header = QHBoxLayout()
        log_label = QLabel("日志")
        log_label.setObjectName("sectionLabel")
        log_header.addWidget(log_label)
        log_header.addStretch(1)
        self.clear_log_button = QPushButton("清空日志")
        self.clear_log_button.clicked.connect(self._clear_log)
        log_header.addWidget(self.clear_log_button)
        layout.addLayout(log_header)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setPlaceholderText("状态输出")
        self.log_view.setFixedHeight(104)
        layout.addWidget(self.log_view)

        self.setCentralWidget(root)

    def _build_batch_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._build_uniform_command_group())
        layout.addWidget(self._build_motor_selection_bar())
        layout.addWidget(self._build_motor_table(), stretch=1)
        layout.addWidget(self._build_actions_panel())
        return tab

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
        group.setObjectName("toolbarGroup")
        grid = QGridLayout(group)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.interface_input = QComboBox()
        self.interface_input.addItems(ALLOWED_INTERFACES)
        self.interface_input.setCurrentText(DEFAULT_INTERFACE)
        self.interface_input.setMinimumWidth(120)
        self.interface_input.setToolTip("SocketCAN 接口，只允许 can0 或 can1")
        self.interface_input.currentIndexChanged.connect(lambda _index: self._save_state())
        self.nominal_input = QLineEdit(str(DEFAULT_NOMINAL_BITRATE))
        self.nominal_input.setValidator(QIntValidator(1, 10_000_000, self))
        self.nominal_input.setToolTip("仲裁波特率，默认 1000000")
        self.data_input = QLineEdit(str(DEFAULT_DATA_BITRATE))
        self.data_input.setValidator(QIntValidator(1, 20_000_000, self))
        self.data_input.setToolTip("CAN FD 数据波特率，默认 5000000")
        self.configure_checkbox = QCheckBox("启动前自动配置接口")
        self.configure_checkbox.setToolTip("执行 ip link set ... fd on；可能需要权限")
        self.start_can_button = QPushButton("启动 CAN")
        self.start_can_button.setToolTip("立即执行 ip link set ... fd on 并拉起所选 CAN 接口")
        self.start_can_button.clicked.connect(self._start_can_interface)

        grid.addWidget(QLabel("接口"), 0, 0)
        grid.addWidget(self.interface_input, 0, 1)
        grid.addWidget(QLabel("仲裁波特率"), 0, 2)
        grid.addWidget(self.nominal_input, 0, 3)
        grid.addWidget(QLabel("数据波特率"), 0, 4)
        grid.addWidget(self.data_input, 0, 5)
        grid.addWidget(self.configure_checkbox, 0, 6)
        grid.addWidget(self.start_can_button, 0, 7)
        grid.setColumnStretch(8, 1)
        return group

    def _build_uniform_command_group(self) -> QGroupBox:
        group = QGroupBox("统一 MIT 指令")
        group.setObjectName("compactGroup")
        layout = QHBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 8, 10, 8)

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

    def _build_motor_selection_bar(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("toolbarPanel")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        title = QLabel("目标电机")
        title.setObjectName("subtitleLabel")
        layout.addWidget(title)

        self.selected_count_label = QLabel("已选 0/0")
        self.selected_count_label.setObjectName("statusPill")
        layout.addWidget(self.selected_count_label)

        self.select_all_button = QPushButton("全选")
        self.select_all_button.setToolTip("勾选所有电机作为发送目标")
        self.select_all_button.clicked.connect(self._select_all_motors)
        layout.addWidget(self.select_all_button)

        self.select_none_button = QPushButton("全不选")
        self.select_none_button.setToolTip("取消所有电机发送目标")
        self.select_none_button.clicked.connect(self._select_no_motors)
        layout.addWidget(self.select_none_button)

        layout.addStretch(1)
        return panel

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
        for column in range(5, len(columns)):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        return table

    def _build_actions_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self.feedback_button = QPushButton("开启读取数据")
        self.feedback_button.setToolTip("按当前页面读取反馈数据，并同步打开 Rerun viewer")
        self.feedback_button.clicked.connect(self._start_current_tab_feedback_monitor)
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

    def _build_single_motor_debug_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_debug_input_group())
        layout.addWidget(self._build_debug_feedback_group())
        layout.addWidget(self._build_debug_action_group())
        layout.addStretch(1)
        return tab

    def _build_debug_input_group(self) -> QGroupBox:
        group = QGroupBox("连接与 ID")
        group.setObjectName("compactGroup")
        self.debug_connection_group = group
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 8, 10, 8)

        warning = QLabel("此页用于总线上只连接一个待配置电机的场景；多电机同时在线时不建议修改 ID。")
        warning.setObjectName("warningText")
        warning.setWordWrap(True)
        layout.addWidget(warning)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.debug_current_can_input = QLineEdit("0x01")
        self.debug_current_can_input.setToolTip("当前电机 CAN ID，支持十六进制或十进制，范围 0x001..0x7FE")
        self.debug_current_can_input.setAlignment(Qt.AlignRight)
        self.debug_current_can_input.textChanged.connect(self._handle_debug_input_changed)

        self.debug_new_can_input = QLineEdit("0x01")
        self.debug_new_can_input.setToolTip("要写入的目标 CAN ID，支持十六进制或十进制，范围 0x001..0x7FE")
        self.debug_new_can_input.setAlignment(Qt.AlignRight)
        self.debug_new_can_input.textChanged.connect(self._handle_debug_input_changed)

        self.debug_new_mst_input = QLineEdit("0x011")
        self.debug_new_mst_input.setToolTip("要写入的目标反馈 ID / Master ID，支持十六进制或十进制，默认等于目标 CAN ID + 0x10")
        self.debug_new_mst_input.setAlignment(Qt.AlignRight)
        self.debug_new_mst_input.textChanged.connect(self._handle_debug_feedback_input_changed)

        self.debug_mode_combo = QComboBox()
        for label, offset in DEBUG_MODE_OPTIONS:
            self.debug_mode_combo.addItem(f"{label} (0x{offset:03X})", offset)
        self.debug_mode_combo.setToolTip("保存零点控制帧使用的当前模式 ID 偏移")
        self.debug_mode_combo.currentIndexChanged.connect(self._handle_debug_mode_changed)

        grid.addWidget(QLabel("当前 CAN ID"), 0, 0)
        grid.addWidget(self.debug_current_can_input, 0, 1)
        grid.addWidget(QLabel("目标 CAN ID"), 0, 2)
        grid.addWidget(self.debug_new_can_input, 0, 3)
        grid.addWidget(QLabel("目标反馈 ID"), 0, 4)
        grid.addWidget(self.debug_new_mst_input, 0, 5)
        grid.addWidget(QLabel("当前模式"), 1, 0)
        grid.addWidget(self.debug_mode_combo, 1, 1)
        grid.setColumnStretch(6, 1)
        layout.addLayout(grid)

        summary = QFrame()
        summary.setObjectName("toolbarPanel")
        summary_layout = QGridLayout(summary)
        summary_layout.setContentsMargins(10, 8, 10, 8)
        summary_layout.setHorizontalSpacing(10)
        summary_layout.setVerticalSpacing(6)

        self.debug_effective_can_label = self._readonly_label("0x001")
        self.debug_effective_feedback_label = self._readonly_label("0x011")
        self.debug_effective_mode_label = self._readonly_label("MIT (0x000)")
        self.debug_detected_can_label = self._readonly_label("-")
        self.debug_detected_feedback_label = self._readonly_label("-")
        self.debug_detected_state_label = self._readonly_label("-")

        summary_layout.addWidget(QLabel("将写入 CAN ID"), 0, 0)
        summary_layout.addWidget(self.debug_effective_can_label, 0, 1)
        summary_layout.addWidget(QLabel("将写入反馈 ID"), 0, 2)
        summary_layout.addWidget(self.debug_effective_feedback_label, 0, 3)
        summary_layout.addWidget(QLabel("将写入模式"), 0, 4)
        summary_layout.addWidget(self.debug_effective_mode_label, 0, 5)
        summary_layout.addWidget(QLabel("当前检测 CAN"), 1, 0)
        summary_layout.addWidget(self.debug_detected_can_label, 1, 1)
        summary_layout.addWidget(QLabel("当前检测反馈"), 1, 2)
        summary_layout.addWidget(self.debug_detected_feedback_label, 1, 3)
        summary_layout.addWidget(QLabel("当前状态"), 1, 4)
        summary_layout.addWidget(self.debug_detected_state_label, 1, 5)
        summary_layout.setColumnStretch(6, 1)
        layout.addWidget(summary)

        manual_row = QHBoxLayout()
        manual_row.setSpacing(8)
        manual_row.addWidget(QLabel("协议说明书"))
        self.manual_path_label = QLabel(str(self._manual_path()))
        self.manual_path_label.setObjectName("pathLabel")
        self.manual_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        manual_row.addWidget(self.manual_path_label, stretch=1)
        self.open_manual_button = QPushButton("打开说明书")
        self.open_manual_button.clicked.connect(self._open_manual)
        manual_row.addWidget(self.open_manual_button)
        layout.addLayout(manual_row)

        return group

    def _build_debug_feedback_group(self) -> QGroupBox:
        group = QGroupBox("实时反馈")
        group.setObjectName("compactGroup")
        self.debug_feedback_group = group
        grid = QGridLayout(group)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.debug_feedback_status_label = self._readonly_label("未读取")
        self.debug_feedback_position_label = self._readonly_label("-")
        self.debug_feedback_velocity_label = self._readonly_label("-")
        self.debug_feedback_torque_label = self._readonly_label("-")
        self.debug_feedback_time_label = self._readonly_label("-")
        self.debug_read_button = QPushButton("开启读取数据")
        self.debug_read_button.setToolTip("主动扫描 1-7：挨个使能并发送全 0 MIT 帧，读到反馈后直接启动 Rerun")
        self.debug_read_button.clicked.connect(self._start_debug_tab_feedback_monitor)

        grid.addWidget(QLabel("读取状态"), 0, 0)
        grid.addWidget(self.debug_feedback_status_label, 0, 1)
        grid.addWidget(QLabel("当前位置"), 0, 2)
        grid.addWidget(self.debug_feedback_position_label, 0, 3)
        grid.addWidget(QLabel("当前速度"), 0, 4)
        grid.addWidget(self.debug_feedback_velocity_label, 0, 5)
        grid.addWidget(QLabel("当前力矩"), 1, 0)
        grid.addWidget(self.debug_feedback_torque_label, 1, 1)
        grid.addWidget(QLabel("反馈时间"), 1, 2)
        grid.addWidget(self.debug_feedback_time_label, 1, 3)
        grid.addWidget(self.debug_read_button, 1, 5)
        grid.setColumnStretch(6, 1)
        return group

    def _build_debug_action_group(self) -> QFrame:
        panel = QGroupBox("调试动作")
        panel.setObjectName("compactGroup")
        self.debug_actions_group = panel
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        self.debug_zero_button = QPushButton("保存当前位置为零点")
        self.debug_zero_button.clicked.connect(lambda: self._start_action("debug_zero"))
        layout.addWidget(self.debug_zero_button)

        self.debug_set_mit_button = QPushButton("写入 MIT 模式")
        self.debug_set_mit_button.clicked.connect(lambda: self._start_action("debug_set_mit"))
        layout.addWidget(self.debug_set_mit_button)

        self.debug_set_ids_button = QPushButton("设置 ID")
        self.debug_set_ids_button.clicked.connect(lambda: self._start_action("debug_set_ids"))
        layout.addWidget(self.debug_set_ids_button)

        layout.addStretch(1)

        self.debug_configure_button = QPushButton("失能后配置单电机")
        self.debug_configure_button.setObjectName("primaryButton")
        self.debug_configure_button.clicked.connect(lambda: self._start_action("debug_configure"))
        layout.addWidget(self.debug_configure_button)

        return panel

    def _readonly_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("readonlyValue")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label.setMinimumWidth(92)
        return label

    def _apply_saved_state(self) -> None:
        state = self.saved_state
        self.interface_input.setCurrentText(state.transport.interface if state.transport.interface in ALLOWED_INTERFACES else DEFAULT_INTERFACE)
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

        self.debug_current_can_input.setText(self._format_can_id(state.debug_command.current_can_id))
        self.debug_new_can_input.setText(self._format_can_id(state.debug_command.new_can_id))
        self.debug_new_mst_input.setText(self._format_can_id(state.debug_command.new_mst_id))
        self._last_auto_feedback_id = self._default_feedback_id_text(state.debug_command.new_can_id)
        self._manual_feedback_id_override = state.debug_command.new_mst_id != self._parse_int_text(self._last_auto_feedback_id)
        self._set_debug_mode_offset(state.debug_command.current_mode_offset)
        self._update_debug_feedback_label()
        self._update_debug_summary()

        if state.window_geometry:
            self.restoreGeometry(state.window_geometry)
        self._sync_enable_button()
        self._append_log("已恢复上次界面输入。")

    def _mit_value_validator(self) -> QDoubleValidator:
        validator = QDoubleValidator(-100000.0, 100000.0, 6, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        return validator

    def _settings(self) -> TransportSettings:
        self._mark_valid(self.nominal_input)
        self._mark_valid(self.data_input)
        interface = self.interface_input.currentText().strip()
        if interface not in ALLOWED_INTERFACES:
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

    def _debug_command(self) -> SingleMotorDebugCommand:
        current_can_id = self._read_can_id(self.debug_current_can_input, "当前 CAN ID")
        new_can_id = self._read_can_id(self.debug_new_can_input, "目标 CAN ID")
        new_mst_id = self._read_can_id(self.debug_new_mst_input, "目标反馈 ID")
        self._update_debug_feedback_label()
        return SingleMotorDebugCommand(
            current_can_id=current_can_id,
            current_mode_offset=int(self.debug_mode_combo.currentData()),
            new_can_id=new_can_id,
            new_mst_id=new_mst_id,
        )

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

    def _read_can_id(self, line_edit: QLineEdit, label: str) -> int:
        text = line_edit.text().strip()
        if not text:
            self._mark_invalid(line_edit)
            raise ValueError(f"{label} 不能为空")
        try:
            value = self._parse_int_text(text)
        except ValueError as exc:
            self._mark_invalid(line_edit)
            raise ValueError(f"{label} 必须是十六进制或十进制整数") from exc
        if value < 0x001 or value > 0x7FE:
            self._mark_invalid(line_edit)
            raise ValueError(f"{label} 必须在 0x001..0x7FE 范围内")
        self._mark_valid(line_edit)
        return value

    def _toggle_enable(self) -> None:
        action = "disable" if self._selected_motors_are_enabled() else "enable"
        self._start_action(action)

    def _select_all_motors(self) -> None:
        self._set_motor_selection(True)

    def _select_no_motors(self) -> None:
        self._set_motor_selection(False)

    def _set_motor_selection(self, checked: bool) -> None:
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(bool(checked))
        self._sync_enable_button()
        self._save_state()

    def _send_commands(self) -> None:
        self._start_action("send")

    def _send_uniform_commands(self) -> None:
        self._start_action("send_uniform")

    def _start_can_interface(self) -> None:
        if self.can_setup_thread is not None:
            return
        try:
            settings = self._settings()
        except Exception as exc:  # noqa: BLE001 - input errors should be visible.
            QMessageBox.warning(self, "输入错误", str(exc))
            return
        self._save_state()
        self._set_busy(True)
        self._set_status(f"启动 CAN: {settings.interface}")
        self._append_log(
            f"启动 CAN: {settings.interface} bitrate={settings.nominal_bitrate} dbitrate={settings.data_bitrate}"
        )
        self.can_setup_thread = QThread(self)
        self.can_setup_worker = CanSetupWorker(settings)
        self.can_setup_worker.moveToThread(self.can_setup_thread)
        self.can_setup_thread.started.connect(self.can_setup_worker.run)
        self.can_setup_worker.finished.connect(self._finish_can_setup)
        self.can_setup_worker.finished.connect(self.can_setup_thread.quit)
        self.can_setup_worker.finished.connect(self.can_setup_worker.deleteLater)
        self.can_setup_thread.finished.connect(self.can_setup_thread.deleteLater)
        self.can_setup_thread.finished.connect(self._clear_can_setup_worker)
        self.can_setup_thread.start()

    def _finish_can_setup(self, ok: bool, message: str) -> None:
        self._set_busy(False)
        if ok:
            self._set_status("CAN 已就绪")
            self._append_log(f"完成: {message}")
            return
        self._set_status("CAN 启动失败")
        self._append_log(f"失败: {message}")
        QMessageBox.critical(self, "CAN 启动失败", message)

    def _clear_can_setup_worker(self) -> None:
        self.can_setup_thread = None
        self.can_setup_worker = None

    def _show_feedback_monitor(self, motor_specs: list[MotorSpec] | None = None) -> None:
        desired_specs = list(motor_specs or self.motor_specs)
        if self.feedback_dialog is None:
            self.feedback_motor_specs = desired_specs
            self.feedback_dialog = FeedbackMonitorDialog(self.feedback_motor_specs, self.settings_store, self)
            self.feedback_dialog.restore_saved_geometry(self.saved_state)
            self.feedback_dialog.start_requested.connect(self._start_feedback_monitor)
            self.feedback_dialog.stop_requested.connect(self._stop_feedback_monitor)
            self.feedback_dialog.destroyed.connect(self._feedback_dialog_destroyed)
        elif self.feedback_thread is None and self.feedback_motor_specs != desired_specs:
            self.feedback_dialog.close()
            self.feedback_dialog = FeedbackMonitorDialog(desired_specs, self.settings_store, self)
            self.feedback_motor_specs = desired_specs
            self.feedback_dialog.start_requested.connect(self._start_feedback_monitor)
            self.feedback_dialog.stop_requested.connect(self._stop_feedback_monitor)
            self.feedback_dialog.destroyed.connect(self._feedback_dialog_destroyed)
        self.feedback_dialog.show()
        self.feedback_dialog.raise_()
        self.feedback_dialog.activateWindow()

    def _start_current_tab_feedback_monitor(self) -> None:
        try:
            specs = self._feedback_specs_for_current_tab()
        except Exception as exc:  # noqa: BLE001 - input errors should be visible.
            QMessageBox.warning(self, "输入错误", str(exc))
            return
        self._start_feedback_monitor(specs)

    def _start_debug_tab_feedback_monitor(self) -> None:
        self._start_single_motor_discovery(self._debug_discovery_specs())

    def _feedback_specs_for_current_tab(self) -> list[MotorSpec]:
        if self.tabs.currentIndex() == 0:
            selected_specs = [
                spec
                for spec in self.motor_specs
                if self.checkboxes[spec.motor_id].isChecked()
            ]
            if not selected_specs:
                raise ValueError("请至少勾选一个电机")
            return selected_specs

        return self._debug_discovery_specs()

    def _debug_discovery_specs(self) -> list[MotorSpec]:
        specs = [
            MotorSpec(
                motor_id=index,
                can_id=index,
                mst_id=index + DEFAULT_DEBUG_FEEDBACK_OFFSET,
                motor_type=self.motor_specs[0].motor_type,
            )
            for index in range(1, 8)
        ]
        try:
            debug_command = self._debug_command()
        except Exception:
            return specs

        extra_pairs = (
            (debug_command.current_can_id, debug_command.current_can_id + DEFAULT_DEBUG_FEEDBACK_OFFSET),
            (debug_command.new_can_id, debug_command.new_mst_id),
        )
        seen = {(spec.can_id, spec.mst_id) for spec in specs}
        for can_id, mst_id in extra_pairs:
            if mst_id < 0x001 or mst_id > 0x7FE or (can_id, mst_id) in seen:
                continue
            specs.append(
                MotorSpec(
                    motor_id=len(specs) + 1,
                    can_id=can_id,
                    mst_id=mst_id,
                    motor_type=self.motor_specs[0].motor_type,
                )
            )
            seen.add((can_id, mst_id))
        return specs

    def _start_single_motor_discovery(self, motor_specs: list[MotorSpec] | None = None) -> None:
        if self.feedback_thread is not None:
            return
        try:
            settings = self._settings()
        except Exception as exc:  # noqa: BLE001 - input errors should be visible.
            QMessageBox.warning(self, "输入错误", str(exc))
            return
        desired_specs = list(motor_specs or self._debug_discovery_specs())
        self._save_state()
        self.feedback_motor_specs = desired_specs
        self._single_motor_discovery_active = True
        self._single_motor_locked_feedback_id = None
        self.debug_detected_can_label.setText("-")
        self.debug_detected_feedback_label.setText("-")
        self.debug_detected_state_label.setText("-")
        self.debug_feedback_position_label.setText("-")
        self.debug_feedback_velocity_label.setText("-")
        self.debug_feedback_torque_label.setText("-")
        self.debug_feedback_time_label.setText("-")
        self.debug_feedback_status_label.setText("主动扫描 1-7 中")
        feedback_ids = ", ".join(f"0x{spec.mst_id:03X}" for spec in desired_specs)
        self._append_log(
            f"单电机主动扫描: {settings.interface} bitrate={settings.nominal_bitrate} "
            f"dbitrate={settings.data_bitrate} feedback=[{feedback_ids}]"
        )
        self.feedback_button.setEnabled(False)
        self.debug_read_button.setEnabled(False)
        self._set_status("读取线程启动中")

        self.feedback_thread = QThread(self)
        self.feedback_worker = FeedbackMonitorWorker(
            settings,
            desired_specs,
            lock_first_feedback_id=True,
            active_probe=True,
        )
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

    def _debug_feedback_specs(self) -> list[MotorSpec]:
        debug_command = self._debug_command()
        return [
            MotorSpec(
                motor_id=1,
                can_id=debug_command.new_can_id,
                mst_id=debug_command.new_mst_id,
                motor_type=self.motor_specs[0].motor_type,
            )
        ]

    def _start_feedback_monitor(self, motor_specs: list[MotorSpec] | None = None) -> None:
        if self.feedback_thread is not None:
            return
        try:
            settings = self._settings()
        except Exception as exc:  # noqa: BLE001 - input errors should be visible.
            QMessageBox.warning(self, "输入错误", str(exc))
            return
        desired_specs = list(motor_specs or self.feedback_motor_specs or self.motor_specs)
        self._save_state()
        feedback_ids = ", ".join(f"0x{spec.mst_id:03X}" for spec in desired_specs)
        self._append_log(
            f"开启读取数据: {settings.interface} bitrate={settings.nominal_bitrate} "
            f"dbitrate={settings.data_bitrate} feedback=[{feedback_ids}]"
        )
        self._show_feedback_monitor(desired_specs)
        assert self.feedback_dialog is not None
        self.feedback_dialog.set_running(True)
        self.feedback_dialog.set_status("正在启动 Rerun...")
        self.feedback_button.setEnabled(False)
        self.debug_read_button.setEnabled(False)
        self.debug_feedback_status_label.setText("启动中")
        self._set_status("读取数据中")

        self.feedback_thread = QThread(self)
        self.feedback_worker = FeedbackMonitorWorker(settings, desired_specs)
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
        self._handle_single_motor_discovery_frame(feedback)
        self._update_debug_feedback_values(feedback, elapsed_seconds)

    def _set_feedback_status(self, status: str) -> None:
        if self.feedback_dialog is not None:
            self.feedback_dialog.set_status(status)
        if self._single_motor_discovery_active and self._single_motor_locked_feedback_id is None:
            self._set_status("主动扫描 1-7 中")
            self.debug_feedback_status_label.setText("主动扫描 1-7 中")
            self._append_log(str(status))
            return
        self._set_status("读取数据中" if "正在读取" in str(status) else status)
        self.debug_feedback_status_label.setText(str(status))
        self._append_log(str(status))

    def _fail_feedback_monitor(self, message: str) -> None:
        if self.feedback_dialog is not None:
            self.feedback_dialog.set_status(f"失败: {message}")
            self.feedback_dialog.set_running(False)
        self._set_status("读取失败")
        self._single_motor_discovery_active = False
        self.debug_feedback_status_label.setText("读取失败")
        self._append_log(f"反馈读取失败: {message}")
        QMessageBox.critical(self, "反馈读取失败", message)

    def _clear_feedback_worker(self) -> None:
        self.feedback_thread = None
        self.feedback_worker = None
        self._single_motor_discovery_active = False
        self.feedback_button.setEnabled(True)
        self.debug_read_button.setEnabled(True)
        self._set_status("就绪")
        if self.debug_feedback_status_label.text() != "读取失败":
            self.debug_feedback_status_label.setText("未读取")
        if self.feedback_dialog is not None:
            self.feedback_dialog.set_running(False)

    def _feedback_dialog_destroyed(self) -> None:
        self.feedback_dialog = None

    def _start_action(self, action: str) -> None:
        if self.worker_thread is not None:
            return
        try:
            settings = self._settings()
            debug_command = self._debug_command() if action.startswith("debug_") else None
            if debug_command is not None:
                commands: list[SelectedMotorCommand] = []
            else:
                commands = self._uniform_commands() if action == "send_uniform" else self._selected_commands()
        except Exception as exc:  # noqa: BLE001 - input errors should be visible.
            QMessageBox.warning(self, "输入错误", str(exc))
            return

        self._save_state()
        self._set_busy(True)
        self._set_status(f"执行中: {self._action_label(action)}")
        self._append_log(f"开始: {self._action_label(action)}")
        self.worker_thread = QThread(self)
        self.worker = MotorActionWorker(action, settings, commands, debug_command=debug_command)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        command_ids = [command.motor_id for command in commands]
        self._active_action_context = (action, command_ids)
        self.worker.finished.connect(self._finish_current_action)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker)
        self.worker_thread.start()

    @Slot(bool, str)
    def _finish_current_action(self, ok: bool, message: str) -> None:
        action, motor_ids = self._active_action_context or ("unknown", [])
        self._active_action_context = None
        self._finish_action(action, motor_ids, ok, message)

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
            if action in {"debug_zero", "debug_set_mit", "debug_set_ids", "debug_configure"}:
                if action in {"debug_set_ids", "debug_configure"}:
                    self._apply_successful_id_change()
                    self._stop_feedback_monitor()
                self.debug_feedback_status_label.setText("操作完成，点击开启读取数据")
                self._append_log("调试动作已完成；如需查看反馈，请点击单电机页的“开启读取数据”。")
            return
        self._set_status("操作失败")
        self._append_log(f"失败: {message}")
        QMessageBox.critical(self, "操作失败", message)

    def _clear_worker(self) -> None:
        self.worker_thread = None
        self.worker = None
        self._active_action_context = None

    def _set_busy(self, busy: bool) -> None:
        self.toggle_enable_button.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.uniform_send_button.setEnabled(not busy)
        self.select_all_button.setEnabled(not busy)
        self.select_none_button.setEnabled(not busy)
        self.start_can_button.setEnabled(not busy)
        self.feedback_button.setEnabled(not busy and self.feedback_thread is None)
        self.debug_read_button.setEnabled(not busy and self.feedback_thread is None)
        self.debug_zero_button.setEnabled(not busy)
        self.debug_set_mit_button.setEnabled(not busy)
        self.debug_set_ids_button.setEnabled(not busy)
        self.debug_configure_button.setEnabled(not busy)
        self.open_manual_button.setEnabled(not busy)
        self.clear_log_button.setEnabled(not busy)

    def _sync_enable_button(self) -> None:
        self.toggle_enable_button.setText("全部失能" if self._selected_motors_are_enabled() else "全部使能")
        if hasattr(self, "selected_count_label"):
            selected_count = len(self._selected_motor_ids())
            self.selected_count_label.setText(f"已选 {selected_count}/{len(self.motor_specs)}")

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

    def _clear_log(self) -> None:
        self.log_view.clear()

    def _action_label(self, action: str) -> str:
        labels = {
            "enable": "全部使能",
            "disable": "全部失能",
            "send": "一键发送",
            "send_uniform": "统一发送",
            "debug_zero": "保存当前位置为零点",
            "debug_set_mit": "写入 MIT 模式",
            "debug_set_ids": "设置 ID",
            "debug_configure": "失能后配置单电机",
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

    def _handle_debug_input_changed(self, _text: str) -> None:
        self._sync_debug_feedback_default()
        self._update_debug_feedback_label()
        self._update_debug_summary()
        self._save_state()

    def _handle_debug_feedback_input_changed(self, _text: str) -> None:
        try:
            current_feedback_id = self._parse_int_text(self.debug_new_mst_input.text())
            auto_feedback_id = self._parse_int_text(self._last_auto_feedback_id)
        except ValueError:
            self._manual_feedback_id_override = True
        else:
            self._manual_feedback_id_override = current_feedback_id != auto_feedback_id
        self._update_debug_feedback_label()
        self._update_debug_summary()
        self._save_state()

    def _handle_debug_mode_changed(self, _index: int) -> None:
        self._update_debug_summary()
        self._save_state()

    def _apply_successful_id_change(self) -> None:
        try:
            debug_command = self._debug_command()
        except Exception as exc:  # noqa: BLE001 - keep the action completion path robust.
            self._append_log(f"无法同步新 ID 到当前 CAN ID: {exc}")
            return
        self.debug_current_can_input.blockSignals(True)
        self.debug_current_can_input.setText(self._format_can_id(debug_command.new_can_id))
        self.debug_current_can_input.blockSignals(False)
        self._mark_valid(self.debug_current_can_input)
        self._append_log(f"当前 CAN ID 已更新为 {self._format_can_id(debug_command.new_can_id)}。")
        self._save_state()

    def _update_debug_feedback_label(self, new_mst_id: int | None = None) -> None:
        if new_mst_id is None:
            try:
                new_mst_id = self._parse_int_text(self.debug_new_mst_input.text())
            except ValueError:
                self.debug_effective_feedback_label.setText("-")
                return
        self.debug_effective_feedback_label.setText(self._format_can_id(new_mst_id) if 0x001 <= new_mst_id <= 0x7FE else "超出范围")

    def _update_debug_summary(self) -> None:
        self.debug_effective_can_label.setText(self.debug_new_can_input.text().strip() or "-")
        self._update_debug_feedback_label()
        self.debug_effective_mode_label.setText(self.debug_mode_combo.currentText())
        if self.feedback_thread is None and self.debug_feedback_status_label.text() not in {"读取失败", "启动中"}:
            self.debug_feedback_status_label.setText("未读取")

    def _update_debug_feedback_values(self, feedback: MotorFeedback, elapsed_seconds: float) -> None:
        try:
            debug_command = self._debug_command()
        except Exception:
            debug_command = None
        expected_feedback_ids = {spec.mst_id for spec in self.feedback_motor_specs}
        if debug_command is not None:
            expected_feedback_ids.add(debug_command.new_mst_id)
            expected_feedback_ids.add(debug_command.current_can_id + DEFAULT_DEBUG_FEEDBACK_OFFSET)
        if self._single_motor_locked_feedback_id is not None:
            expected_feedback_ids = {self._single_motor_locked_feedback_id}
        if feedback.can_id not in expected_feedback_ids:
            return
        if self._single_motor_discovery_active and self._single_motor_locked_feedback_id is not None:
            self.debug_feedback_status_label.setText(
                "已锁定 CAN "
                f"{self._format_can_id(self._can_id_for_feedback(feedback))} / "
                f"反馈 {self._format_can_id(feedback.can_id)}"
            )
        else:
            self.debug_feedback_status_label.setText(f"读取中 #{feedback.can_id:03X}")
        self.debug_feedback_position_label.setText(f"{feedback.position:+.6f}")
        self.debug_feedback_velocity_label.setText(f"{feedback.velocity:+.6f}")
        self.debug_feedback_torque_label.setText(f"{feedback.torque:+.6f}")
        self.debug_feedback_time_label.setText(f"{elapsed_seconds:.3f}s")

    def _handle_single_motor_discovery_frame(self, feedback: MotorFeedback) -> None:
        if not self._single_motor_discovery_active:
            return
        if self._single_motor_locked_feedback_id is not None and feedback.can_id != self._single_motor_locked_feedback_id:
            return
        if self._single_motor_locked_feedback_id is None:
            self._single_motor_locked_feedback_id = int(feedback.can_id)
            detected_can_id = self._can_id_for_feedback(feedback)
            if 0x001 <= detected_can_id <= 0x7FE:
                self.debug_current_can_input.blockSignals(True)
                self.debug_current_can_input.setText(self._format_can_id(detected_can_id))
                self.debug_current_can_input.blockSignals(False)
                self._mark_valid(self.debug_current_can_input)
            self.debug_detected_can_label.setText(self._format_can_id(detected_can_id))
            self.debug_detected_feedback_label.setText(self._format_can_id(feedback.can_id))
            self.debug_detected_state_label.setText(str(feedback.state))
            self.debug_feedback_status_label.setText(
                f"已锁定 CAN {self._format_can_id(detected_can_id)} / 反馈 {self._format_can_id(feedback.can_id)}"
            )
            self._set_status("已锁定单电机")
            self._append_log(
                f"单电机已锁定: CAN {self._format_can_id(detected_can_id)}, "
                f"反馈 {self._format_can_id(feedback.can_id)}"
            )
            self._save_state()
            return
        self.debug_detected_state_label.setText(str(feedback.state))

    def _can_id_for_feedback(self, feedback: MotorFeedback) -> int:
        spec = next((item for item in self.feedback_motor_specs if item.mst_id == int(feedback.can_id)), None)
        return spec.can_id if spec is not None else int(feedback.controller_id)

    def _start_debug_feedback_after_action(self, action: str) -> None:
        try:
            debug_command = self._debug_command()
        except Exception as exc:  # noqa: BLE001 - state was just validated, but keep UI robust.
            self._append_log(f"无法启动调试反馈读取: {exc}")
            return
        feedback_can_id = (
            debug_command.new_mst_id
            if action in {"debug_set_ids", "debug_configure"}
            else debug_command.current_can_id + DEFAULT_DEBUG_FEEDBACK_OFFSET
        )
        if feedback_can_id < 0x001 or feedback_can_id > 0x7FE:
            self._append_log("无法启动调试反馈读取: 反馈 ID 超出 0x001..0x7FE")
            return
        feedback_spec = MotorSpec(
            motor_id=1,
            can_id=debug_command.new_can_id if action in {"debug_set_ids", "debug_configure"} else debug_command.current_can_id,
            mst_id=feedback_can_id,
            motor_type=self.motor_specs[0].motor_type,
        )
        self.debug_feedback_status_label.setText(f"准备读取 0x{feedback_can_id:03X}")
        self._start_feedback_monitor([feedback_spec])

    def _set_debug_mode_offset(self, offset: int) -> None:
        index = self.debug_mode_combo.findData(int(offset))
        self.debug_mode_combo.setCurrentIndex(index if index >= 0 else 0)

    def _format_can_id(self, value: int) -> str:
        return f"0x{int(value):03X}"

    def _default_feedback_id_text(self, can_id: int) -> str:
        value = int(can_id) + DEFAULT_DEBUG_FEEDBACK_OFFSET
        if value < 0x001 or value > 0x7FE:
            value = DEFAULT_DEBUG_FEEDBACK_OFFSET + 1
        return self._format_can_id(value)

    def _sync_debug_feedback_default(self) -> None:
        try:
            new_can_id = self._parse_int_text(self.debug_new_can_input.text())
        except ValueError:
            return
        next_auto = self._default_feedback_id_text(new_can_id)
        if not getattr(self, "_manual_feedback_id_override", False):
            self.debug_new_mst_input.blockSignals(True)
            self.debug_new_mst_input.setText(next_auto)
            self.debug_new_mst_input.blockSignals(False)
        self._last_auto_feedback_id = next_auto

    def _parse_int_text(self, value: str) -> int:
        text = str(value).strip()
        base = 16 if text.lower().startswith("0x") else 10
        return int(text, base)

    def _manual_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "send" / "调试助手使用说明书（达妙驱动控制协议）V1.4.pdf"

    def _open_manual(self) -> None:
        path = self._manual_path()
        if not path.exists():
            QMessageBox.warning(self, "说明书不存在", f"未找到协议说明书:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _save_state(self) -> None:
        if self._loading_state:
            return
        try:
            transport = self._settings()
        except Exception:
            transport = self.saved_state.transport
        try:
            debug_command = self._debug_command()
        except Exception:
            debug_command = self.saved_state.debug_command
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
            debug_command=debug_command,
        )

    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt supplies QCloseEvent at runtime.
        self._save_state()
        self._stop_feedback_monitor()
        if self.feedback_thread is not None:
            self.feedback_thread.quit()
            if not self.feedback_thread.wait(2000):
                self._append_log("反馈线程未在 2 秒内停止，继续关闭窗口。")
        if self.can_setup_thread is not None:
            self.can_setup_thread.quit()
            if not self.can_setup_thread.wait(2000):
                self._append_log("CAN 启动线程未在 2 秒内停止，继续关闭窗口。")
        super().closeEvent(event)


def create_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setApplicationName("DamiaoMitSender")
    app.setOrganizationName("MitSender")
    app.setStyleSheet(STYLESHEET)
    return app
