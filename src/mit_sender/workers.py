from __future__ import annotations

import time
import traceback

from PySide6.QtCore import QObject, Signal

from mit_sender.commands import SelectedMotorCommand, TransportSettings
from mit_sender.damiao import (
    DamiaoMitController,
    MotorSpec,
    SocketCanTransport,
    configure_can_interface,
    decode_feedback_frame,
    default_motor_specs,
    ensure_interface_ready,
)
from mit_sender.rerun_feedback import FeedbackRerunLogger


class FeedbackMonitorWorker(QObject):
    frame_received = Signal(object, float, int)
    status_changed = Signal(str)
    failed = Signal(str)
    finished = Signal()

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


class MotorActionWorker(QObject):
    finished = Signal(bool, str)

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
