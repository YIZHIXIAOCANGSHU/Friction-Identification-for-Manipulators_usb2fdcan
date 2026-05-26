from __future__ import annotations

from collections.abc import Callable
import time
import traceback

from PySide6.QtCore import QObject, Signal

from mit_sender.commands import SelectedMotorCommand, SingleMotorDebugCommand, TransportSettings
from mit_sender.damiao import (
    DamiaoMitController,
    ENABLE_CMD,
    MotorSpec,
    SocketCanTransport,
    build_control_cmd_frame,
    build_mit_frame,
    can_setup_hint,
    configure_can_interface,
    decode_feedback_frame,
    default_motor_specs,
    ensure_interface_ready,
    get_can_interface_state,
)
from mit_sender.rerun_feedback import FeedbackRerunLogger


def ensure_or_configure_interface(settings: TransportSettings) -> None:
    if get_can_interface_state(settings.interface) == "up":
        return
    if settings.configure_interface:
        configure_can_interface(
            settings.interface,
            settings.nominal_bitrate,
            settings.data_bitrate,
        )
    ensure_interface_ready(
        settings.interface,
        settings.nominal_bitrate,
        settings.data_bitrate,
    )


class CanSetupWorker(QObject):
    finished = Signal(bool, str)

    def __init__(self, settings: TransportSettings) -> None:
        super().__init__()
        self.settings = settings

    def run(self) -> None:
        try:
            if get_can_interface_state(self.settings.interface) == "up":
                self.finished.emit(True, f"{self.settings.interface} 已经启动。")
                return
            ensure_interface_ready(
                self.settings.interface,
                self.settings.nominal_bitrate,
                self.settings.data_bitrate,
            )
            self.finished.emit(True, f"{self.settings.interface} 已经启动。")
            return
        except RuntimeError as exc:
            if "is not UP" not in str(exc):
                detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                if "sudo ip link set" not in detail:
                    detail = f"{detail}\n{can_setup_hint(self.settings.interface, self.settings.nominal_bitrate, self.settings.data_bitrate)}"
                self.finished.emit(False, detail)
                return
        try:
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
            self.finished.emit(True, f"{self.settings.interface} 已执行配置并启动。")
        except Exception as exc:  # noqa: BLE001 - show system command errors directly.
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            if "sudo ip link set" not in detail:
                detail = f"{detail}\n{can_setup_hint(self.settings.interface, self.settings.nominal_bitrate, self.settings.data_bitrate)}"
            self.finished.emit(False, detail)


class FeedbackMonitorWorker(QObject):
    frame_received = Signal(object, float, int)
    status_changed = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        settings: TransportSettings,
        motor_specs: list[MotorSpec],
        logger: FeedbackRerunLogger | None = None,
        *,
        logger_factory: Callable[[list[MotorSpec]], FeedbackRerunLogger] | None = None,
        lock_first_feedback_id: bool = False,
        active_probe: bool = False,
        probe_interval_seconds: float = 0.02,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.motor_specs = list(motor_specs)
        self.logger = logger
        self._logger_factory = logger_factory or FeedbackRerunLogger
        self._running = True
        self._lock_first_feedback_id = bool(lock_first_feedback_id)
        self._locked_feedback_id: int | None = None
        self._locked_can_id: int | None = None
        self._active_probe = bool(active_probe)
        self._probe_interval_seconds = float(probe_interval_seconds)
        self._probe_index = 0
        self._last_probe_monotonic = 0.0

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        transport = None
        logger = self.logger
        try:
            ensure_or_configure_interface(self.settings)
            transport = SocketCanTransport(self.settings.interface)
            if logger is None:
                self.status_changed.emit("正在启动 Rerun viewer...")
                logger = self._logger_factory(self.motor_specs)
                self.logger = logger
            start_time = time.monotonic()
            self.status_changed.emit("正在读取反馈帧，Rerun 窗口已弹出。")
            while self._running:
                self._send_active_probe_if_due(transport)
                packet = transport.recv(timeout=0.01 if self._active_probe else 0.05)
                if packet is None:
                    continue
                feedback = decode_feedback_frame(packet[0], packet[1], self.motor_specs)
                if feedback is None:
                    continue
                if self._lock_first_feedback_id:
                    if self._locked_feedback_id is None:
                        self._locked_feedback_id = int(feedback.can_id)
                        self._locked_can_id = self._can_id_for_feedback(feedback.can_id)
                    elif int(feedback.can_id) != self._locked_feedback_id:
                        continue
                elapsed_seconds = time.monotonic() - start_time
                frame_count = logger.log_feedback(feedback, elapsed_seconds)
                self.frame_received.emit(feedback, elapsed_seconds, frame_count)
            self.status_changed.emit("反馈读取已停止。")
        except Exception as exc:  # noqa: BLE001 - show hardware/viewer errors directly.
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            if "sudo ip link set" not in detail and (
                "CAN interface" in detail or "CalledProcessError" in detail or "does not exist" in detail
            ):
                detail = f"{detail}\n{can_setup_hint(self.settings.interface, self.settings.nominal_bitrate, self.settings.data_bitrate)}"
            self.failed.emit(detail)
        finally:
            if transport is not None:
                transport.close()
            if logger is not None:
                logger.close()
            self.finished.emit()

    def _send_active_probe_if_due(self, transport: SocketCanTransport) -> None:
        if not self._active_probe or not self.motor_specs:
            return
        now = time.monotonic()
        if now - self._last_probe_monotonic < self._probe_interval_seconds:
            return
        self._last_probe_monotonic = now
        if self._locked_can_id is not None:
            spec = next((item for item in self.motor_specs if item.can_id == self._locked_can_id), None)
            if spec is not None:
                _send_zero_mit_probe(transport, spec)
            return
        spec = self.motor_specs[self._probe_index % len(self.motor_specs)]
        self._probe_index += 1
        transport.send(*build_control_cmd_frame(spec.can_id, ENABLE_CMD))
        _send_zero_mit_probe(transport, spec)

    def _can_id_for_feedback(self, feedback_id: int) -> int | None:
        spec = next((item for item in self.motor_specs if item.mst_id == int(feedback_id)), None)
        return spec.can_id if spec is not None else None


def _send_zero_mit_probe(transport: SocketCanTransport, spec: MotorSpec) -> None:
    transport.send(*build_mit_frame(spec.can_id, spec.motor_type, 0.0, 0.0, 0.0, 0.0, 0.0))


class MotorActionWorker(QObject):
    finished = Signal(bool, str)

    def __init__(
        self,
        action: str,
        settings: TransportSettings,
        commands: list[SelectedMotorCommand],
        debug_command: SingleMotorDebugCommand | None = None,
    ) -> None:
        super().__init__()
        self.action = action
        self.settings = settings
        self.commands = commands
        self.debug_command = debug_command

    def run(self) -> None:
        transport = None
        try:
            ensure_or_configure_interface(self.settings)
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
            elif self.action in {"debug_zero", "debug_set_mit", "debug_set_ids", "debug_configure"}:
                debug_command = self._require_debug_command()
                if self.action == "debug_zero":
                    controller.save_zero_position_raw(
                        debug_command.current_can_id,
                        debug_command.current_mode_offset,
                    )
                    message = f"已保存 0x{debug_command.current_can_id:03X} 当前零点。"
                elif self.action == "debug_set_mit":
                    controller.set_mit_mode_persistent_raw(
                        debug_command.current_can_id,
                        debug_command.current_mode_offset,
                    )
                    message = (
                        f"已将 0x{debug_command.current_can_id:03X} 写入 MIT 模式并保存，"
                        "断电保持；驱动器可能自动复位。"
                    )
                elif self.action == "debug_set_ids":
                    controller.set_motor_ids_persistent_raw(
                        debug_command.current_can_id,
                        debug_command.current_mode_offset,
                        debug_command.new_can_id,
                        debug_command.new_mst_id,
                    )
                    message = (
                        f"已设置并保存 ID: CAN 0x{debug_command.new_can_id:03X}, "
                        f"反馈 0x{debug_command.new_mst_id:03X}，断电保持；驱动器可能自动复位。"
                    )
                else:
                    controller.configure_single_motor_raw(
                        debug_command.current_can_id,
                        debug_command.current_mode_offset,
                        debug_command.new_can_id,
                        debug_command.new_mst_id,
                    )
                    message = (
                        "已完成单电机配置并保存，断电保持；驱动器可能自动复位。"
                        f"新 CAN ID 0x{debug_command.new_can_id:03X}, "
                        f"反馈 ID 0x{debug_command.new_mst_id:03X}。"
                    )
            else:
                raise ValueError(f"unknown action: {self.action}")
            self.finished.emit(True, message)
        except Exception as exc:  # noqa: BLE001 - show hardware/config errors directly.
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            if "sudo ip link set" not in detail and (
                "CAN interface" in detail or "CalledProcessError" in detail or "does not exist" in detail
            ):
                detail = f"{detail}\n{can_setup_hint(self.settings.interface, self.settings.nominal_bitrate, self.settings.data_bitrate)}"
            self.finished.emit(False, detail)
        finally:
            if transport is not None:
                transport.close()

    def _require_debug_command(self) -> SingleMotorDebugCommand:
        if self.debug_command is None:
            raise ValueError("debug command is required")
        return self.debug_command
