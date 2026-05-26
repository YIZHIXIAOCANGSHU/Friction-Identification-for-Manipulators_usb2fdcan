from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QSettings

from mit_sender.commands import (
    DEBUG_MODE_OFFSETS,
    DEFAULT_DEBUG_FEEDBACK_OFFSET,
    MIT_COMMAND_DEFAULTS,
    SingleMotorDebugCommand,
    TransportSettings,
    default_single_motor_debug_command,
)
from mit_sender.damiao import (
    ALLOWED_INTERFACES,
    DEFAULT_DATA_BITRATE,
    DEFAULT_INTERFACE,
    DEFAULT_NOMINAL_BITRATE,
    MotorSpec,
)


ORGANIZATION_NAME = "MitSender"
APPLICATION_NAME = "DamiaoMitSender"


@dataclass(frozen=True)
class SavedAppState:
    transport: TransportSettings = field(
        default_factory=lambda: TransportSettings(
            interface=DEFAULT_INTERFACE,
            nominal_bitrate=DEFAULT_NOMINAL_BITRATE,
            data_bitrate=DEFAULT_DATA_BITRATE,
            configure_interface=False,
        )
    )
    selected_motor_ids: set[int] = field(default_factory=set)
    motor_commands: dict[int, dict[str, str]] = field(default_factory=dict)
    uniform_command: dict[str, str] = field(default_factory=lambda: dict(MIT_COMMAND_DEFAULTS))
    debug_command: SingleMotorDebugCommand = field(default_factory=default_single_motor_debug_command)
    window_geometry: bytes | None = None
    feedback_geometry: bytes | None = None


class SettingsStore:
    def __init__(self, settings: QSettings | None = None) -> None:
        self._settings = settings or QSettings(ORGANIZATION_NAME, APPLICATION_NAME)

    @property
    def settings(self) -> QSettings:
        return self._settings

    def load(self, motor_specs: list[MotorSpec]) -> SavedAppState:
        selected_motor_ids = self._read_selected_motor_ids(motor_specs)

        return SavedAppState(
            transport=TransportSettings(
                interface=self._read_interface("transport/interface", DEFAULT_INTERFACE),
                nominal_bitrate=self._read_int(
                    "transport/nominal_bitrate",
                    DEFAULT_NOMINAL_BITRATE,
                    minimum=1,
                    maximum=10_000_000,
                ),
                data_bitrate=self._read_int(
                    "transport/data_bitrate",
                    DEFAULT_DATA_BITRATE,
                    minimum=1,
                    maximum=20_000_000,
                ),
                configure_interface=self._read_bool("transport/configure_interface", False),
            ),
            selected_motor_ids=selected_motor_ids,
            motor_commands={
                spec.motor_id: self._read_command_group(f"motors/{spec.motor_id}/command")
                for spec in motor_specs
            },
            uniform_command=self._read_command_group("uniform_command"),
            debug_command=self._read_debug_command(),
            window_geometry=self._read_bytes("window/geometry"),
            feedback_geometry=self._read_bytes("feedback/geometry"),
        )

    def save(
        self,
        *,
        transport: TransportSettings,
        selected_motor_ids: set[int],
        motor_commands: dict[int, dict[str, str]],
        uniform_command: dict[str, str],
        window_geometry: bytes | None,
        feedback_geometry: bytes | None = None,
        debug_command: SingleMotorDebugCommand | None = None,
    ) -> None:
        self._settings.setValue("transport/interface", transport.interface)
        self._settings.setValue("transport/nominal_bitrate", int(transport.nominal_bitrate))
        self._settings.setValue("transport/data_bitrate", int(transport.data_bitrate))
        self._settings.setValue("transport/configure_interface", bool(transport.configure_interface))
        self._settings.setValue(
            "motors/selected_ids",
            ",".join(str(motor_id) for motor_id in sorted(selected_motor_ids)),
        )

        for motor_id, command in motor_commands.items():
            self._settings.setValue(f"motors/{int(motor_id)}/selected", int(motor_id) in selected_motor_ids)
            self._write_command_group(f"motors/{int(motor_id)}/command", command)

        self._write_command_group("uniform_command", uniform_command)
        if debug_command is not None:
            self._write_debug_command(debug_command)
        if window_geometry is not None:
            self._settings.setValue("window/geometry", window_geometry)
        if feedback_geometry is not None:
            self._settings.setValue("feedback/geometry", feedback_geometry)
        self._settings.sync()

    def _read_command_group(self, prefix: str) -> dict[str, str]:
        values: dict[str, str] = {}
        for field, default in MIT_COMMAND_DEFAULTS.items():
            value = self._read_text(f"{prefix}/{field}", default)
            values[field] = value if _is_float_text(value) else default
        return values

    def _write_command_group(self, prefix: str, command: dict[str, str]) -> None:
        for field, default in MIT_COMMAND_DEFAULTS.items():
            value = str(command.get(field, default)).strip()
            self._settings.setValue(f"{prefix}/{field}", value if _is_float_text(value) else default)

    def _read_debug_command(self) -> SingleMotorDebugCommand:
        defaults = default_single_motor_debug_command()
        current_can_id = self._read_can_id("debug/current_can_id", defaults.current_can_id)
        new_can_id = self._read_can_id("debug/new_can_id", defaults.new_can_id)
        auto_mst_id = new_can_id + DEFAULT_DEBUG_FEEDBACK_OFFSET
        if auto_mst_id > 0x7FE:
            auto_mst_id = defaults.new_mst_id
        new_mst_id = self._read_can_id("debug/new_mst_id", auto_mst_id)
        mode_offset = self._read_int(
            "debug/current_mode_offset",
            defaults.current_mode_offset,
            minimum=min(DEBUG_MODE_OFFSETS),
            maximum=max(DEBUG_MODE_OFFSETS),
        )
        if mode_offset not in DEBUG_MODE_OFFSETS:
            mode_offset = defaults.current_mode_offset
        return SingleMotorDebugCommand(
            current_can_id=current_can_id,
            current_mode_offset=mode_offset,
            new_can_id=new_can_id,
            new_mst_id=new_mst_id,
        )

    def _write_debug_command(self, command: SingleMotorDebugCommand) -> None:
        defaults = default_single_motor_debug_command()
        current_can_id = _valid_can_id_or_default(command.current_can_id, defaults.current_can_id)
        new_can_id = _valid_can_id_or_default(command.new_can_id, defaults.new_can_id)
        new_mst_id = _valid_can_id_or_default(
            command.new_mst_id,
            _default_feedback_id(new_can_id, defaults.new_mst_id),
        )
        mode_offset = command.current_mode_offset
        if mode_offset not in DEBUG_MODE_OFFSETS:
            mode_offset = defaults.current_mode_offset
        self._settings.setValue("debug/current_can_id", current_can_id)
        self._settings.setValue("debug/current_mode_offset", mode_offset)
        self._settings.setValue("debug/new_can_id", new_can_id)
        self._settings.setValue("debug/new_mst_id", new_mst_id)

    def _read_selected_motor_ids(self, motor_specs: list[MotorSpec]) -> set[int]:
        valid_ids = {spec.motor_id for spec in motor_specs}
        saved_ids = self._settings.value("motors/selected_ids")
        if saved_ids is not None:
            selected_ids: set[int] = set()
            for item in str(saved_ids).split(","):
                text = item.strip()
                if not text:
                    continue
                try:
                    motor_id = int(text)
                except ValueError:
                    continue
                if motor_id in valid_ids:
                    selected_ids.add(motor_id)
            return selected_ids
        return {
            spec.motor_id
            for spec in motor_specs
            if self._read_bool(f"motors/{spec.motor_id}/selected", True)
        }

    def _read_text(self, key: str, default: str) -> str:
        value = self._settings.value(key, default)
        text = str(value).strip()
        return text or default

    def _read_interface(self, key: str, default: str) -> str:
        text = self._read_text(key, default)
        return text if text in ALLOWED_INTERFACES else default

    def _read_int(self, key: str, default: int, *, minimum: int, maximum: int) -> int:
        value = self._settings.value(key, default)
        try:
            parsed = _parse_int_text(str(value))
        except (TypeError, ValueError):
            return default
        if parsed < minimum or parsed > maximum:
            return default
        return parsed

    def _read_can_id(self, key: str, default: int) -> int:
        return self._read_int(key, default, minimum=0x001, maximum=0x7FE)

    def _read_bool(self, key: str, default: bool) -> bool:
        value = self._settings.value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    def _read_bytes(self, key: str) -> bytes | None:
        value = self._settings.value(key)
        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        data = getattr(value, "data", None)
        if callable(data):
            return bytes(data())
        return None


def _is_float_text(value: str) -> bool:
    try:
        float(str(value).strip())
    except (TypeError, ValueError):
        return False
    return True


def _valid_can_id_or_default(value: int, default: int) -> int:
    parsed = int(value)
    if parsed < 0x001 or parsed > 0x7FE:
        return int(default)
    return parsed


def _default_feedback_id(new_can_id: int, fallback: int) -> int:
    value = int(new_can_id) + DEFAULT_DEBUG_FEEDBACK_OFFSET
    if value < 0x001 or value > 0x7FE:
        return int(fallback)
    return value


def _parse_int_text(value: str) -> int:
    text = str(value).strip()
    base = 16 if text.lower().startswith("0x") else 10
    return int(text, base)
