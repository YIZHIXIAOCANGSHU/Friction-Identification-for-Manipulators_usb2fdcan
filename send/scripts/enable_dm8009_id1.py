import argparse
import errno
import subprocess
import sys
import time

import rerun as rr
import rerun.blueprint as rrb

try:
    from scripts.socketcan_damiao import (
        Control_Mode,
        DM_Motor_Type,
        RUNNING,
        STATE_CODE_LABELS,
        SocketCanTransport,
        build_control_cmd_frame,
        build_mit_frame,
        configure_can_interface,
        decode_feedback,
        ensure_interface_ready,
    )
except ImportError:
    from socketcan_damiao import (
        Control_Mode,
        DM_Motor_Type,
        RUNNING,
        STATE_CODE_LABELS,
        SocketCanTransport,
        build_control_cmd_frame,
        build_mit_frame,
        configure_can_interface,
        decode_feedback,
        ensure_interface_ready,
    )


DEFAULT_CAN_ID = 0x01
DEFAULT_MST_ID = 0x11
DEFAULT_MOTOR_TYPE = "DM8009"
DEFAULT_INTERFACE = "can0"
ENABLE_CMD = 0xFC
DISABLE_CMD = 0xFD
DEFAULT_COMMAND_INTERVAL_MS = 0.0
DEFAULT_LISTEN_DURATION = 0.0
DEFAULT_PRINT_INTERVAL = 0.1
DEFAULT_SEND_RATE_LOG_INTERVAL = 0.1
DEFAULT_BACKPRESSURE_SLEEP = 0.0005
MAX_BACKPRESSURE_SLEEP = 0.01


def build_enable_frame(can_id: int) -> tuple[int, bytes]:
    return build_control_cmd_frame(can_id + Control_Mode.MIT_MODE, ENABLE_CMD)


def build_disable_frame(can_id: int) -> tuple[int, bytes]:
    return build_control_cmd_frame(can_id + Control_Mode.MIT_MODE, DISABLE_CMD)


def build_zero_mit_frame(can_id: int, motor_type: DM_Motor_Type) -> tuple[int, bytes]:
    return build_mit_frame(can_id, motor_type, 0.0, 0.0, 0.0, 0.0, 0.0)


def send_frame(transport: SocketCanTransport, frame: tuple[int, bytes]) -> None:
    can_id, payload = frame
    backpressure_sleep = DEFAULT_BACKPRESSURE_SLEEP
    while True:
        try:
            transport.send(can_id, payload)
            return
        except OSError as exc:
            if exc.errno != errno.ENOBUFS:
                raise
            time.sleep(backpressure_sleep)
            backpressure_sleep = min(backpressure_sleep * 2.0, MAX_BACKPRESSURE_SLEEP)


def send_repeated_frame(transport: SocketCanTransport, frame: tuple[int, bytes], count: int, interval_seconds: float) -> None:
    for _ in range(count):
        send_frame(transport, frame)
        time.sleep(interval_seconds)


def build_feedback_views(base_path: str) -> list:
    return [
        rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/position"], name="Position"),
        rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/velocity"], name="Velocity"),
        rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/torque"], name="Torque"),
        rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/state_code"], name="State Code"),
        rrb.TimeSeriesView(origin="/", contents=[f"{base_path}/send_rate_hz"], name="Send Rate"),
        rrb.TimeSeriesView(
            origin="/",
            contents=[f"{base_path}/mos_temp", f"{base_path}/rotor_temp"],
            name="Temperatures",
        ),
        rrb.TextLogView(origin=f"{base_path}/events", name="Motor Events"),
    ]


def build_rerun_blueprint(base_path: str) -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(contents=build_feedback_views(base_path)),
        collapse_panels=True,
    )


def setup_rerun(base_path: str, application_id: str):
    rr.init(application_id, spawn=True, default_blueprint=build_rerun_blueprint(base_path))
    rr.log(f"{base_path}/position", rr.SeriesLines(colors=[[255, 99, 71]], names=["position"], widths=[2]), static=True)
    rr.log(f"{base_path}/velocity", rr.SeriesLines(colors=[[30, 144, 255]], names=["velocity"], widths=[2]), static=True)
    rr.log(f"{base_path}/torque", rr.SeriesLines(colors=[[60, 179, 113]], names=["torque"], widths=[2]), static=True)
    rr.log(f"{base_path}/state_code", rr.SeriesLines(colors=[[255, 215, 0]], names=["state_code"], widths=[2]), static=True)
    rr.log(f"{base_path}/send_rate_hz", rr.SeriesLines(colors=[[138, 43, 226]], names=["send_rate_hz"], widths=[2]), static=True)
    rr.log(f"{base_path}/mos_temp", rr.SeriesLines(colors=[[255, 140, 0]], names=["mos_temp"], widths=[2]), static=True)
    rr.log(f"{base_path}/rotor_temp", rr.SeriesLines(colors=[[220, 20, 60]], names=["rotor_temp"], widths=[2]), static=True)


def log_feedback_to_rerun(base_path: str, elapsed_seconds: float, feedback):
    rr.set_time("feedback_time", duration=elapsed_seconds)
    rr.log(f"{base_path}/position", rr.Scalars([feedback.position]))
    rr.log(f"{base_path}/velocity", rr.Scalars([feedback.velocity]))
    rr.log(f"{base_path}/torque", rr.Scalars([feedback.torque]))
    rr.log(f"{base_path}/state_code", rr.Scalars([feedback.state_code]))
    rr.log(f"{base_path}/mos_temp", rr.Scalars([feedback.mos_temp]))
    rr.log(f"{base_path}/rotor_temp", rr.Scalars([feedback.rotor_temp]))
    state_label = STATE_CODE_LABELS.get(feedback.state_code, f"unknown_{feedback.state_code:X}")
    rr.log(
        f"{base_path}/events",
        rr.TextLog(
            (
                f"state={state_label} controller_id=0x{feedback.controller_id:02X} "
                f"pos={feedback.position:.4f} vel={feedback.velocity:.4f} "
                f"tau={feedback.torque:.4f} mos_temp={feedback.mos_temp:.1f} "
                f"rotor_temp={feedback.rotor_temp:.1f}"
            ),
            level="INFO",
        ),
    )


def log_send_rate_to_rerun(base_path: str, elapsed_seconds: float, send_rate_hz: float):
    rr.set_time("feedback_time", duration=elapsed_seconds)
    rr.log(f"{base_path}/send_rate_hz", rr.Scalars([send_rate_hz]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="持续向 DM8009 id=0x01 发送零 MIT 控制帧，并实时监听反馈（绕过 usb_class，直接走 SocketCAN）"
    )
    parser.add_argument("--mst-id", type=lambda value: int(value, 0), default=DEFAULT_MST_ID, help="反馈帧 ID，默认 0x11")
    parser.add_argument(
        "--mode",
        default="mit",
        choices=["mit", "pos-vel", "vel", "pos-force"],
        help="保留兼容参数；当前脚本仅支持 mit，默认 mit",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="兼容旧参数；当前 MIT 连续控制脚本会忽略它",
    )
    parser.add_argument("--count", type=int, default=5, help="初始使能帧重复发送次数，默认 5 次")
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=DEFAULT_COMMAND_INTERVAL_MS,
        help="零 MIT 帧发送周期；<=0 表示不主动休眠、尽可能高频发送，默认 0ms",
    )
    parser.add_argument("--nom-bitrate", type=int, default=1000000, help="CAN 仲裁域波特率")
    parser.add_argument("--data-bitrate", type=int, default=5000000, help="CAN FD 数据域波特率")
    parser.add_argument("--configure-interface", action="store_true", help="启动前尝试配置并拉起接口")
    parser.add_argument(
        "--listen-duration",
        type=float,
        default=DEFAULT_LISTEN_DURATION,
        help="连续发送零 MIT 帧的时长；<=0 表示一直运行直到 Ctrl+C，默认 0 秒",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=DEFAULT_PRINT_INTERVAL,
        help="终端反馈打印间隔，单位秒；<=0 表示收到反馈就打印，默认 0.1 秒",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将要发送的帧，不真正发包")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode != "mit":
        raise RuntimeError("当前脚本仅支持 mit 模式连续零命令流")

    if args.configure_interface:
        configure_can_interface(DEFAULT_INTERFACE, args.nom_bitrate, args.data_bitrate)

    if not args.dry_run:
        ensure_interface_ready(DEFAULT_INTERFACE, args.nom_bitrate, args.data_bitrate)

    base_path = f"/motor/dm8009_id_{DEFAULT_CAN_ID:02X}"
    motor_type = DM_Motor_Type[DEFAULT_MOTOR_TYPE]
    enable_frame = build_enable_frame(DEFAULT_CAN_ID)
    disable_frame = build_disable_frame(DEFAULT_CAN_ID)
    zero_mit_frame = build_zero_mit_frame(DEFAULT_CAN_ID, motor_type)
    duration_label = "forever" if args.listen_duration <= 0 else str(args.listen_duration)

    print(
        f"target_motor={DEFAULT_MOTOR_TYPE} can_id=0x{DEFAULT_CAN_ID:02X} "
        f"mst_id=0x{args.mst_id:02X} interface={DEFAULT_INTERFACE} "
        f"nominal_bitrate={args.nom_bitrate} data_bitrate={args.data_bitrate} frame_type=canfd_brs "
        f"control_mode=mit command_interval_ms={args.interval_ms} listen_duration={duration_label}"
    )
    if args.all_modes:
        print("warning: --all-modes is ignored in continuous MIT mode")
    print(f"enable_frame id=0x{enable_frame[0]:03X} data={enable_frame[1].hex()}")
    print(f"zero_mit_frame id=0x{zero_mit_frame[0]:03X} data={zero_mit_frame[1].hex()}")

    if args.dry_run:
        return 0

    setup_rerun(base_path, application_id="dm8009_zero_mit_feedback")
    transport = SocketCanTransport(DEFAULT_INTERFACE)
    start_time = time.monotonic()
    end_time = None if args.listen_duration <= 0 else start_time + args.listen_duration
    command_interval_seconds = args.interval_ms / 1000.0 if args.interval_ms > 0 else None
    last_state_code = None
    next_feedback_print_at = start_time
    last_send_rate_log_time = start_time
    sends_since_rate_log = 0
    try:
        send_repeated_frame(transport, enable_frame, args.count, 0.002)

        while RUNNING.is_set() and (end_time is None or time.monotonic() < end_time):
            loop_start = time.perf_counter()
            send_frame(transport, zero_mit_frame)
            sends_since_rate_log += 1

            loop_time = time.monotonic()
            send_rate_window = loop_time - last_send_rate_log_time
            if send_rate_window >= DEFAULT_SEND_RATE_LOG_INTERVAL:
                log_send_rate_to_rerun(base_path, loop_time - start_time, sends_since_rate_log / send_rate_window)
                last_send_rate_log_time = loop_time
                sends_since_rate_log = 0

            while True:
                packet = transport.recv(timeout=0.0)
                if packet is None:
                    break

                can_id, payload = packet
                if can_id != args.mst_id:
                    continue

                feedback = decode_feedback(payload, motor_type)
                feedback_time = time.monotonic()
                log_feedback_to_rerun(base_path, feedback_time - start_time, feedback)

                state_label = STATE_CODE_LABELS.get(feedback.state_code, f"unknown_{feedback.state_code:X}")
                if feedback.state_code != last_state_code:
                    print(f"feedback_state={state_label} controller_id=0x{feedback.controller_id:02X}")
                    last_state_code = feedback.state_code

                if args.print_interval <= 0 or feedback_time >= next_feedback_print_at:
                    print(
                        f"feedback pos={feedback.position:.4f} vel={feedback.velocity:.4f} "
                        f"tau={feedback.torque:.4f} mos={feedback.mos_temp:.1f} rotor={feedback.rotor_temp:.1f}"
                    )
                    if args.print_interval > 0:
                        next_feedback_print_at = feedback_time + args.print_interval

            if command_interval_seconds is not None:
                sleep_time = command_interval_seconds - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        try:
            send_repeated_frame(transport, disable_frame, args.count, 0.002)
        except OSError:
            pass
        transport.close()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    except subprocess.CalledProcessError as exc:
        print(f"命令执行失败: {' '.join(exc.cmd)}", file=sys.stderr)
        raise SystemExit(exc.returncode) from None
