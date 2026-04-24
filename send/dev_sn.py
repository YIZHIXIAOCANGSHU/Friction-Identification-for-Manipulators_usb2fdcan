import argparse
import sys

import usb.core
import usb.util

TARGET_VID = 0x34B7
TARGET_PID = 0x6877
MISSING_SERIAL = "[No serial number]"


def get_u2canfd_devices():
    devices = usb.core.find(find_all=True, idVendor=TARGET_VID, idProduct=TARGET_PID)
    return list(devices or [])


def get_serial_number(dev):
    if not getattr(dev, "iSerialNumber", 0):
        return MISSING_SERIAL

    try:
        return usb.util.get_string(dev, dev.iSerialNumber) or MISSING_SERIAL
    except usb.core.USBError:
        return MISSING_SERIAL


def list_u2canfd_devices(stream=sys.stdout, sn_only=False):
    devices = get_u2canfd_devices()
    if not devices:
        print(
            f"未找到 U2CANFD 设备 (VID: 0x{TARGET_VID:04X}, PID: 0x{TARGET_PID:04X})。",
            file=stream,
        )
        return 1

    for i, dev in enumerate(devices):
        serial_number = get_serial_number(dev)

        if sn_only:
            print(serial_number, file=stream)
            continue

        print(f"U2CANFD_DEV {i}:", file=stream)
        print(f"  VID: 0x{dev.idVendor:04x}", file=stream)
        print(f"  PID: 0x{dev.idProduct:04x}", file=stream)
        print(f"  SN: {serial_number}", file=stream)
        print(file=stream)

    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="识别 U2CANFD 设备的 SN 码")
    parser.add_argument(
        "--sn-only",
        action="store_true",
        help="仅输出 SN，便于在命令行中继续处理",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return list_u2canfd_devices(sn_only=args.sn_only)


if __name__ == "__main__":
    raise SystemExit(main())
