import io
import unittest
from types import SimpleNamespace
from unittest import mock

import dev_sn


def make_device(serial_index=1):
    return SimpleNamespace(
        idVendor=dev_sn.TARGET_VID,
        idProduct=dev_sn.TARGET_PID,
        iSerialNumber=serial_index,
    )


class DevSnTests(unittest.TestCase):
    def test_no_devices_prints_helpful_message(self):
        stream = io.StringIO()

        with mock.patch.object(dev_sn.usb.core, "find", return_value=[]):
            exit_code = dev_sn.list_u2canfd_devices(stream=stream)

        self.assertEqual(exit_code, 1)
        self.assertIn("未找到 U2CANFD 设备", stream.getvalue())

    def test_default_output_contains_sn(self):
        stream = io.StringIO()

        with mock.patch.object(dev_sn.usb.core, "find", return_value=[make_device()]), mock.patch.object(
            dev_sn.usb.util, "get_string", return_value="SN-001"
        ):
            exit_code = dev_sn.list_u2canfd_devices(stream=stream)

        self.assertEqual(exit_code, 0)
        output = stream.getvalue()
        self.assertIn("U2CANFD_DEV 0:", output)
        self.assertIn("VID: 0x34b7", output)
        self.assertIn("PID: 0x6877", output)
        self.assertIn("SN: SN-001", output)

    def test_sn_only_outputs_plain_serial_numbers(self):
        stream = io.StringIO()
        devices = [make_device(1), make_device(2)]

        with mock.patch.object(dev_sn.usb.core, "find", return_value=devices), mock.patch.object(
            dev_sn.usb.util, "get_string", side_effect=["SN-001", "SN-002"]
        ):
            exit_code = dev_sn.list_u2canfd_devices(stream=stream, sn_only=True)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stream.getvalue().strip().splitlines(), ["SN-001", "SN-002"])


if __name__ == "__main__":
    unittest.main()
