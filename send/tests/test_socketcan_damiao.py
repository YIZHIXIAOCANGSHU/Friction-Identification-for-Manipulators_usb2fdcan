import struct
import unittest

from scripts import socketcan_damiao as scd


class SocketCanDamiaoTests(unittest.TestCase):
    def test_build_vel_frame_uses_little_endian_float(self):
        can_id, payload = scd.build_vel_frame(0x01, 2.0)
        self.assertEqual(can_id, 0x201)
        self.assertEqual(payload, struct.pack("<f", 2.0))

    def test_build_param_write_frame_layout_matches_existing_protocol(self):
        can_id, payload = scd.build_param_write_frame(0x01, 10, bytes([3, 0, 0, 0]))
        self.assertEqual(can_id, 0x7FF)
        self.assertEqual(payload, bytes([0x01, 0x00, 0x55, 0x0A, 0x03, 0x00, 0x00, 0x00]))

    def test_build_mit_frame_matches_expected_zero_command_encoding(self):
        can_id, payload = scd.build_mit_frame(0x01, scd.DM_Motor_Type.DM4310, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(can_id, 0x01)
        self.assertEqual(payload, bytes([0x7F, 0xFF, 0x7F, 0xF0, 0x00, 0x00, 0x07, 0xFF]))

    def test_decode_feedback_round_trips_midpoint_values(self):
        feedback = scd.decode_feedback(bytes([0x11, 0x7F, 0xFF, 0x7F, 0xF7, 0xFF, 40, 50]), scd.DM_Motor_Type.DM4310)
        self.assertAlmostEqual(feedback.position, 0.0, delta=5e-4)
        self.assertAlmostEqual(feedback.velocity, 0.0, delta=2e-2)
        self.assertAlmostEqual(feedback.torque, 0.0, delta=1e-2)
        self.assertEqual(feedback.controller_id, 0x01)
        self.assertEqual(feedback.state_code, 0x01)
        self.assertEqual(feedback.mos_temp, 40.0)
        self.assertEqual(feedback.rotor_temp, 50.0)

    def test_pack_and_unpack_classic_can_frame(self):
        packet = scd.pack_can_frame(0x201, b"\x01\x02\x03\x04")
        can_id, payload = scd.unpack_can_packet(packet)
        self.assertEqual(can_id, 0x201)
        self.assertEqual(payload, b"\x01\x02\x03\x04")

    def test_pack_canfd_frame_uses_brs_flag(self):
        packet = scd.pack_canfd_frame(0x201, b"\x01\x02\x03\x04", flags=scd.CANFD_BRS)
        can_id, payload = scd.unpack_can_packet(packet)
        self.assertEqual(can_id, 0x201)
        self.assertEqual(payload, b"\x01\x02\x03\x04")


if __name__ == "__main__":
    unittest.main()
