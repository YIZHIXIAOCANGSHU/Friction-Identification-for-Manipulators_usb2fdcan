from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from friction_identification_core.io import (
    DamiaoSocketCanTransport,
    FeedbackFrameParser,
    RECV_FRAME_HEAD,
    RECV_FRAME_SIZE,
    RECV_FRAME_STRUCT,
)
from friction_identification_core.runtime_config import DEFAULT_CONFIG_PATH, load_config
from send import damiao as damiao_socketcan


class FakeCanTransport:
    def __init__(self) -> None:
        self.sent: list[tuple[int, bytes]] = []
        self.pending_recv: list[tuple[int, bytes]] = []
        self.closed = False

    def send(self, can_id: int, payload: bytes) -> None:
        self.sent.append((int(can_id), bytes(payload)))

    def recv(self, timeout: float = 0.0):  # noqa: ANN001
        _ = timeout
        if not self.pending_recv:
            return None
        return self.pending_recv.pop(0)

    def close(self) -> None:
        self.closed = True


class CoreConfigAndTransportTests(unittest.TestCase):
    def test_feedback_frame_parser(self) -> None:
        parser = FeedbackFrameParser(max_motor_id=7)
        payload = RECV_FRAME_STRUCT.pack(RECV_FRAME_HEAD, 3, 2, 1.0, -2.0, 0.5, 40.0)
        parser.feed(b"\x00\x01" + payload)
        frame = parser.pop_frame()
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.motor_id, 3)
        self.assertAlmostEqual(frame.velocity, -2.0, places=6)

    def test_default_config_reads_new_sections(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)

        self.assertEqual(
            config.transport.motor_types,
            ("DM8009", "DM8009", "DM4340", "DM4340", "DM4310", "DM4310", "DM4310"),
        )
        self.assertEqual(config.safety.hard_speed_abort_abs, 10.0)
        self.assertEqual(config.breakaway.torque_step, 0.01)
        self.assertEqual(config.breakaway.hold_duration, 0.25)
        self.assertEqual(config.identification.repeat_count, 3)
        self.assertEqual(config.identification.steady_speed_points, (0.5, 1.0, 2.0, 4.0, 6.0))
        self.assertEqual(config.identification.generation_safety_margin_ratio, 0.80)
        self.assertEqual(config.identification.phase_quality_retry_count, 1)
        self.assertEqual(config.identification.min_low_speed_fit_samples_per_direction, 8)
        self.assertEqual(config.identification.inertia_savgol_window_candidates, (21, 31, 41, 61, 81))
        self.assertEqual(config.identification.joint_speed_hold_weight, 3.0)
        self.assertEqual(config.identification.joint_inertia_weight, 1.0)
        self.assertEqual(config.identification.joint_dynamic_mit_weight, 0.5)
        self.assertEqual(config.identification.min_joint_fit_sample_count, 20)
        self.assertEqual(config.identification.min_tracking_ratio, 0.75)
        self.assertEqual(config.identification.max_steady_velocity_std_ratio, 0.15)
        self.assertEqual(config.identification.min_publishable_rounds, 2)
        self.assertAlmostEqual(float(config.breakaway.scan_max_torque[1]), 0.80)
        self.assertAlmostEqual(float(config.mit_velocity.kd_speed[1]), 1.2)
        self.assertEqual(config.mit_velocity.steady_hold_duration, 1.2)
        self.assertTrue(config.low_speed.enabled)
        self.assertEqual(config.low_speed.speed_points, (0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50, 0.75, 1.00))
        self.assertEqual(config.low_speed.hold_duration, 1.0)
        self.assertEqual(config.low_speed.ramp_acceleration, 0.5)
        self.assertEqual(config.low_speed.micro_motion_velocity_limit, 0.12)
        self.assertEqual(config.low_speed.micro_motion_kp, 0.0)
        self.assertEqual(config.low_speed.micro_motion_kd, 0.25)
        self.assertTrue(config.compensation.require_published_model)
        self.assertEqual(config.compensation.torque_limit_ratio, 0.25)
        self.assertEqual(config.compensation.torque_slew_rate_nm_s, 2.0)
        self.assertEqual(config.compensation.static_velocity_threshold_rad_s, 0.20)
        self.assertEqual(config.compensation.static_transition_velocity_rad_s, 0.50)
        self.assertEqual(config.compensation.soft_abort_start_ratio, 0.60)
        self.assertEqual(config.compensation.soft_abort_stop_ratio, 0.85)
        self.assertEqual(config.compensation.max_inertia_torque_ratio, 0.25)
        self.assertEqual(config.compensation.direction_hold_samples, 3)
        self.assertEqual(config.output.latest_parameters_json_filename, "latest_motor_parameters.json")

    def test_legacy_sections_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "motors:",
                        "  ids: [1]",
                        "step_torque:",
                        "  torque_step: 0.1",
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported legacy config section"):
                load_config(config_path)

    def test_inertia_savgol_window_candidates_default_to_savgol_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "identification:",
                        "  savgol_window: 25",
                        "  savgol_polyorder: 3",
                    )
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.identification.inertia_savgol_window_candidates, (25,))

    def test_invalid_inertia_savgol_window_candidates_are_rejected(self) -> None:
        cases = (
            ("[]", "must not be empty"),
            ("[20]", "must be odd"),
            ("[3]", "larger than identification.savgol_polyorder"),
        )
        for raw_value, error_text in cases:
            with self.subTest(raw_value=raw_value):
                with tempfile.TemporaryDirectory() as tmpdir:
                    config_path = Path(tmpdir) / "config.yaml"
                    config_path.write_text(
                        "\n".join(
                            (
                                "identification:",
                                "  savgol_window: 21",
                                "  savgol_polyorder: 3",
                                f"  inertia_savgol_window_candidates: {raw_value}",
                            )
                        ),
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(ValueError, error_text):
                        load_config(config_path)

    def test_invalid_transport_motor_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "motors:",
                        "  ids: [1, 2, 3, 4, 5, 6, 7]",
                        "transport:",
                        "  motor_types: [DM8009, DM8009, DM4340, DM4340, DM4310, DM4310, INVALID]",
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "transport.motor_types"):
                load_config(config_path)

    def test_socketcan_transport_supports_semantic_commands(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        fake_can = FakeCanTransport()
        transport = DamiaoSocketCanTransport(base_config, can_transport=fake_can)

        enable_trace = transport.enable_motor(1)
        self.assertTrue(enable_trace)
        self.assertEqual(
            fake_can.sent[0],
            damiao_socketcan.build_param_write_frame(0x01, 10, bytes([1, 0, 0, 0])),
        )
        self.assertEqual(fake_can.sent[1], damiao_socketcan.build_control_cmd_frame(0x01, 0xFC))

        fake_can.sent.clear()
        transport.send_mit_torque(1, 999.0)
        self.assertEqual(
            fake_can.sent[-1],
            damiao_socketcan.build_mit_frame(
                0x01,
                damiao_socketcan.DM_Motor_Type.DM8009,
                0.0,
                0.0,
                0.0,
                0.0,
                damiao_socketcan.get_motor_limits("DM8009").tmax,
            ),
        )

        fake_can.sent.clear()
        transport.send_mit_velocity(3, 2.0, 0.8)
        self.assertEqual(
            fake_can.sent[0],
            damiao_socketcan.build_param_write_frame(0x03, 10, bytes([1, 0, 0, 0])),
        )
        self.assertEqual(
            fake_can.sent[-1],
            damiao_socketcan.build_mit_frame(
                0x03,
                damiao_socketcan.DM_Motor_Type.DM4340,
                0.0,
                0.8,
                0.0,
                2.0,
                0.0,
            ),
        )

        fake_can.sent.clear()
        transport.send_mit_state(3, position=1.25, velocity=-2.0, kp=12.0, kd=0.8, torque_ff=999.0)
        self.assertEqual(
            fake_can.sent[-1],
            damiao_socketcan.build_mit_frame(
                0x03,
                damiao_socketcan.DM_Motor_Type.DM4340,
                12.0,
                0.8,
                1.25,
                -2.0,
                damiao_socketcan.get_motor_limits("DM4340").tmax,
            ),
        )

        fake_can.sent.clear()
        transport.send_zero_command(3, "mit_velocity")
        self.assertEqual(
            fake_can.sent[-1],
            damiao_socketcan.build_mit_frame(
                0x03,
                damiao_socketcan.DM_Motor_Type.DM4340,
                0.0,
                0.8,
                0.0,
                0.0,
                0.0,
            ),
        )

        fake_can.sent.clear()
        transport.send_velocity_mode(3, 1.5)
        self.assertEqual(
            fake_can.sent[0],
            damiao_socketcan.build_param_write_frame(0x03, 10, bytes([3, 0, 0, 0])),
        )
        self.assertEqual(fake_can.sent[-1], damiao_socketcan.build_vel_frame(0x03, 1.5))

        fake_can.sent.clear()
        transport.clear_error(3)
        self.assertEqual(fake_can.sent[0], damiao_socketcan.build_control_cmd_frame(0x203, 0xFB))

        fake_can.pending_recv.append((0x11, bytes([0x11, 0x7F, 0xFF, 0x7F, 0xF7, 0xFF, 40, 50])))
        transport.read(RECV_FRAME_SIZE)
        frame = transport.pop_feedback_frame()
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.motor_id, 1)
        self.assertEqual(frame.state, 1)
        self.assertAlmostEqual(frame.position, 0.0, delta=5e-4)
        self.assertAlmostEqual(frame.velocity, 0.0, delta=2e-2)
        self.assertAlmostEqual(frame.torque, 0.0, delta=2e-2)
        self.assertAlmostEqual(frame.mos_temperature, 40.0, places=6)

        fake_can.sent.clear()
        transport.close()
        self.assertTrue(fake_can.closed)
        self.assertEqual(len(fake_can.sent), 35)

    def test_default_config_reads_dynamic_mit_sections(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)

        self.assertEqual(config.inertia.waypoints, (0.0, 2.0, 4.0, 6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0, -4.0, -2.0, 0.0))
        self.assertTrue(config.dynamic_mit.enabled)
        self.assertEqual(config.dynamic_mit.trajectory_type, "sine")
        self.assertEqual(config.dynamic_mit.position_amplitude, 0.20)
        self.assertEqual(config.dynamic_mit.velocity_limit, 1.0)
        self.assertEqual(config.dynamic_mit.frequency_hz, 0.20)
        self.assertEqual(config.dynamic_mit.warmup_duration, 1.0)
        self.assertEqual(config.dynamic_mit.kp, 0.0)
        self.assertEqual(config.dynamic_mit.kd, 0.25)
        self.assertFalse(config.dynamic_mit.use_mit_estimated_torque)
        self.assertFalse(config.dynamic_mit.use_for_publish)

    def test_invalid_dynamic_mit_velocity_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "motors:",
                        "  ids: [1]",
                        "safety:",
                        "  hard_speed_abort_abs: 5.0",
                        "dynamic_mit:",
                        "  velocity_limit: 5.0",
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "dynamic_mit.velocity_limit"):
                load_config(config_path)

    def test_generation_speed_points_must_fit_identification_budget(self) -> None:
        cases = (
            ("identification:\n  steady_speed_points: [1.0, 4.0]", "identification.steady_speed_points"),
            ("identification:\n  steady_speed_points: [1.0, 2.0]\ninertia:\n  waypoints: [0.0, 4.1, 0.0]", "inertia.waypoints"),
            ("identification:\n  steady_speed_points: [1.0, 2.0]\nlow_speed:\n  speed_points: [0.1, 4.0]", "low_speed.speed_points"),
        )
        for body, error_text in cases:
            with self.subTest(error_text=error_text):
                with tempfile.TemporaryDirectory() as tmpdir:
                    config_path = Path(tmpdir) / "config.yaml"
                    config_path.write_text(
                        "\n".join(
                            (
                                "motors:",
                                "  ids: [1]",
                                "safety:",
                                "  hard_speed_abort_abs: 5.0",
                                "dynamic_mit:",
                                "  velocity_limit: 2.0",
                                body,
                            )
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, error_text):
                        load_config(config_path)

    def test_invalid_dynamic_mit_gain_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    (
                        "motors:",
                        "  ids: [1]",
                        "dynamic_mit:",
                        "  kp: -1.0",
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "dynamic_mit.kp"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
