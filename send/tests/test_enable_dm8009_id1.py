import unittest
from unittest import mock

from scripts import enable_dm8009_id1 as script


class EnableDm8009Id1Tests(unittest.TestCase):
    def test_parser_defaults_to_continuous_mit_mode(self):
        args = script.build_parser().parse_args([])
        self.assertEqual(args.mode, "mit")
        self.assertEqual(args.interval_ms, script.DEFAULT_COMMAND_INTERVAL_MS)
        self.assertEqual(args.listen_duration, script.DEFAULT_LISTEN_DURATION)
        self.assertEqual(args.print_interval, script.DEFAULT_PRINT_INTERVAL)

    def test_parser_rejects_custom_interface_argument(self):
        with self.assertRaises(SystemExit):
            script.build_parser().parse_args(["--interface", "can1"])

    def test_build_feedback_views_separates_metrics(self):
        views = script.build_feedback_views("/motor/dm8009_id_01")
        self.assertEqual([view.name for view in views], ["Position", "Velocity", "Torque", "State Code", "Send Rate", "Temperatures", "Motor Events"])
        self.assertEqual(views[0].contents, ["/motor/dm8009_id_01/position"])
        self.assertEqual(views[1].contents, ["/motor/dm8009_id_01/velocity"])
        self.assertEqual(views[2].contents, ["/motor/dm8009_id_01/torque"])
        self.assertEqual(views[3].contents, ["/motor/dm8009_id_01/state_code"])
        self.assertEqual(views[4].contents, ["/motor/dm8009_id_01/send_rate_hz"])
        self.assertEqual(views[5].contents, ["/motor/dm8009_id_01/mos_temp", "/motor/dm8009_id_01/rotor_temp"])

    def test_build_zero_mit_frame_for_default_motor(self):
        frame_id, payload = script.build_zero_mit_frame(0x01, script.DM_Motor_Type.DM8009)
        self.assertEqual(frame_id, 0x001)
        self.assertEqual(payload, bytes([0x7F, 0xFF, 0x7F, 0xF0, 0x00, 0x00, 0x07, 0xFF]))

    def test_build_enable_frame_for_mit_mode(self):
        frame_id, payload = script.build_enable_frame(0x01)
        self.assertEqual(frame_id, 0x001)
        self.assertEqual(payload, bytes([0xFF] * 7 + [0xFC]))

    def test_build_disable_frame_for_mit_mode(self):
        frame_id, payload = script.build_disable_frame(0x01)
        self.assertEqual(frame_id, 0x001)
        self.assertEqual(payload, bytes([0xFF] * 7 + [0xFD]))

    def test_send_repeated_frame_replays_same_frame_count_times(self):
        transport = mock.Mock()
        frame = (0x001, bytes([0xFF] * 7 + [0xFC]))

        with mock.patch.object(script.time, "sleep") as sleep:
            script.send_repeated_frame(transport, frame, count=3, interval_seconds=0.002)

        self.assertEqual(transport.send.call_args_list, [mock.call(0x001, frame[1]), mock.call(0x001, frame[1]), mock.call(0x001, frame[1])])
        self.assertEqual(sleep.call_args_list, [mock.call(0.002), mock.call(0.002), mock.call(0.002)])

    def test_send_frame_retries_after_enobufs(self):
        transport = mock.Mock()
        frame = (0x001, bytes([0xFF] * 7 + [0xFC]))
        transport.send.side_effect = [
            OSError(script.errno.ENOBUFS, "No buffer space available"),
            None,
        ]

        with mock.patch.object(script.time, "sleep") as sleep:
            script.send_frame(transport, frame)

        self.assertEqual(transport.send.call_args_list, [mock.call(0x001, frame[1]), mock.call(0x001, frame[1])])
        sleep.assert_called_once_with(script.DEFAULT_BACKPRESSURE_SLEEP)

    def test_log_feedback_to_rerun_logs_expected_paths(self):
        feedback = mock.Mock(
            position=1.0,
            velocity=2.0,
            torque=3.0,
            state_code=1,
            controller_id=1,
            mos_temp=40.0,
            rotor_temp=45.0,
        )

        with mock.patch.object(script.rr, "set_time") as set_time, mock.patch.object(
            script.rr, "log"
        ) as log:
            script.log_feedback_to_rerun("/motor/dm8009_id_01", 1.5, feedback)

        set_time.assert_called_once_with("feedback_time", duration=1.5)
        logged_paths = [call.args[0] for call in log.call_args_list]
        self.assertEqual(
            logged_paths,
            [
                "/motor/dm8009_id_01/position",
                "/motor/dm8009_id_01/velocity",
                "/motor/dm8009_id_01/torque",
                "/motor/dm8009_id_01/state_code",
                "/motor/dm8009_id_01/mos_temp",
                "/motor/dm8009_id_01/rotor_temp",
                "/motor/dm8009_id_01/events",
            ],
        )

    def test_log_send_rate_to_rerun_logs_expected_path(self):
        with mock.patch.object(script.rr, "set_time") as set_time, mock.patch.object(script.rr, "log") as log:
            script.log_send_rate_to_rerun("/motor/dm8009_id_01", 2.5, 1234.0)

        set_time.assert_called_once_with("feedback_time", duration=2.5)
        self.assertEqual(log.call_args[0][0], "/motor/dm8009_id_01/send_rate_hz")


if __name__ == "__main__":
    unittest.main()
