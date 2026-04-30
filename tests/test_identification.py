from __future__ import annotations

import unittest

import numpy as np

from friction_identification_core.core import piecewise_static_linear_level, piecewise_static_linear_torque
from friction_identification_core.identification import (
    _robust_refit_mask,
    fit_dynamic_motor_model,
    fit_weighted_dynamic_motor_model,
)


class DynamicIdentificationTests(unittest.TestCase):
    def test_robust_refit_mask_rejects_spike_when_mad_is_zero(self) -> None:
        design = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        target = np.asarray([1.0, 2.0, 3.0, 1.0], dtype=np.float64)
        mask = np.ones(target.size, dtype=bool)

        fit_mask, rejected_mask = _robust_refit_mask(design, target, mask)

        self.assertEqual(fit_mask.tolist(), [True, True, True, False])
        self.assertEqual(rejected_mask.tolist(), [False, False, False, True])

    def test_dynamic_fit_rejects_single_torque_spike_when_mad_collapses(self) -> None:
        velocity = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, -2.5, 2.5, -1.5, 1.5], dtype=np.float64)
        acceleration = np.asarray([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 1.2, -1.2, 0.8, -0.8], dtype=np.float64)
        true_j = 0.04
        true_viscous = 0.06
        true_tau_c = 0.18
        true_tau_bias = 0.02
        torque = (
            true_j * acceleration
            + true_viscous * velocity
            + true_tau_c * np.sign(velocity)
            + true_tau_bias
        )
        torque[4] += 2.0
        mask = np.ones(velocity.size, dtype=bool)

        result = fit_dynamic_motor_model(
            velocity,
            acceleration,
            torque,
            train_mask=mask,
            valid_mask=mask,
        )

        self.assertEqual(result.metadata["status"], "ok")
        self.assertFalse(bool(result.train_mask[4]))
        self.assertGreaterEqual(int(result.metadata["rejected_train_sample_count"]), 1)
        self.assertAlmostEqual(result.inertia, true_j, delta=0.006)
        self.assertAlmostEqual(result.viscous, true_viscous, delta=0.012)
        self.assertAlmostEqual(result.tau_c, true_tau_c, delta=0.02)
        self.assertAlmostEqual(result.tau_bias, true_tau_bias, delta=0.012)

    def test_dynamic_fit_recovers_synthetic_parameters(self) -> None:
        time_s = np.linspace(0.0, 8.0, 800, dtype=np.float64)
        velocity = 2.4 * np.sin(2.0 * np.pi * 0.35 * time_s) + 0.8 * np.sin(2.0 * np.pi * 0.9 * time_s)
        acceleration = np.gradient(velocity, time_s, edge_order=1)
        true_j = 0.035
        true_viscous = 0.08
        true_tau_c = 0.22
        true_tau_bias = -0.035
        torque = (
            true_j * acceleration
            + true_viscous * velocity
            + true_tau_c * np.sign(velocity)
            + true_tau_bias
        )
        torque += 0.004 * np.sin(2.0 * np.pi * 3.0 * time_s)

        train_mask = time_s < 5.5
        valid_mask = ~train_mask
        result = fit_dynamic_motor_model(
            velocity,
            acceleration,
            torque,
            train_mask=train_mask,
            valid_mask=valid_mask,
        )

        self.assertAlmostEqual(result.inertia, true_j, delta=0.006)
        self.assertAlmostEqual(result.viscous, true_viscous, delta=0.015)
        self.assertAlmostEqual(result.tau_c, true_tau_c, delta=0.025)
        self.assertAlmostEqual(result.tau_bias, true_tau_bias, delta=0.015)
        self.assertLess(result.valid_rmse, 0.03)

    def test_dynamic_fit_enforces_non_negative_physical_terms(self) -> None:
        velocity = np.asarray([-2.0, -1.0, 1.0, 2.0, -2.0, -1.0, 1.0, 2.0], dtype=np.float64)
        acceleration = np.asarray([-1.0, -0.5, 0.5, 1.0, -1.2, -0.4, 0.4, 1.2], dtype=np.float64)
        torque = -0.1 * acceleration - 0.2 * velocity - 0.3 * np.sign(velocity)
        mask = np.ones(velocity.size, dtype=bool)

        result = fit_dynamic_motor_model(
            velocity,
            acceleration,
            torque,
            train_mask=mask,
            valid_mask=mask,
        )

        self.assertGreaterEqual(result.inertia, 0.0)
        self.assertGreaterEqual(result.viscous, 0.0)
        self.assertGreaterEqual(result.tau_c, 0.0)

    def test_weighted_joint_dynamic_fit_recovers_mixed_source_parameters(self) -> None:
        time_s = np.linspace(0.0, 6.0, 600, dtype=np.float64)
        dynamic_velocity = 1.8 * np.sin(2.0 * np.pi * 0.4 * time_s)
        dynamic_acceleration = np.gradient(dynamic_velocity, time_s, edge_order=1)
        hold_velocity = np.asarray([-4.0, -2.0, 2.0, 4.0, -6.0, 6.0], dtype=np.float64)
        hold_acceleration = np.zeros_like(hold_velocity)

        true_j = 0.042
        true_viscous = 0.055
        true_tau_c = 0.18
        true_tau_bias = 0.025
        dynamic_torque = (
            true_j * dynamic_acceleration
            + true_viscous * dynamic_velocity
            + true_tau_c * np.sign(dynamic_velocity)
            + true_tau_bias
        )
        hold_torque = true_viscous * hold_velocity + true_tau_c * np.sign(hold_velocity) + true_tau_bias

        velocity = np.concatenate([hold_velocity, dynamic_velocity])
        acceleration = np.concatenate([hold_acceleration, dynamic_acceleration])
        torque = np.concatenate([hold_torque, dynamic_torque])
        train_mask = np.ones(velocity.size, dtype=bool)
        valid_mask = np.zeros(velocity.size, dtype=bool)
        valid_mask[-100:] = True
        train_mask[-100:] = False
        sample_weight = np.concatenate([np.full(hold_velocity.size, 3.0), np.ones(dynamic_velocity.size)])

        result = fit_weighted_dynamic_motor_model(
            velocity,
            acceleration,
            torque,
            train_mask=train_mask,
            valid_mask=valid_mask,
            sample_weight=sample_weight,
        )

        self.assertEqual(result.metadata["status"], "ok")
        self.assertAlmostEqual(result.inertia, true_j, delta=0.006)
        self.assertAlmostEqual(result.viscous, true_viscous, delta=0.012)
        self.assertAlmostEqual(result.tau_c, true_tau_c, delta=0.02)
        self.assertAlmostEqual(result.tau_bias, true_tau_bias, delta=0.012)
        self.assertLess(result.valid_rmse, 0.02)

    def test_piecewise_static_linear_model_blends_static_to_coulomb_level(self) -> None:
        levels = piecewise_static_linear_level(
            np.asarray([0.0, 0.20, 0.35, 0.50, 1.0], dtype=np.float64),
            tau_static=0.80,
            tau_c=0.45,
            static_velocity_threshold_rad_s=0.20,
            static_transition_velocity_rad_s=0.50,
        )

        self.assertAlmostEqual(float(levels[0]), 0.80, places=6)
        self.assertAlmostEqual(float(levels[1]), 0.80, places=6)
        self.assertAlmostEqual(float(levels[2]), 0.625, places=6)
        self.assertAlmostEqual(float(levels[3]), 0.45, places=6)
        self.assertAlmostEqual(float(levels[4]), 0.45, places=6)

    def test_piecewise_static_linear_torque_uses_supplied_direction_at_zero_speed(self) -> None:
        torque = piecewise_static_linear_torque(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            acceleration=np.asarray([0.0, 2.0, -1.0], dtype=np.float64),
            direction=np.asarray([-1.0, 1.0, 1.0], dtype=np.float64),
            tau_static=0.60,
            tau_c=0.20,
            viscous=0.05,
            tau_bias=0.01,
            inertia=0.10,
            static_velocity_threshold_rad_s=0.20,
            static_transition_velocity_rad_s=0.50,
        )

        np.testing.assert_allclose(torque, np.asarray([-0.59, 0.81, 0.16], dtype=np.float64), atol=1.0e-9)


if __name__ == "__main__":
    unittest.main()
