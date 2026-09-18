"""Preference updates, lagged baselines, and exact fixed-policy diagnostics."""

import unittest

import numpy as np

from reinforcement_learning.gradient_bandits import (
    BaselineKind,
    GradientBanditBatch,
    gradient_moments,
    minimum_variance_baseline,
)


class TestGradientBanditBatch(unittest.TestCase):
    def test_fixed_baseline_changes_sign_and_size_of_reinforcement(self) -> None:
        for baseline in (-4.0, 0.0, 4.0):
            with self.subTest(baseline=baseline):
                agent = GradientBanditBatch(1, baseline_kind="fixed", initial_baseline=baseline)
                agent.update(np.array([0]), np.array([2.0]))
                expected = np.full(10, -0.01 * (2.0 - baseline))
                expected[0] = 0.09 * (2.0 - baseline)
                np.testing.assert_allclose(agent.preferences[0], expected)
                self.assertAlmostEqual(float(np.sum(agent.preferences)), 0.0)
                self.assertEqual(agent.baseline[0], baseline)

    def test_running_means_are_independent_and_updated_after_preferences(self) -> None:
        agent = GradientBanditBatch(2, k=2, initial_baseline=5.0)
        agent.update(np.array([0, 1]), np.array([1.0, 9.0]))
        np.testing.assert_allclose(agent.preferences, [[-0.2, 0.2], [-0.2, 0.2]])
        np.testing.assert_array_equal(agent.baseline, [1.0, 9.0])
        agent.update(np.array([0, 1]), np.array([3.0, 5.0]))
        np.testing.assert_array_equal(agent.baseline, [2.0, 7.0])

    def test_ema_uses_separate_beta_and_old_baseline(self) -> None:
        agent = GradientBanditBatch(1, k=2, baseline_kind="ema", initial_baseline=2.0, beta=0.25)
        agent.update(np.array([0]), np.array([6.0]))
        np.testing.assert_allclose(agent.preferences, [[0.2, -0.2]])
        self.assertEqual(agent.baseline[0], 3.0)

    def test_zero_advantage_leaves_preferences_unchanged(self) -> None:
        agent = GradientBanditBatch(1, initial_baseline=2.0)
        agent.update(np.array([1]), np.array([2.0]))
        np.testing.assert_array_equal(agent.preferences, np.zeros((1, 10)))

    def test_softmax_is_stable_and_invariant_to_common_preference_shift(self) -> None:
        agent = GradientBanditBatch(2, k=2)
        agent.preferences[:] = [[1000.0, 1001.0], [-1000.0, -999.0]]
        probabilities = agent.probabilities()
        np.testing.assert_allclose(probabilities[0], probabilities[1])
        np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)
        self.assertTrue(np.all(np.isfinite(probabilities)))
        agent.preferences[:] = [[1000.0, -1000.0], [-1000.0, 1000.0]]
        np.testing.assert_array_equal(agent.choose_actions(np.random.default_rng(0)), [0, 1])

    def test_reward_and_baseline_shift_preserves_updates_for_all_baseline_kinds(self) -> None:
        kinds: tuple[BaselineKind, ...] = ("fixed", "average", "ema")
        for kind in kinds:
            with self.subTest(kind=kind):
                original = GradientBanditBatch(1, baseline_kind=kind)
                shifted = GradientBanditBatch(1, baseline_kind=kind, initial_baseline=4.0)
                for action, reward in ((0, 2.0), (1, -1.0), (0, 5.0)):
                    original.update(np.array([action]), np.array([reward]))
                    shifted.update(np.array([action]), np.array([reward + 4.0]))
                    np.testing.assert_allclose(original.preferences, shifted.preferences, atol=1e-14)
                    np.testing.assert_allclose(shifted.baseline, original.baseline + 4.0)

    def test_rejects_invalid_parameters_and_observations(self) -> None:
        with self.assertRaises(ValueError):
            GradientBanditBatch(0)
        with self.assertRaises(ValueError):
            GradientBanditBatch(1, k=1)
        for value in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(alpha=value), self.assertRaises(ValueError):
                GradientBanditBatch(1, alpha=value)
        for beta in (0.0, -0.1, 1.1, np.nan):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                GradientBanditBatch(1, beta=beta)
        with self.assertRaises(ValueError):
            GradientBanditBatch(1, initial_baseline=np.nan)
        agent = GradientBanditBatch(1)
        for action in (-1, 10):
            with self.subTest(action=action), self.assertRaises(ValueError):
                agent.update(np.array([action]), np.array([1.0]))
        with self.assertRaises(ValueError):
            agent.update(np.array([0]), np.array([np.nan]))
        with self.assertRaises(ValueError):
            agent.update(np.array([0, 1]), np.array([1.0]))
        self.assertEqual(agent.steps, 0)


class TestBaselineDiagnostics(unittest.TestCase):
    def test_moments_match_explicit_enumeration_and_mean_does_not_change(self) -> None:
        policy = np.array([0.2, 0.3, 0.5])
        values = np.array([-1.0, 0.5, 2.0])
        expected_gradient = np.array([-0.39, -0.135, 0.525])
        variances = []
        for baseline in (-4.0, 0.0, 4.0):
            samples = (values - baseline)[:, None] * (np.eye(3) - policy)
            enumerated_mean = policy @ samples
            enumerated_variance = float(policy @ np.sum((samples - enumerated_mean) ** 2, axis=1))
            mean, variance = gradient_moments(policy, values, baseline, reward_std=0.0)
            np.testing.assert_allclose(mean, expected_gradient)
            np.testing.assert_allclose(mean, enumerated_mean)
            self.assertAlmostEqual(variance, enumerated_variance)
            variances.append(variance)
        self.assertGreater(variances[0], variances[1])
        self.assertGreater(variances[2], variances[1])

    def test_reward_noise_adds_expected_score_weighted_variance(self) -> None:
        policy = np.array([0.5, 0.5])
        values = np.array([0.0, 1.0])
        _, noiseless = gradient_moments(policy, values, 0.5, reward_std=0.0)
        _, noisy = gradient_moments(policy, values, 0.5, reward_std=2.0)
        self.assertAlmostEqual(noisy - noiseless, 2.0)

    def test_variance_minimum_matches_mean_only_for_uniform_policy_in_example(self) -> None:
        values = np.linspace(-1.0, 1.0, 10)
        uniform = np.full(10, 0.1)
        self.assertAlmostEqual(minimum_variance_baseline(uniform, values), float(np.mean(values)))
        concentrated = np.array([0.01] * 9 + [0.91])
        optimal = minimum_variance_baseline(concentrated, values)
        mean_reward = float(concentrated @ values)
        self.assertAlmostEqual(optimal, -0.057894736842105, places=12)
        self.assertAlmostEqual(mean_reward, 0.9)
        _, minimum = gradient_moments(concentrated, values, optimal)
        for other in (optimal - 0.1, optimal + 0.1, mean_reward):
            self.assertLess(minimum, gradient_moments(concentrated, values, other)[1])

    def test_deterministic_policy_has_zero_gradient_and_variance(self) -> None:
        policy = np.array([1.0, 0.0])
        values = np.array([3.0, 10.0])
        self.assertEqual(minimum_variance_baseline(policy, values), 3.0)
        mean, variance = gradient_moments(policy, values, -100.0)
        np.testing.assert_array_equal(mean, [0.0, 0.0])
        self.assertEqual(variance, 0.0)

    def test_invalid_diagnostic_inputs(self) -> None:
        values = np.array([0.0, 1.0])
        for policy in (
            np.array([]),
            np.array([1.0]),
            np.array([-0.1, 1.1]),
            np.array([0.1, 0.1]),
            np.array([np.nan, 1.0]),
        ):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                minimum_variance_baseline(policy, values)
        with self.assertRaises(ValueError):
            gradient_moments(np.array([0.5, 0.5]), values, 0.0, reward_std=-1.0)


if __name__ == "__main__":
    unittest.main()
