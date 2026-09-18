"""Algorithm updates and Figure-2.6-style blackjack parameter sweeps."""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from reinforcement_learning.blackjack_twelve import TwelveRound
from reinforcement_learning.playground.exercise008 import (
    METHODS,
    PARAMETERS,
    BanditBatch,
    plot_results,
    run_experiments,
    save_results,
)


class TestBanditBatch(unittest.TestCase):
    def test_sample_average_updates_only_selected_arms(self) -> None:
        agent = BanditBatch(2, "epsilon_greedy", 0.1, np.random.default_rng(0))
        agent.update(np.array([0, 1]), np.array([1.0, -1.0]))
        agent.update(np.array([0, 0]), np.array([-1.0, 1.0]))
        np.testing.assert_array_equal(agent.q, [[0.0, 0.0], [1.0, -1.0]])
        np.testing.assert_array_equal(agent.counts, [[2, 0], [1, 1]])

    def test_optimistic_greedy_uses_fixed_point_one_step_size(self) -> None:
        agent = BanditBatch(1, "optimistic", 4.0, np.random.default_rng(0))
        np.testing.assert_array_equal(agent.q, [[4.0, 4.0]])
        agent.update(np.array([0]), np.array([-1.0]))
        np.testing.assert_array_equal(agent.q, [[3.5, 4.0]])
        self.assertEqual(int(agent.choose_actions()[0]), 1)

    def test_gradient_uses_softmax_and_preceding_reward_baseline(self) -> None:
        agent = BanditBatch(1, "gradient", 0.4, np.random.default_rng(0))
        np.testing.assert_array_equal(agent.probabilities(), [[0.5, 0.5]])
        agent.update(np.array([0]), np.array([1.0]))
        np.testing.assert_allclose(agent.preferences, [[0.2, -0.2]])
        self.assertEqual(agent.baseline[0], 1.0)
        before = agent.preferences.copy()
        agent.update(np.array([1]), np.array([1.0]))
        np.testing.assert_array_equal(agent.preferences, before)  # Zero advantage.
        agent.preferences[:] = [[1000.0, 1000.0]]
        np.testing.assert_array_equal(agent.probabilities(), [[0.5, 0.5]])

    def test_ucb_tries_both_arms_then_uses_confidence_bonus(self) -> None:
        agent = BanditBatch(20, "ucb", 2.0, np.random.default_rng(0))
        first = agent.choose_actions()
        agent.update(first, np.ones(20))
        second = agent.choose_actions()
        np.testing.assert_array_equal(second, 1 - first)
        agent.update(second, np.ones(20))
        agent.q[:] = [[0.0, 0.1]]
        agent.counts[:] = [[1, 10]]
        agent.steps = 11
        np.testing.assert_array_equal(agent.choose_actions(), np.zeros(20, dtype=np.int64))

    def test_full_epsilon_exploration_does_not_follow_greedy_values(self) -> None:
        agent = BanditBatch(1000, "epsilon_greedy", 1.0, np.random.default_rng(0))
        agent.q[:, 0] = 100.0
        actions = agent.choose_actions()
        self.assertGreater(int(np.sum(actions == 1)), 400)
        self.assertLess(int(np.sum(actions == 1)), 600)

    def test_rejects_invalid_settings_and_updates(self) -> None:
        for method in METHODS:
            for value in (0.0, -1.0, np.nan, np.inf):
                with self.subTest(method=method, value=value), self.assertRaises(ValueError):
                    BanditBatch(1, method, value, np.random.default_rng(0))
        for method in ("epsilon_greedy", "gradient"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                BanditBatch(1, method, 2.0, np.random.default_rng(0))
        agent = BanditBatch(1, "ucb", 1.0, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            agent.update(np.array([2]), np.array([1.0]))
        with self.assertRaises(ValueError):
            agent.update(np.array([0]), np.array([np.nan]))
        with self.assertRaises(ValueError):
            agent.update(np.array([0, 1]), np.array([1.0]))


class TestBlackjackParameterStudy(unittest.TestCase):
    def test_sweep_is_reproducible_and_caps_epsilon_and_alpha(self) -> None:
        first = run_experiments(trials=3, steps=8, seed=7)
        second = run_experiments(trials=3, steps=8, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first.points), 36)
        self.assertEqual((first.trials, first.steps), (3, 8))
        for method in METHODS:
            points = [point for point in first.points if point.method == method]
            expected = PARAMETERS[:8] if method in ("epsilon_greedy", "gradient") else PARAMETERS
            self.assertEqual(tuple(point.parameter for point in points), expected)
            self.assertTrue(all(-1 <= point.average_reward <= 1 for point in points))
            self.assertTrue(all(point.standard_error >= 0 for point in points))

    def test_single_trial_standard_error_is_zero(self) -> None:
        result = run_experiments(trials=1, steps=2, parameters=(1.0,))
        self.assertTrue(all(point.standard_error == 0.0 for point in result.points))

    def test_results_are_average_over_all_steps_and_trials(self) -> None:
        loss = TwelveRound((10, 2), (10, 10), np.array([10]))
        win = TwelveRound((1, 1), (10, 6), np.array([10, 10]))
        # Each of two trials gets rewards [-1, +1, -1], regardless of actions.
        with patch(
            "reinforcement_learning.playground.exercise008.TwoDeckTwelveBandit.deal",
            side_effect=[loss, loss, win, win, loss, loss],
        ):
            result = run_experiments(trials=2, steps=3, parameters=(1.0,))
        for point in result.points:
            self.assertAlmostEqual(point.average_reward, -1 / 3)
            self.assertEqual(point.standard_error, 0.0)
        self.assertEqual(result.action_means, (-1 / 3, -1 / 3))

    def test_export_and_parameter_plot(self) -> None:
        result = run_experiments(trials=2, steps=4)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "nested"
            path = root / "results.csv"
            save_results(result, path)
            plot_results(result, root / "plot.png")
            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 36)
            self.assertTrue(all(row["steps"] == "4" and row["trials"] == "2" for row in rows))
            self.assertGreater((root / "plot.png").stat().st_size, 0)

    def test_rejects_invalid_experiment_parameters(self) -> None:
        for trials, steps in ((0, 1), (1, 0), (-1, 1)):
            with self.subTest(trials=trials, steps=steps), self.assertRaises(ValueError):
                run_experiments(trials=trials, steps=steps)
        for parameters in ((), (0.0,), (8.0,), (np.nan,), (1.0, 0.5), (1.0, 1.0)):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                run_experiments(trials=1, steps=1, parameters=parameters)


if __name__ == "__main__":
    unittest.main()
