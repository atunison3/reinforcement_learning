"""Small reproducible integration tests for the associative-search project."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from reinforcement_learning.projects.project007 import plot_results, run_experiments


class TestAssociativeSearchProject(unittest.TestCase):
    def test_curves_are_reproducible_and_have_requested_length(self) -> None:
        first = run_experiments(trials=3, steps=10, seed=7)
        second = run_experiments(trials=3, steps=10, seed=7)
        self.assertEqual(set(first), {"Without context", "With context"})
        for name, curves in first.items():
            with self.subTest(name=name):
                self.assertEqual(curves.average_reward.shape, (10,))
                self.assertEqual(curves.optimal_action_percentage.shape, (10,))
                self.assertTrue(np.all(np.isfinite(curves.average_reward)))
                self.assertTrue(
                    np.all((curves.optimal_action_percentage >= 0) & (curves.optimal_action_percentage <= 100))
                )
                np.testing.assert_array_equal(curves.average_reward, second[name].average_reward)
                np.testing.assert_array_equal(curves.optimal_action_percentage, second[name].optimal_action_percentage)

    def test_contextual_learner_improves_over_hidden_case_learner(self) -> None:
        results = run_experiments(trials=100, steps=400, reward_std=0.0, seed=42)
        contextual = results["With context"]
        hidden = results["Without context"]
        self.assertGreater(float(np.mean(contextual.average_reward[-100:])), 53.0)
        self.assertAlmostEqual(float(np.mean(hidden.average_reward[-100:])), 50.0, delta=1.0)
        self.assertAlmostEqual(float(np.mean(contextual.optimal_action_percentage[-100:])), 95.0, delta=2.0)
        self.assertAlmostEqual(float(np.mean(hidden.optimal_action_percentage[-100:])), 50.0, delta=2.0)

    def test_plotting_writes_both_figures_with_partial_final_block(self) -> None:
        results = run_experiments(trials=2, steps=7)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "assets"
            plot_results(results, output_directory=output, block_size=5)
            for name in ("project_7_average_reward.png", "project_7_optimal_action_percentage.png"):
                image = output / name
                self.assertTrue(image.is_file())
                self.assertGreater(image.stat().st_size, 0)

    def test_rejects_invalid_experiment_parameters(self) -> None:
        for trials, steps in ((0, 1), (1, 0), (-1, 1), (1, -1)):
            with self.subTest(trials=trials, steps=steps), self.assertRaises(ValueError):
                run_experiments(trials=trials, steps=steps)
        for epsilon in (-0.1, 1.1, np.nan):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                run_experiments(trials=1, steps=1, epsilon=epsilon)
        for reward_std in (-1.0, np.nan, np.inf):
            with self.subTest(reward_std=reward_std), self.assertRaises(ValueError):
                run_experiments(trials=1, steps=1, reward_std=reward_std)


if __name__ == "__main__":
    unittest.main()
