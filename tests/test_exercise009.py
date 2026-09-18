"""Reproducibility, baseline traces, exports, and exercise demonstrations."""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from reinforcement_learning.playground.exercise009 import (
    CONFIGURATIONS,
    REWARD_OFFSETS,
    plot_learning,
    plot_variance_diagnostic,
    reward_shift_demo,
    run_experiments,
    save_summary,
)


class TestGradientBaselineExercise(unittest.TestCase):
    def test_reproducible_curves_and_fixed_baseline_traces(self) -> None:
        first = run_experiments(trials=5, steps=10, seed=7)
        second = run_experiments(trials=5, steps=10, seed=7)
        self.assertEqual(len(first.curves), 10)
        for offset in REWARD_OFFSETS:
            for config in CONFIGURATIONS:
                with self.subTest(offset=offset, name=config.name):
                    curve = first.curves[offset, config.name]
                    repeated = second.curves[offset, config.name]
                    self.assertEqual(curve.average_reward.shape, (10,))
                    np.testing.assert_array_equal(curve.average_reward, repeated.average_reward)
                    np.testing.assert_array_equal(curve.optimal_action_percentage, repeated.optimal_action_percentage)
                    np.testing.assert_array_equal(curve.baseline_before_update, repeated.baseline_before_update)
                    self.assertTrue(np.all(np.isfinite(curve.average_reward)))
                    self.assertTrue(
                        np.all((curve.optimal_action_percentage >= 0) & (curve.optimal_action_percentage <= 100))
                    )
                    self.assertEqual(curve.baseline_before_update[0], config.initial_value)
                    if config.kind == "fixed":
                        np.testing.assert_array_equal(curve.baseline_before_update, np.full(10, config.initial_value))
        self.assertAlmostEqual(first.optimal_expected_rewards[4.0] - first.optimal_expected_rewards[0.0], 4.0)

    def test_shift_demonstration_matches_entire_policy_trajectory(self) -> None:
        for shift in (-4.0, 0.0, 4.0):
            with self.subTest(shift=shift):
                self.assertLess(reward_shift_demo(shift=shift, steps=100), 1e-12)

    def test_summary_and_all_three_plots(self) -> None:
        result = run_experiments(trials=3, steps=7)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "assets"
            save_summary(result, root / "summary.csv")
            self.assertNotIn(b"\r", (root / "summary.csv").read_bytes())
            plot_learning(result, root)
            plot_variance_diagnostic(root / "variance.png")
            with (root / "summary.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row["final_window"] == "7" and row["trials"] == "3" for row in rows))
            for row in rows:
                self.assertEqual(row["mean_reward"], row["final_reward"])
            for filename in ("exercise_9_learning.png", "exercise_9_baselines.png", "variance.png"):
                self.assertGreater((root / filename).stat().st_size, 0)

    def test_invalid_experiment_inputs(self) -> None:
        for trials, steps in ((0, 1), (1, 0), (-1, 1)):
            with self.subTest(trials=trials, steps=steps), self.assertRaises(ValueError):
                run_experiments(trials=trials, steps=steps)
        with self.assertRaises(ValueError):
            run_experiments(trials=1, steps=1, alpha=0.0)
        with self.assertRaises(ValueError):
            run_experiments(trials=1, steps=1, beta=2.0)
        with self.assertRaises(ValueError):
            reward_shift_demo(steps=0)


if __name__ == "__main__":
    unittest.main()
