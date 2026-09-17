"""Small integration tests for one-decision blackjack training and exports."""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from reinforcement_learning.blackjack import N_CONTEXTS, Action, reachable_states
from reinforcement_learning.playground.exercise008 import plot_policy, save_action_values, train


class TestBlackjackAssociativeSearch(unittest.TestCase):
    def test_training_is_reproducible_with_one_legal_update_per_round(self) -> None:
        first = train(rounds=200, seed=7)
        second = train(rounds=200, seed=7)
        np.testing.assert_array_equal(first.q, second.q)
        np.testing.assert_array_equal(first.counts, second.counts)
        self.assertEqual(int(np.sum(first.counts)), 200)
        self.assertEqual(first.q.shape, (N_CONTEXTS, 3))
        self.assertTrue(np.all(np.isfinite(first.q)))
        self.assertTrue(np.all((first.q >= -2) & (first.q <= 2)))
        reachable = {state.index for state in reachable_states()}
        for index in range(N_CONTEXTS):
            if index not in reachable:
                self.assertEqual(int(np.sum(first.counts[index])), 0)
        for state in reachable_states():
            if not state.can_double:
                self.assertEqual(first.counts[state.index, Action.DOUBLE], 0)

    def test_export_includes_counts_and_blanks_for_unsampled_legal_pairs(self) -> None:
        agent = train(rounds=200)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "values.csv"
            save_action_values(agent, path)
            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 1310)
        self.assertEqual(sum(int(row["visits"]) for row in rows), 200)
        for row in rows:
            with self.subTest(row=row):
                self.assertEqual(row["estimated_value"] == "", row["visits"] == "0")
                self.assertFalse(row["can_double"] == "False" and row["action"] == "DOUBLE")
                self.assertGreaterEqual(int(row["hand_value"]), 4)

    def test_plotting_creates_policy_image(self) -> None:
        agent = train(rounds=20)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "policy.png"
            plot_policy(agent, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_invalid_training_parameters(self) -> None:
        for rounds in (0, -1):
            with self.subTest(rounds=rounds), self.assertRaises(ValueError):
                train(rounds=rounds)
        for epsilon in (-0.1, 1.1, np.nan):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                train(rounds=1, epsilon=epsilon)


if __name__ == "__main__":
    unittest.main()
