import unittest
from unittest.mock import patch

import numpy as np

from reinforcement_learning.policies import epsilon_greedy, upper_confidence_bound


class TestEpsilonGreedy(unittest.TestCase):
    def test_epsilon_greedy_explores_by_choosing_random_action(self) -> None:
        values = np.array([1.0, 2.0, 3.0])

        with (
            patch("reinforcement_learning.policies.np.random.random", return_value=0.05),
            patch("reinforcement_learning.policies.np.random.choice", return_value=1) as mock_choice,
        ):
            action = epsilon_greedy(values, epsilon=0.1)

        self.assertEqual(action, 1)
        mock_choice.assert_called_once_with(len(values))

    def test_epsilon_greedy_exploits_by_choosing_among_best_actions(self) -> None:
        values = np.array([1.0, 5.0, 5.0, 2.0])

        with (
            patch("reinforcement_learning.policies.np.random.random", return_value=0.5),
            patch("reinforcement_learning.policies.np.random.choice", return_value=2) as mock_choice,
        ):
            action = epsilon_greedy(values, epsilon=0.1)

        self.assertEqual(action, 2)
        np.testing.assert_array_equal(mock_choice.call_args.args[0], np.array([1, 2]))

    def test_epsilon_greedy_rejects_non_1d_values(self) -> None:
        values = np.array([[1.0, 2.0], [3.0, 4.0]])

        with self.assertRaises(ValueError):
            epsilon_greedy(values, epsilon=0.1)

    def test_epsilon_greedy_rejects_invalid_epsilon(self) -> None:
        values = np.array([1.0, 2.0, 3.0])

        for epsilon in (-0.1, 1.1):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                epsilon_greedy(values, epsilon=epsilon)

    def test_epsilon_greedy_does_not_explore_epsilon_0(self) -> None:

        values = np.array([1.0, 2.0, 3.0])
        action = epsilon_greedy(values, epsilon=0.0)
        self.assertEqual(action, 2)

    def test_edge_case_epsilon_1(self) -> None:

        values = np.array([1.0, 2.0, 3.0, 4.0])

        n_trials = 20_000
        actions = [epsilon_greedy(values, epsilon=1.0) for _ in range(n_trials)]
        action_0_probability = actions.count(0) / n_trials

        self.assertAlmostEqual(
            action_0_probability,
            0.25,
            delta=0.02,
        )


class TestUpperConfidenceBound(unittest.TestCase):
    def test_ucb_chooses_untried_action_first(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        counts = np.array([5, 0, 0])

        with patch("reinforcement_learning.policies.np.random.choice", return_value=2) as mock_choice:
            action = upper_confidence_bound(values, counts, step=10, c=2.0)

        self.assertEqual(action, 2)
        np.testing.assert_array_equal(mock_choice.call_args.args[0], np.array([1, 2]))

    def test_ucb_chooses_action_with_largest_upper_confidence_bound(self) -> None:
        values = np.array([1.0, 1.5, 1.2])
        counts = np.array([10, 2, 5])

        with patch("reinforcement_learning.policies.np.random.choice", side_effect=lambda actions: actions[0]):
            action = upper_confidence_bound(values, counts, step=20, c=2.0)

        bonuses = 2.0 * np.sqrt(np.log(20) / counts)
        expected_action = int(np.argmax(values + bonuses))
        self.assertEqual(action, expected_action)

    def test_ucb_breaks_ties_among_best_actions(self) -> None:
        values = np.array([1.0, 1.0])
        counts = np.array([4, 4])

        with patch("reinforcement_learning.policies.np.random.choice", return_value=1) as mock_choice:
            action = upper_confidence_bound(values, counts, step=10, c=1.0)

        self.assertEqual(action, 1)
        np.testing.assert_array_equal(mock_choice.call_args.args[0], np.array([0, 1]))

    def test_ucb_rejects_invalid_shapes(self) -> None:
        with self.assertRaises(ValueError):
            upper_confidence_bound(np.array([[1.0, 2.0]]), np.array([1, 2]), step=1, c=1.0)

        with self.assertRaises(ValueError):
            upper_confidence_bound(np.array([1.0, 2.0]), np.array([[1, 2]]), step=1, c=1.0)

    def test_ucb_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            upper_confidence_bound(np.array([1.0, 2.0]), np.array([1]), step=1, c=1.0)

    def test_ucb_rejects_invalid_step(self) -> None:
        with self.assertRaises(ValueError):
            upper_confidence_bound(np.array([1.0, 2.0]), np.array([1, 1]), step=0, c=1.0)

    def test_ucb_rejects_negative_c(self) -> None:
        with self.assertRaises(ValueError):
            upper_confidence_bound(np.array([1.0, 2.0]), np.array([1, 1]), step=1, c=-0.5)


if __name__ == "__main__":
    unittest.main()
