"""Tests for contextual value estimates and information-value benchmarks."""

import unittest
from unittest.mock import patch

import numpy as np

from reinforcement_learning.contextual_bandits import ContextualBanditAgent, optimal_expected_rewards


class TestOptimalExpectedRewards(unittest.TestCase):
    def test_exercise_2_10_optima(self) -> None:
        values = np.array([[10.0, 20.0], [90.0, 80.0]])
        self.assertEqual(optimal_expected_rewards(values, np.array([0.5, 0.5])), (50.0, 55.0))

    def test_unequal_context_probabilities(self) -> None:
        values = np.array([[10.0, 20.0], [90.0, 80.0]])
        self.assertEqual(optimal_expected_rewards(values, np.array([0.75, 0.25])), (35.0, 37.5))

    def test_context_has_no_value_when_one_action_is_best_everywhere(self) -> None:
        values = np.array([[3.0, 1.0], [9.0, 7.0]])
        self.assertEqual(optimal_expected_rewards(values, np.array([0.5, 0.5])), (6.0, 6.0))

    def test_single_context_and_zero_probability_context(self) -> None:
        self.assertEqual(optimal_expected_rewards(np.array([[-2.0, -1.0]]), np.array([1.0])), (-1.0, -1.0))
        values = np.array([[10.0, 20.0], [90.0, 80.0]])
        self.assertEqual(optimal_expected_rewards(values, np.array([1.0, 0.0])), (20.0, 20.0))

    def test_rejects_invalid_action_values(self) -> None:
        for values in (np.array([1.0, 2.0]), np.empty((0, 2)), np.empty((2, 0)), np.array([[np.nan, 1.0]])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                optimal_expected_rewards(values, np.array([1.0]))

    def test_rejects_invalid_context_probabilities(self) -> None:
        values = np.array([[10.0, 20.0], [90.0, 80.0]])
        for probabilities in (
            np.array([[0.5, 0.5]]),
            np.array([1.0]),
            np.array([-0.5, 1.5]),
            np.array([0.2, 0.3]),
            np.array([np.nan, 0.5]),
            np.array([np.inf, 0.0]),
        ):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                optimal_expected_rewards(values, probabilities)


class TestContextualBanditAgent(unittest.TestCase):
    def test_sample_averages_and_counts_are_separate_for_each_pair(self) -> None:
        agent = ContextualBanditAgent(n_contexts=2, n_actions=2)
        agent.update(context=0, action=1, reward=10.0)
        agent.update(context=0, action=1, reward=20.0)
        agent.update(context=1, action=1, reward=80.0)

        np.testing.assert_array_equal(agent.q, [[0.0, 15.0], [0.0, 80.0]])
        np.testing.assert_array_equal(agent.counts, [[0, 2], [0, 1]])

    def test_greedy_policy_depends_on_context(self) -> None:
        agent = ContextualBanditAgent(n_contexts=2, n_actions=2, epsilon=0.0)
        for context, rewards in enumerate(((10.0, 20.0), (90.0, 80.0))):
            for action, reward in enumerate(rewards):
                agent.update(context, action, reward)

        self.assertEqual(agent.choose_action(0), 1)
        self.assertEqual(agent.choose_action(1), 0)

    def test_action_selection_reuses_epsilon_greedy_with_only_current_row(self) -> None:
        agent = ContextualBanditAgent(n_contexts=2, n_actions=2, epsilon=0.2)
        agent.update(1, 0, 90.0)
        with patch("reinforcement_learning.contextual_bandits.epsilon_greedy", return_value=1) as choose:
            self.assertEqual(agent.choose_action(1), 1)
        choose.assert_called_once()
        np.testing.assert_array_equal(choose.call_args.args[0], [90.0, 0.0])
        self.assertEqual(choose.call_args.args[1], 0.2)

    def test_one_context_pools_rewards_as_an_ordinary_bandit(self) -> None:
        agent = ContextualBanditAgent(n_contexts=1, n_actions=2)
        agent.update(0, 0, 10.0)
        agent.update(0, 0, 90.0)
        self.assertEqual(agent.q[0, 0], 50.0)

    def test_agents_have_independent_tables(self) -> None:
        first = ContextualBanditAgent(2, 2)
        second = ContextualBanditAgent(2, 2)
        first.update(0, 0, 10.0)
        np.testing.assert_array_equal(second.q, np.zeros((2, 2)))

    def test_rejects_invalid_configuration(self) -> None:
        for n_contexts, n_actions in ((0, 2), (2, 0), (-1, 2), (2, -1)):
            with self.subTest(n_contexts=n_contexts, n_actions=n_actions), self.assertRaises(ValueError):
                ContextualBanditAgent(n_contexts, n_actions)
        for epsilon in (-0.1, 1.1, np.nan):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                ContextualBanditAgent(2, 2, epsilon)

    def test_rejects_invalid_observations_without_mutating_tables(self) -> None:
        agent = ContextualBanditAgent(2, 2)
        for context in (-1, 2):
            with self.subTest(context=context):
                with self.assertRaises(ValueError):
                    agent.choose_action(context)
                with self.assertRaises(ValueError):
                    agent.update(context, 0, 1.0)
        for action in (-1, 2):
            with self.subTest(action=action), self.assertRaises(ValueError):
                agent.update(0, action, 1.0)
        for reward in (np.nan, np.inf):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                agent.update(0, 0, reward)
        np.testing.assert_array_equal(agent.counts, np.zeros((2, 2)))
        np.testing.assert_array_equal(agent.q, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
