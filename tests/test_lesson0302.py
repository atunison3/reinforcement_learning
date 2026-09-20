"""Checks for the goals-and-rewards lesson's deterministic examples."""

import unittest

from reinforcement_learning.lessons.lesson0302 import (
    Action,
    Can,
    RecyclingEnvironment,
    RewardRule,
    State,
    recycle_policy,
    repeat_pickup_policy,
    run_episode,
)


class TestGoalsAndRewards(unittest.TestCase):
    def test_documented_returns_and_actual_outcomes(self) -> None:
        cases = (
            (RewardRule.RECYCLING, recycle_policy, (0, 1, 0, 0, 0, 0), Can.RECYCLED),
            (RewardRule.RECYCLING, repeat_pickup_policy, (0, 0, 0, 0, 0, 0), Can.ON_FLOOR),
            (RewardRule.PICKUP, recycle_policy, (1, 0, 0, 0, 0, 0), Can.RECYCLED),
            (RewardRule.PICKUP, repeat_pickup_policy, (1, 0, 1, 0, 1, 0), Can.ON_FLOOR),
        )
        for rule, policy, rewards, final_can in cases:
            with self.subTest(rule=rule, policy=policy.__name__):
                trajectory = run_episode(RecyclingEnvironment(rule), policy)
                self.assertEqual(tuple(step.reward for step in trajectory), rewards)
                self.assertEqual(trajectory[-1].next_state, State(final_can, 0))

    def test_changing_reward_rule_leaves_physical_transitions_unchanged(self) -> None:
        for policy in (recycle_policy, repeat_pickup_policy):
            with self.subTest(policy=policy.__name__):
                outcome = run_episode(RecyclingEnvironment(RewardRule.RECYCLING), policy)
                proxy = run_episode(RecyclingEnvironment(RewardRule.PICKUP), policy)
                self.assertEqual(
                    [(step.state, step.action, step.next_state) for step in outcome],
                    [(step.state, step.action, step.next_state) for step in proxy],
                )

    def test_one_decision_is_not_enough_to_recycle(self) -> None:
        for rule, reward in ((RewardRule.RECYCLING, 0), (RewardRule.PICKUP, 1)):
            for policy in (recycle_policy, repeat_pickup_policy):
                with self.subTest(rule=rule, policy=policy.__name__):
                    trajectory = run_episode(RecyclingEnvironment(rule, horizon=1), policy)
                    self.assertEqual(len(trajectory), 1)
                    self.assertEqual(trajectory[0].reward, reward)
                    self.assertEqual(trajectory[0].next_state, State(Can.HELD, 0))

    def test_eight_decisions_allow_four_repeated_pickup_rewards(self) -> None:
        trajectory = run_episode(RecyclingEnvironment(RewardRule.PICKUP, horizon=8), repeat_pickup_policy)
        self.assertEqual(sum(step.reward for step in trajectory), 4)
        self.assertEqual(trajectory[-1].next_state.can, Can.ON_FLOOR)

    def test_invalid_action_and_post_terminal_action_are_rejected(self) -> None:
        environment = RecyclingEnvironment(RewardRule.RECYCLING, horizon=1)
        with self.assertRaises(ValueError):
            environment.step(Action.RECYCLE)
        self.assertEqual(environment.state, State(Can.ON_FLOOR, 1))
        environment.step(Action.PICK_UP)
        self.assertEqual(environment.available_actions(environment.state), ())
        with self.assertRaises(ValueError):
            environment.step(Action.WAIT)

    def test_recycled_can_cannot_be_retrieved_or_rewarded_again(self) -> None:
        environment = RecyclingEnvironment(RewardRule.RECYCLING)
        environment.step(Action.PICK_UP)
        self.assertEqual(environment.step(Action.RECYCLE).reward, 1)
        self.assertEqual(environment.available_actions(environment.state), (Action.WAIT,))
        for action in (Action.PICK_UP, Action.PUT_DOWN, Action.RECYCLE):
            with self.subTest(action=action), self.assertRaises(ValueError):
                environment.step(action)
        self.assertEqual(environment.step(Action.WAIT).reward, 0)
        self.assertEqual(environment.state.can, Can.RECYCLED)

    def test_each_episode_resets_the_same_environment(self) -> None:
        environment = RecyclingEnvironment(RewardRule.RECYCLING)
        first = run_episode(environment, recycle_policy)
        second = run_episode(environment, recycle_policy)
        self.assertEqual(first, second)
        self.assertEqual(environment.reset(), State(Can.ON_FLOOR, 6))

    def test_nonpositive_horizons_are_rejected(self) -> None:
        for horizon in (0, -1):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                RecyclingEnvironment(RewardRule.RECYCLING, horizon=horizon)


if __name__ == "__main__":
    unittest.main()
