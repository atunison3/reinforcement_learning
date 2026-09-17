"""Blackjack rules, usable aces, immediate rewards, and legal contexts."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from reinforcement_learning.blackjack import (
    N_CONTEXTS,
    Action,
    BlackjackState,
    OneDecisionBlackjack,
    draw_card,
    hand_value,
    reachable_states,
)


class TestBlackjackHelpers(unittest.TestCase):
    def test_hand_values_and_usable_aces(self) -> None:
        for cards, expected in (
            ([1, 4], (15, True)),
            ([1, 4, 10], (15, False)),
            ([1, 1], (12, True)),
            ([1, 1, 9], (21, True)),
            ([1, 1, 9, 10], (21, False)),
            ([10, 10, 2], (22, False)),
            ([2, 2], (4, False)),
        ):
            with self.subTest(cards=cards):
                self.assertEqual(hand_value(cards), expected)
        for cards in ([], [0, 10], [1, 11]):
            with self.subTest(cards=cards), self.assertRaises(ValueError):
                hand_value(cards)

    def test_draw_preserves_four_ten_valued_ranks(self) -> None:
        rng = Mock(spec=np.random.Generator)
        rng.integers.side_effect = range(1, 14)
        cards = [draw_card(rng) for _ in range(13)]
        self.assertEqual(cards, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
        rng.integers.assert_called_with(1, 14)

    def test_reachable_states_have_unique_indices_and_legal_actions(self) -> None:
        states = reachable_states()
        indices = {state.index for state in states}
        self.assertEqual(len(states), 520)
        self.assertEqual(len(indices), len(states))
        self.assertTrue(all(0 <= index < N_CONTEXTS for index in indices))
        self.assertEqual(sum(len(state.legal_actions) for state in states), 1310)
        self.assertTrue(all(state.hand_value >= 4 for state in states))
        self.assertNotIn(BlackjackState(21, False, 10, True), states)
        self.assertNotIn(BlackjackState(5, False, 10, False), states)
        self.assertNotIn(BlackjackState(12, True, 10, False), states)
        self.assertIn(BlackjackState(21, True, 10, True), states)
        self.assertEqual(BlackjackState(15, True, 10, False).legal_actions, (Action.HIT, Action.STAND))

    def test_context_rejects_out_of_grid_values(self) -> None:
        for total, upcard in ((2, 1), (22, 1), (10, 0), (10, 11)):
            with self.subTest(total=total, upcard=upcard), self.assertRaises(ValueError):
                BlackjackState(total, False, upcard, True)


class TestOneDecisionBlackjack(unittest.TestCase):
    def test_action_rewards_and_dealer_rules(self) -> None:
        cases = (
            ("stand win", [10, 10, 10, 9], Action.STAND, 1.0),
            ("stand loss", [10, 7, 10, 10], Action.STAND, -1.0),
            ("stand push", [10, 10, 10, 10], Action.STAND, 0.0),
            ("dealer bust", [10, 8, 10, 6, 10], Action.STAND, 1.0),
            ("hit once then stand on twelve", [5, 5, 10, 7, 2], Action.HIT, -1.0),
            ("hit win", [5, 6, 10, 7, 10], Action.HIT, 1.0),
            ("hit push", [5, 6, 10, 10, 9], Action.HIT, 0.0),
            ("double win", [5, 6, 10, 7, 10], Action.DOUBLE, 2.0),
            ("double loss", [5, 6, 10, 10, 2], Action.DOUBLE, -2.0),
            ("double push", [5, 6, 10, 10, 9], Action.DOUBLE, 0.0),
            ("hit bust before dealer", [10, 10, 10, 6, 10], Action.HIT, -1.0),
            ("double bust before dealer", [10, 10, 10, 6, 10], Action.DOUBLE, -2.0),
            ("dealer stands on soft seventeen", [10, 7, 1, 6], Action.STAND, 0.0),
            ("dealer demotes usable ace", [10, 7, 1, 5, 10, 2], Action.STAND, -1.0),
            ("standing natural", [1, 10, 10, 7], Action.STAND, 1.5),
            ("two naturals push", [1, 10, 10, 1], Action.STAND, 0.0),
            ("dealer natural beats twenty", [10, 10, 1, 10], Action.STAND, -1.0),
            ("hit forfeits natural bonus", [1, 10, 10, 10, 10], Action.HIT, 1.0),
            ("double forfeits natural bonus", [1, 10, 10, 10, 10], Action.DOUBLE, 2.0),
            ("hit twenty-one loses to dealer natural", [1, 10, 1, 10, 10], Action.HIT, -1.0),
            ("no peek: double loses full stake to natural", [5, 6, 1, 10, 10], Action.DOUBLE, -2.0),
        )
        for name, cards, action, expected in cases:
            with (
                self.subTest(name=name),
                patch("reinforcement_learning.blackjack.draw_card", side_effect=cards) as draw,
            ):
                environment = OneDecisionBlackjack()
                state = environment.reset(n_player_cards=2)
                self.assertTrue(state.can_double)
                self.assertEqual(state.dealer_upcard, cards[2])
                self.assertEqual(environment.step(action), expected)
                self.assertEqual(draw.call_count, len(cards))
                with self.assertRaises(RuntimeError):
                    environment.step(Action.STAND)

    def test_three_card_twenty_one_has_no_blackjack_bonus(self) -> None:
        for dealer, expected in (([10, 10], 1.0), ([1, 10], -1.0)):
            with (
                self.subTest(dealer=dealer),
                patch("reinforcement_learning.blackjack.draw_card", side_effect=[7, 7, 7, *dealer]),
            ):
                environment = OneDecisionBlackjack()
                state = environment.reset(n_player_cards=3)
                self.assertFalse(state.can_double)
                self.assertEqual(environment.step(Action.STAND), expected)

    def test_initial_blackjack_settles_without_dealer_draws(self) -> None:
        with patch("reinforcement_learning.blackjack.draw_card", side_effect=[1, 10, 10, 6]) as draw:
            environment = OneDecisionBlackjack()
            environment.reset(2)
            self.assertEqual(environment.step(Action.STAND), 1.5)
            self.assertEqual(draw.call_count, 4)

    def test_double_rejected_on_three_cards_without_finishing_round(self) -> None:
        with patch("reinforcement_learning.blackjack.draw_card", side_effect=[5, 5, 8, 10, 7]):
            environment = OneDecisionBlackjack()
            environment.reset(3)
            with self.assertRaises(ValueError):
                environment.step(Action.DOUBLE)
            self.assertEqual(environment.step(Action.STAND), 1.0)

    def test_reset_rejects_busted_exploring_start(self) -> None:
        with patch("reinforcement_learning.blackjack.draw_card", side_effect=[10, 10, 10, 2, 2, 2, 10, 7]):
            state = OneDecisionBlackjack().reset(3)
            self.assertEqual(state, BlackjackState(6, False, 10, False))

    def test_requires_reset_and_rejects_invalid_hand_length(self) -> None:
        environment = OneDecisionBlackjack()
        with self.assertRaises(RuntimeError):
            environment.step(Action.HIT)
        for length in (0, 1, 4):
            with self.subTest(length=length), self.assertRaises(ValueError):
                environment.reset(length)


if __name__ == "__main__":
    unittest.main()
