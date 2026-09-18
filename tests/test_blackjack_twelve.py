"""Finite-shoe and one-decision rules for the single-state blackjack bandit."""

import unittest

import numpy as np

from reinforcement_learning.blackjack import Action, hand_value
from reinforcement_learning.blackjack_twelve import (
    CARD_COUNTS,
    HAND_WEIGHTS,
    PLAYER_HANDS,
    TwelveRound,
    TwoDeckTwelveBandit,
    remaining_shoe,
)


class TestTwoDeckTwelveBandit(unittest.TestCase):
    def test_conditional_pair_weights_match_physical_card_multiplicities(self) -> None:
        for (first, second), weight in zip(PLAYER_HANDS, HAND_WEIGHTS):
            with self.subTest(hand=(first, second)):
                a, b = int(CARD_COUNTS[first - 1]), int(CARD_COUNTS[second - 1])
                expected = a * (a - 1) / 2 if first == second else a * b
                self.assertEqual(weight, expected)
                self.assertEqual(hand_value((first, second))[0], 12)
        self.assertEqual(float(np.sum(HAND_WEIGHTS)), 504.0)

    def test_every_deal_preserves_exactly_two_decks(self) -> None:
        environment = TwoDeckTwelveBandit(np.random.default_rng(42))
        for _ in range(100):
            deal = environment.deal()
            self.assertIn(deal.player_hand, PLAYER_HANDS)
            self.assertEqual(deal.remaining_cards.shape, (100,))
            all_cards = np.concatenate((deal.player_hand, deal.dealer_hand, deal.remaining_cards))
            np.testing.assert_array_equal(np.bincount(all_cards, minlength=11)[1:], CARD_COUNTS)

    def test_remaining_shoe_removes_both_player_cards(self) -> None:
        shoe = remaining_shoe((1, 1))
        self.assertEqual(len(shoe), 102)
        self.assertEqual(int(np.sum(shoe == 1)), 6)
        self.assertEqual(int(np.sum(shoe == 10)), 32)
        with self.assertRaises(ValueError):
            remaining_shoe((10, 10))

    def test_deals_are_reproducible(self) -> None:
        first = TwoDeckTwelveBandit(np.random.default_rng(7)).deal()
        second = TwoDeckTwelveBandit(np.random.default_rng(7)).deal()
        self.assertEqual(first.player_hand, second.player_hand)
        self.assertEqual(first.dealer_hand, second.dealer_hand)
        np.testing.assert_array_equal(first.remaining_cards, second.remaining_cards)

    def test_rewards_use_exactly_one_player_action_and_finite_shoe_order(self) -> None:
        cases = (
            ((10, 2), (10, 7), [9], -1.0, 1.0),
            ((10, 2), (10, 7), [5], -1.0, 0.0),
            ((10, 2), (10, 6), [10], 1.0, -1.0),
            # Hit consumes the first card: dealer now draws 10, not 5.
            ((10, 2), (10, 6), [5, 10], -1.0, 1.0),
            ((1, 1), (10, 6), [10, 10], 1.0, 1.0),
            ((1, 1), (1, 10), [9], -1.0, -1.0),
        )
        for player, dealer, cards, stand_reward, hit_reward in cases:
            with self.subTest(player=player, dealer=dealer, cards=cards):
                deal = TwelveRound(player, dealer, np.array(cards, dtype=np.int64))
                self.assertEqual(deal.reward(Action.STAND), stand_reward)
                self.assertEqual(deal.reward(Action.HIT), hit_reward)
                self.assertEqual(deal.reward(Action.STAND), stand_reward)
                np.testing.assert_array_equal(deal.remaining_cards, cards)

    def test_dealer_stands_on_soft_seventeen_and_double_is_rejected(self) -> None:
        deal = TwelveRound((6, 6), (1, 6), np.array([], dtype=np.int64))
        self.assertEqual(deal.reward(Action.STAND), -1.0)
        with self.assertRaises(ValueError):
            deal.reward(Action.DOUBLE)


if __name__ == "__main__":
    unittest.main()
