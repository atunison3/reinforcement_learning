"""Finite-shoe blackjack rules, observable counts, and whole-round credit."""

import csv
import unittest
from collections import Counter
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from reinforcement_learning.projects.project010 import (
    DECK,
    INVALID_REWARD,
    Action,
    BlackjackEnvironment,
    BlackjackState,
    CountBucket,
    Experience,
    HandCategory,
    MonteCarloAgent,
    categorize_hand,
    format_strategy_table,
    hi_lo,
    load_strategy,
    main,
    run_episode,
    save_strategy_tables,
    strategy_cell,
    table_hands,
)


def ordered_shoe(*prefix: int, decks: int = 6) -> tuple[int, ...]:
    """Build a physical shoe with a specified deal/draw order at its front."""
    remaining = Counter(DECK * decks)
    remaining.subtract(prefix)
    if any(count < 0 for count in remaining.values()):
        raise ValueError("Prefix uses more cards than are in the shoe")
    return (*prefix, *remaining.elements())


def require_state(state: BlackjackState | None) -> BlackjackState:
    """These test deals must reach a player decision, not a dealer blackjack."""
    if state is None:
        raise AssertionError("Expected a playable round")
    return state


class TestHandCategories(unittest.TestCase):
    def test_all_requested_two_card_categories_ignore_card_order(self) -> None:
        cases = (
            ((6, 2), HandCategory.SIX_TWO),
            ((5, 3), HandCategory.FIVE_THREE),
            ((4, 4), HandCategory.FOUR_FOUR),
            ((5, 4), HandCategory.NINE),
            ((3, 6), HandCategory.NINE),
            ((6, 4), HandCategory.TEN),
            ((6, 5), HandCategory.ELEVEN),
            ((10, 2), HandCategory.TWELVE),
            ((10, 3), HandCategory.THIRTEEN),
            ((10, 4), HandCategory.FOURTEEN),
            ((10, 5), HandCategory.FIFTEEN),
            ((10, 6), HandCategory.SIXTEEN),
            ((1, 1), HandCategory.ACE_ACE),
            ((2, 2), HandCategory.TWO_TWO),
            ((3, 3), HandCategory.THREE_THREE),
            ((5, 5), HandCategory.FIVE_FIVE),
            ((6, 6), HandCategory.SIX_SIX),
            ((7, 7), HandCategory.SEVEN_SEVEN),
            ((8, 8), HandCategory.EIGHT_EIGHT),
            ((9, 9), HandCategory.NINE_NINE),
            ((10, 10), HandCategory.TEN_TEN),
            ((1, 2), HandCategory.ACE_TWO),
            ((1, 3), HandCategory.ACE_THREE),
            ((1, 4), HandCategory.ACE_FOUR),
            ((1, 5), HandCategory.ACE_FIVE),
            ((1, 6), HandCategory.ACE_SIX),
            ((1, 7), HandCategory.ACE_SEVEN),
            ((1, 8), HandCategory.ACE_EIGHT),
            ((2, 3), HandCategory.BELOW_8),
            ((10, 7), HandCategory.ABOVE_16),
            ((1, 9), HandCategory.ABOVE_16),
            ((1, 10), HandCategory.BLACKJACK),
        )
        for cards, expected in cases:
            with self.subTest(cards=cards):
                self.assertIs(categorize_hand(cards), expected)
                self.assertIs(categorize_hand(cards[::-1]), expected)

    def test_multicard_and_split_hands_use_total_buckets_not_false_pairs_or_naturals(self) -> None:
        for cards, expected in (
            ((2, 2, 4), HandCategory.EIGHT),
            ((2, 3, 3), HandCategory.EIGHT),
            ((2, 3, 4), HandCategory.NINE),
            ((1, 1, 1), HandCategory.ACE_TWO),
            ((1, 2, 4), HandCategory.ACE_SIX),
            ((1, 6, 10), HandCategory.ABOVE_16),
            ((2, 9, 10), HandCategory.ABOVE_16),
        ):
            with self.subTest(cards=cards):
                self.assertIs(categorize_hand(cards), expected)
        self.assertIs(categorize_hand((1, 10), from_split=True), HandCategory.ABOVE_16)

    def test_rejects_empty_invalid_or_busted_decision_hands(self) -> None:
        for cards in ((), (4,), (0, 5), (11, 4), (10, 10, 2)):
            with self.subTest(cards=cards), self.assertRaises(ValueError):
                categorize_hand(cards)


class TestBlackjackRound(unittest.TestCase):
    def test_stand_and_double_rewards_and_dealer_rules(self) -> None:
        cases = (
            ("stand win", (10, 10, 10, 9), Action.STAND, 1),
            ("stand loss", (10, 7, 10, 10), Action.STAND, -1),
            ("stand push", (10, 10, 10, 10), Action.STAND, 0),
            ("dealer bust", (10, 8, 10, 6, 10), Action.STAND, 1),
            ("soft seventeen stands", (10, 7, 1, 6), Action.STAND, 0),
            ("dealer demotes ace", (10, 7, 1, 5, 10, 2), Action.STAND, -1),
            ("double win", (5, 6, 10, 7, 10), Action.DOUBLE, 2),
            ("double loss", (5, 6, 10, 10, 2), Action.DOUBLE, -2),
            ("double push", (5, 6, 10, 10, 9), Action.DOUBLE, 0),
            ("double bust", (10, 6, 10, 6, 10), Action.DOUBLE, -2),
            ("natural wins", (1, 10, 10, 7), Action.STAND, 1.5),
            ("natural needs no dealer draws", (1, 10, 10, 6), Action.STAND, 1.5),
        )
        for name, cards, action, reward in cases:
            with self.subTest(name=name):
                env = BlackjackEnvironment()
                env.reset(shoe=ordered_shoe(*cards))
                result = env.step(action)
                self.assertTrue(result.terminated)
                self.assertIsNone(result.state)
                self.assertIsNone(result.invalid_reason)
                self.assertEqual(result.reward, reward)
                self.assertEqual(result.hand_rewards, (reward,))
                self.assertEqual(env.cards_remaining, 312 - len(cards))

    def test_hits_continue_until_stand_and_twenty_one_is_not_natural(self) -> None:
        env = BlackjackEnvironment()
        state = require_state(env.reset(shoe=ordered_shoe(2, 3, 10, 7, 4, 5, 7)))
        self.assertEqual(state.hand, HandCategory.BELOW_8)
        for category in (HandCategory.NINE, HandCategory.FOURTEEN, HandCategory.ABOVE_16):
            result = env.step(Action.HIT)
            self.assertFalse(result.terminated)
            self.assertEqual(result.reward, 0)
            self.assertIsNotNone(result.state)
            if result.state is not None:
                self.assertEqual(result.state.hand, category)
                self.assertFalse(result.state.can_double)
                self.assertFalse(result.state.can_split)
        self.assertEqual(env.step(Action.STAND).reward, 1)

    def test_player_bust_loses_without_dealer_drawing(self) -> None:
        env = BlackjackEnvironment()
        env.reset(shoe=ordered_shoe(10, 6, 10, 6, 10))
        result = env.step(Action.HIT)
        self.assertEqual(result.hand_rewards, (-1,))
        self.assertTrue(result.terminated)
        self.assertEqual(env.cards_remaining, 307)

    def test_nonstanding_natural_actions_terminate_with_penalty(self) -> None:
        for action in (Action.HIT, Action.DOUBLE, Action.SPLIT):
            with self.subTest(action=action):
                env = BlackjackEnvironment()
                state = require_state(env.reset(shoe=ordered_shoe(1, 10, 10, 7)))
                self.assertIs(state.hand, HandCategory.BLACKJACK)
                self.assertFalse(state.can_double)
                self.assertFalse(state.can_split)
                self.assertEqual(env.actions, tuple(Action))
                result = env.step(action)
                self.assertEqual(result.reward, INVALID_REWARD)
                self.assertTrue(result.terminated)
                self.assertIsNotNone(result.invalid_reason)
                self.assertEqual(result.hand_rewards, ())
                self.assertEqual(env.cards_remaining, 308)

    def test_nonpair_split_and_multicard_double_or_split_are_penalized(self) -> None:
        for actions in ((Action.SPLIT,), (Action.HIT, Action.DOUBLE), (Action.HIT, Action.SPLIT)):
            with self.subTest(actions=actions):
                env = BlackjackEnvironment()
                env.reset(shoe=ordered_shoe(2, 3, 10, 7, 4))
                for action in actions[:-1]:
                    env.step(action)
                before = env.cards_remaining
                result = env.step(actions[-1])
                self.assertEqual(result.reward, -100)
                self.assertTrue(result.terminated)
                self.assertEqual(env.cards_remaining, before)

    def test_split_pays_two_wins_only_at_round_end(self) -> None:
        env = BlackjackEnvironment()
        initial = require_state(env.reset(shoe=ordered_shoe(8, 8, 10, 7, 10, 10)))
        self.assertTrue(initial.can_split)
        split = env.step(Action.SPLIT)
        first_stand = env.step(Action.STAND)
        for result in (split, first_stand):
            self.assertEqual(result.reward, 0)
            self.assertFalse(result.terminated)
            self.assertEqual(result.hand_rewards, ())
        end = env.step(Action.STAND)
        self.assertTrue(end.terminated)
        self.assertEqual(end.reward, 2)
        self.assertEqual(end.hand_rewards, (1, 1))
        self.assertEqual(env.cards_remaining, 306)  # One dealer for both hands.

    def test_bust_advances_to_other_split_hand_and_returns_net_result(self) -> None:
        env = BlackjackEnvironment()
        env.reset(shoe=ordered_shoe(8, 8, 10, 7, 5, 10, 10))
        env.step(Action.SPLIT)
        busted = env.step(Action.HIT)
        self.assertFalse(busted.terminated)
        self.assertEqual(busted.reward, 0)
        end = env.step(Action.STAND)
        self.assertEqual(end.hand_rewards, (-1, 1))
        self.assertEqual(end.reward, 0)

    def test_double_after_split_aggregates_both_two_unit_stakes(self) -> None:
        env = BlackjackEnvironment()
        env.reset(shoe=ordered_shoe(5, 5, 10, 7, 6, 6, 10, 10))
        split = env.step(Action.SPLIT)
        self.assertIsNotNone(split.state)
        if split.state is not None:
            self.assertTrue(split.state.can_double)
        first = env.step(Action.DOUBLE)
        self.assertFalse(first.terminated)
        self.assertEqual(first.reward, 0)
        end = env.step(Action.DOUBLE)
        self.assertEqual(end.hand_rewards, (2, 2))
        self.assertEqual(end.reward, 4)

    def test_split_ace_tens_are_not_naturals_and_push_against_dealer_drawn_twenty_one(self) -> None:
        for dealer, expected in (((10, 10, 7), 2), ((10, 6, 5), 0)):
            with self.subTest(dealer=dealer):
                env = BlackjackEnvironment()
                env.reset(shoe=ordered_shoe(1, 1, *dealer[:2], 10, 10, dealer[2]))
                split = env.step(Action.SPLIT)
                self.assertIsNotNone(split.state)
                if split.state is not None:
                    self.assertEqual(split.state.hand, HandCategory.ABOVE_16)
                    self.assertTrue(split.state.can_double)
                env.step(Action.STAND)
                self.assertEqual(env.step(Action.STAND).reward, expected)

    def test_split_aces_can_hit(self) -> None:
        env = BlackjackEnvironment()
        env.reset(shoe=ordered_shoe(1, 1, 10, 7, 2, 3, 5))
        env.step(Action.SPLIT)
        hit = env.step(Action.HIT)
        self.assertFalse(hit.terminated)
        self.assertIsNone(hit.invalid_reason)
        env.step(Action.STAND)
        self.assertEqual(env.step(Action.STAND).hand_rewards, (1, -1))

    def test_resplits_stop_at_configured_limit(self) -> None:
        for max_hands in (1, 2, 3, 4):
            with self.subTest(max_hands=max_hands):
                env = BlackjackEnvironment(max_hands=max_hands)
                state = require_state(env.reset(shoe=ordered_shoe(8, 8, 10, 7, 8, 8, 8, 8, 8, 8)))
                for _ in range(max_hands - 1):
                    self.assertTrue(state.can_split)
                    result = env.step(Action.SPLIT)
                    self.assertIsNotNone(result.state)
                    if result.state is not None:
                        state = result.state
                self.assertFalse(state.can_split)
                self.assertEqual(env.step(Action.SPLIT).reward, -100)

    def test_invalid_second_hand_discards_other_outcomes_for_whole_round_penalty(self) -> None:
        env = BlackjackEnvironment()
        env.reset(shoe=ordered_shoe(8, 8, 10, 7, 10, 10))
        env.step(Action.SPLIT)
        env.step(Action.STAND)
        end = env.step(Action.SPLIT)
        self.assertEqual(end.reward, -100)
        self.assertEqual(end.hand_rewards, ())

    def test_reset_and_step_lifecycle(self) -> None:
        env = BlackjackEnvironment()
        with self.assertRaises(RuntimeError):
            env.step(Action.STAND)
        env.reset(shoe=ordered_shoe(10, 10, 10, 7))
        with self.assertRaises(RuntimeError):
            env.reset()
        env.step(Action.STAND)
        count = env.running_count
        with self.assertRaises(RuntimeError):
            env.step(Action.STAND)
        self.assertEqual(env.running_count, count)
        env.reset()


class TestDealerPeek(unittest.TestCase):
    def test_dealer_natural_discards_round_before_any_player_action(self) -> None:
        for player in ((5, 6), (8, 8), (1, 10)):
            for dealer in ((10, 1), (1, 10)):
                with self.subTest(player=player, dealer=dealer):
                    env = BlackjackEnvironment()
                    cards = (*player, *dealer)
                    state = env.reset(shoe=ordered_shoe(*cards))
                    self.assertIsNone(state)
                    self.assertEqual(env.discard_reason, "dealer_blackjack")
                    self.assertEqual(env.cards_remaining, 308)
                    count = sum(map(hi_lo, cards))
                    self.assertEqual(env.running_count, count)  # Hole already revealed.
                    for action in Action:
                        with self.assertRaises(RuntimeError):
                            env.step(action)
                    self.assertEqual(env.cards_remaining, 308)
                    self.assertEqual(env.running_count, count)  # No second reveal.

    def test_negative_peek_leaves_hole_uncounted_and_allows_player_action(self) -> None:
        for upcard, hole in ((10, 2), (10, 10), (1, 2), (1, 9)):
            with self.subTest(upcard=upcard, hole=hole):
                env = BlackjackEnvironment()
                state = require_state(env.reset(shoe=ordered_shoe(5, 6, upcard, hole, 10, 10)))
                self.assertIsNone(env.discard_reason)
                self.assertTrue(state.can_double)
                self.assertEqual(env.running_count, hi_lo(5) + hi_lo(6) + hi_lo(upcard))
                result = env.step(Action.DOUBLE)
                self.assertTrue(result.terminated)
                self.assertIsNone(result.invalid_reason)
                self.assertEqual(result.reward, 2)

    def test_discarded_deal_skips_policy_and_learning_and_preserves_shoe_for_next_round(self) -> None:
        env = BlackjackEnvironment()
        agent = MonteCarloAgent(seed=7)
        previous_state = BlackjackState(HandCategory.NINE, True, False, 7, CountBucket.POSITIVE)
        agent.learn((Experience(previous_state, Action.STAND, 1),))
        old_q = {state: values.copy() for state, values in agent.q.items()}
        old_visits = {state: values.copy() for state, values in agent.visits.items()}
        reset = env.reset
        cards = (5, 6, 10, 1, 10, 10, 10, 7)
        with (
            patch.object(env, "reset", side_effect=lambda: reset(shoe=ordered_shoe(*cards))),
            patch.object(agent, "choose_action", wraps=agent.choose_action) as choose,
            patch.object(agent, "learn", wraps=agent.learn) as learn,
        ):
            self.assertEqual(run_episode(env, agent), ())
            choose.assert_not_called()
            learn.assert_not_called()
        self.assertEqual(agent.q, old_q)
        self.assertEqual(agent.visits, old_visits)
        self.assertEqual(env.running_count, 0)
        shuffles = env.shuffle_count
        with patch.object(agent, "choose_action", return_value=Action.STAND):
            episode = run_episode(env, agent)
        self.assertEqual(len(episode), 1)
        self.assertEqual(episode[0].reward, 1)
        self.assertIsNone(env.discard_reason)
        self.assertEqual(env.cards_remaining, 304)
        self.assertEqual(env.shuffle_count, shuffles)
        self.assertEqual(env.running_count, sum(map(hi_lo, cards)))

    def test_training_statistics_exclude_discarded_deals_and_handle_all_discarded(self) -> None:
        state = BlackjackState(HandCategory.NINE, True, False, 7, CountBucket.POSITIVE)
        valid_episode = (Experience(state, Action.DOUBLE, 2),)
        for episodes, expected_mean, discarded in (
            (((), valid_episode), "2.000", 1),
            (((), ()), "n/a (no training rounds)", 2),
        ):
            with self.subTest(discarded=discarded), TemporaryDirectory() as temp_dir:
                output = StringIO()
                with (
                    patch("sys.argv", ["project010", "--episodes", "2", "--output-dir", temp_dir]),
                    patch("sys.stdout", output),
                    patch("reinforcement_learning.projects.project010.run_episode", side_effect=episodes),
                ):
                    main()
                self.assertIn(f"Discarded dealer-blackjack rounds: {discarded}", output.getvalue())
                self.assertIn(f"excluding discarded rounds): {expected_mean}", output.getvalue())
                self.assertIn("Invalid-action rounds: 0", output.getvalue())


class TestShoeAndCount(unittest.TestCase):
    def test_hi_lo_weights_and_balanced_deck(self) -> None:
        self.assertEqual([hi_lo(card) for card in range(1, 11)], [-1, 1, 1, 1, 1, 1, 0, 0, 0, -1])
        self.assertEqual(sum(map(hi_lo, DECK)), 0)
        for card in (0, 11):
            with self.subTest(card=card), self.assertRaises(ValueError):
                hi_lo(card)

    def test_count_excludes_hole_until_reveal_and_counts_it_only_once(self) -> None:
        states = []
        for hole in (2, 10):
            env = BlackjackEnvironment()
            state = require_state(env.reset(shoe=ordered_shoe(1, 10, 10, hole)))
            states.append(state)
            self.assertEqual(env.running_count, -3)
            self.assertEqual(state.count, CountBucket.NON_POSITIVE)
            env.step(Action.STAND)
            self.assertEqual(env.running_count, -3 + hi_lo(hole))
        self.assertEqual(states[0], states[1])

    def test_visible_split_and_hit_cards_update_count_bucket(self) -> None:
        env = BlackjackEnvironment()
        initial = require_state(env.reset(shoe=ordered_shoe(8, 8, 10, 7, 2, 3, 10)))
        self.assertEqual(initial.count, CountBucket.NON_POSITIVE)
        split = env.step(Action.SPLIT)
        self.assertEqual(env.running_count, 1)
        self.assertIsNotNone(split.state)
        if split.state is not None:
            self.assertEqual(split.state.count, CountBucket.POSITIVE)
        hit = env.step(Action.HIT)
        self.assertEqual(env.running_count, 0)
        self.assertIsNotNone(hit.state)
        if hit.state is not None:
            self.assertEqual(hit.state.count, CountBucket.NON_POSITIVE)

    def test_shoe_and_running_count_persist_across_rounds(self) -> None:
        env = BlackjackEnvironment()
        env.reset(shoe=ordered_shoe(10, 10, 10, 7, 2, 3, 4, 10, 10))
        env.step(Action.STAND)
        self.assertEqual(env.running_count, -3)
        shuffles = env.shuffle_count
        state = require_state(env.reset())
        self.assertEqual(env.cards_remaining, 304)
        self.assertEqual(env.shuffle_count, shuffles)
        self.assertEqual(env.running_count, 0)  # Old -3 plus three exposed low cards.
        self.assertEqual(state.dealer_upcard, 4)
        self.assertEqual(state.count, CountBucket.NON_POSITIVE)
        env.step(Action.STAND)
        self.assertEqual(env.running_count, -2)  # Hole ten and dealer draw ten.

    def test_cut_card_reshuffles_only_between_rounds(self) -> None:
        env = BlackjackEnvironment(penetration=0.01, seed=7)
        env.reset(shoe=ordered_shoe(10, 10, 10, 7))
        shuffles = env.shuffle_count
        env.step(Action.STAND)
        self.assertEqual(env.shuffle_count, shuffles)
        env.reset()
        self.assertEqual(env.shuffle_count, shuffles + 1)
        self.assertEqual(env.cards_remaining, 308)
        self.assertTrue(-3 <= env.running_count <= 3)

    def test_injected_fresh_shoe_resets_count_and_validates_physical_cards(self) -> None:
        env = BlackjackEnvironment()
        env.reset(shoe=ordered_shoe(10, 10, 10, 7))
        env.step(Action.STAND)
        with self.assertRaises(ValueError):
            env.reset(shoe=(1, 10, 10, 7))
        state = require_state(env.reset(shoe=ordered_shoe(2, 3, 4, 10)))
        self.assertEqual(env.running_count, 3)
        self.assertEqual(state.count, CountBucket.POSITIVE)

    def test_invalid_configuration(self) -> None:
        for decks in (0, -1):
            with self.subTest(decks=decks), self.assertRaises(ValueError):
                BlackjackEnvironment(decks=decks)
        for limit in (0, 5):
            with self.subTest(max_hands=limit), self.assertRaises(ValueError):
                BlackjackEnvironment(max_hands=limit)
        for penetration in (0, 1, float("nan")):
            with self.subTest(penetration=penetration), self.assertRaises(ValueError):
                BlackjackEnvironment(penetration=penetration)


class TestMonteCarloCredit(unittest.TestCase):
    def test_optimistic_tens_drive_sampling_in_the_greedy_branch(self) -> None:
        state = BlackjackState(HandCategory.NINE, True, False, 6, CountBucket.POSITIVE)
        agent = MonteCarloAgent(seed=7, epsilon=0)
        tried: set[Action] = set()
        for _ in Action:
            action = agent.choose_action(state)
            self.assertNotIn(action, tried)
            agent.learn((Experience(state, action, 1),))
            tried.add(action)
            for candidate in Action:
                self.assertEqual(agent.q[state][candidate], 1 if candidate in tried else 10)
                self.assertEqual(agent.visits[state][candidate], int(candidate in tried))
        self.assertEqual(tried, set(Action))  # Invalid actions are still not masked.

    def test_epsilon_boundary_uses_greedy_branch_and_random_tie_breaking(self) -> None:
        state = BlackjackState(HandCategory.NINE, True, False, 6, CountBucket.POSITIVE)
        agent = MonteCarloAgent(seed=7)
        self.assertEqual(agent.epsilon, 0.1)
        for action, reward in zip(Action, (-2, -1, 2, -100), strict=True):
            agent.learn((Experience(state, action, reward),))
        with patch("reinforcement_learning.projects.project010.random.Random.random", return_value=0.1):
            self.assertEqual({agent.choose_action(state) for _ in range(100)}, {Action.DOUBLE})
            agent.q[state] = [2, -1, 2, -100]
            self.assertEqual({agent.choose_action(state) for _ in range(100)}, {Action.HIT, Action.DOUBLE})

    def test_exploration_includes_known_bad_actions_despite_state_flags(self) -> None:
        state = BlackjackState(HandCategory.BLACKJACK, False, False, 6, CountBucket.POSITIVE)
        for epsilon, draw in ((0.1, 0.099), (1.0, 0.999)):
            with self.subTest(epsilon=epsilon):
                agent = MonteCarloAgent(seed=7, epsilon=epsilon)
                agent.q[state] = [-100, 1.5, -100, -100]
                with (
                    patch("reinforcement_learning.projects.project010.random.Random.random", return_value=draw),
                    patch(
                        "reinforcement_learning.projects.project010.random.Random.choice", return_value=Action.SPLIT
                    ) as choose,
                ):
                    self.assertEqual(agent.choose_action(state), Action.SPLIT)
                    choose.assert_called_once_with(tuple(Action))

    def test_invalid_epsilon_is_rejected(self) -> None:
        for epsilon in (-0.1, 1.1, float("nan")):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                MonteCarloAgent(epsilon=epsilon)

    def test_split_receives_sum_of_both_winning_hands(self) -> None:
        env = BlackjackEnvironment()
        state = require_state(env.reset(shoe=ordered_shoe(8, 8, 10, 7, 10, 10)))
        split_state = state
        episode: list[Experience] = []
        for action in (Action.SPLIT, Action.STAND, Action.STAND):
            result = env.step(action)
            episode.append(Experience(state, action, result.reward))
            if result.state is not None:
                state = result.state
        self.assertEqual([item.reward for item in episode], [0, 0, 2])
        agent = MonteCarloAgent()
        agent.learn(episode)
        self.assertEqual(agent.q[split_state][Action.SPLIT], 2)
        self.assertEqual(agent.visits[split_state][Action.SPLIT], 1)

    def test_invalid_actions_are_not_masked_and_penalties_update_values(self) -> None:
        env = BlackjackEnvironment()
        state = require_state(env.reset(shoe=ordered_shoe(1, 10, 10, 7)))
        agent = MonteCarloAgent(seed=0, epsilon=0)
        agent.q[state] = [0, 0, 5, 0]
        self.assertEqual(agent.choose_action(state), Action.DOUBLE)
        result = env.step(Action.DOUBLE)
        agent.learn((Experience(state, Action.DOUBLE, result.reward),))
        self.assertEqual(agent.q[state][Action.DOUBLE], -100)
        self.assertNotEqual(agent.choose_action(state), Action.DOUBLE)

    def test_return_estimates_are_sample_averages(self) -> None:
        state = BlackjackState(HandCategory.NINE, True, False, 7, CountBucket.POSITIVE)
        agent = MonteCarloAgent()
        for reward in (2, -1, 2):
            agent.learn((Experience(state, Action.DOUBLE, reward),))
        self.assertEqual(agent.visits[state][Action.DOUBLE], 3)
        self.assertAlmostEqual(agent.q[state][Action.DOUBLE], 1)

    def test_seeded_training_is_reproducible_and_completes_rounds_without_midround_shuffles(self) -> None:
        for decks in (1, 6):
            with self.subTest(decks=decks):
                left_env = BlackjackEnvironment(decks=decks, penetration=0.99, seed=7)
                right_env = BlackjackEnvironment(decks=decks, penetration=0.99, seed=7)
                left_agent = MonteCarloAgent(seed=8)
                right_agent = MonteCarloAgent(seed=8)
                for _ in range(200):
                    left = run_episode(left_env, left_agent)
                    right = run_episode(right_env, right_agent)
                    self.assertEqual(left, right)
                    if not left:
                        self.assertEqual(left_env.discard_reason, "dealer_blackjack")
                        self.assertEqual(right_env.discard_reason, "dealer_blackjack")
                    self.assertEqual(left_env.running_count, right_env.running_count)
                self.assertEqual(left_agent.q, right_agent.q)
                self.assertGreater(left_env.shuffle_count, 1)
                self.assertGreater(len(left_agent.q), 0)


class TestStrategyLoading(unittest.TestCase):
    def test_round_trip_preserves_means_counts_and_continues_weighted_learning(self) -> None:
        state = BlackjackState(HandCategory.NINE, True, False, 6, CountBucket.POSITIVE)
        other = BlackjackState(HandCategory.BLACKJACK, False, False, 10, CountBucket.NON_POSITIVE)
        original = MonteCarloAgent()
        for reward in (2, -1):
            original.learn((Experience(state, Action.DOUBLE, reward),))
        original.learn((Experience(state, Action.STAND, 0),))  # Sampled zero is not untried.
        original.learn((Experience(other, Action.STAND, 1.5),))
        with TemporaryDirectory() as temp_dir:
            _, path = save_strategy_tables(original, Path(temp_dir))
            restored = MonteCarloAgent(seed=7)
            restored.learn((Experience(other, Action.HIT, -100),))
            load_strategy(restored, path)
        self.assertEqual(restored.q, original.q)
        self.assertEqual(restored.visits, original.visits)
        self.assertEqual(restored.epsilon, 0.1)
        self.assertEqual(restored.q[state][Action.HIT], 10)  # Untried CSV cells regain optimism.
        self.assertEqual(restored.visits[state][Action.HIT], 0)
        self.assertEqual(restored.q[state][Action.STAND], 0)  # Keep genuinely sampled zeros.
        self.assertEqual(restored.visits[state][Action.STAND], 1)
        self.assertIn(restored.choose_action(state), (Action.HIT, Action.SPLIT))
        restored.learn((Experience(state, Action.DOUBLE, 2),))
        self.assertEqual(restored.visits[state][Action.DOUBLE], 3)
        self.assertAlmostEqual(restored.q[state][Action.DOUBLE], 1)

    def test_all_unseen_rows_restore_an_empty_learner(self) -> None:
        with TemporaryDirectory() as temp_dir:
            _, path = save_strategy_tables(MonteCarloAgent(), Path(temp_dir))
            agent = MonteCarloAgent()
            load_strategy(agent, path)
            self.assertEqual(agent.q, {})
            self.assertEqual(agent.visits, {})

    def test_malformed_rows_and_duplicates_are_rejected_without_partial_updates(self) -> None:
        state = BlackjackState(HandCategory.NINE, True, False, 6, CountBucket.POSITIVE)
        other = BlackjackState(HandCategory.BLACKJACK, False, False, 10, CountBucket.NON_POSITIVE)
        original = MonteCarloAgent()
        original.learn((Experience(state, Action.HIT, 1),))
        original.learn((Experience(other, Action.STAND, 1.5),))
        with TemporaryDirectory() as temp_dir:
            _, path = save_strategy_tables(original, Path(temp_dir))
            with path.open(newline="", encoding="utf-8") as source:
                rows = [row for row in csv.DictReader(source) if int(row["state_visits"]) > 0]
            first = next(row for row in rows if row["hand"] == "9")
            second = next(row for row in rows if row["hand"] == "blackjack")
            malformed = (
                {"count": "unknown"},
                {"hand": "unknown"},
                {"dealer_upcard": "11"},
                {"can_double": "yes"},
                {"can_double": "True"},  # A natural cannot double.
                {"can_split": "True"},
                {"visits_stand": "-1"},
                {"visits_stand": "1.5"},
                {"q_stand": ""},
                {"q_stand": "nan"},
                {"q_stand": "inf"},
                {"q_hit": "0"},  # Untried cells must be blank.
                {"state_visits": "2"},
            )
            for update in malformed:
                with self.subTest(update=update):
                    with path.open("w", newline="", encoding="utf-8") as output:
                        writer = csv.DictWriter(output, fieldnames=list(first))
                        writer.writeheader()
                        writer.writerows((first, second | update))
                    agent = MonteCarloAgent()
                    agent.learn((Experience(other, Action.STAND, 3),))
                    before_q = {key: values.copy() for key, values in agent.q.items()}
                    before_visits = {key: values.copy() for key, values in agent.visits.items()}
                    with self.assertRaisesRegex(ValueError, "row 3"):
                        load_strategy(agent, path)
                    self.assertEqual(agent.q, before_q)
                    self.assertEqual(agent.visits, before_visits)
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=list(first))
                writer.writeheader()
                writer.writerows((first, first))
            with self.assertRaisesRegex(ValueError, "duplicate state"):
                load_strategy(MonteCarloAgent(), path)

    def test_rejects_missing_file_headers_empty_data_and_wrong_row_width(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.csv"
            with self.assertRaises(FileNotFoundError):
                load_strategy(MonteCarloAgent(), path)
            for text in ("", "hand,q_hit\n9,1\n"):
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "expected strategy CSV columns"):
                    load_strategy(MonteCarloAgent(), path)
            _, path = save_strategy_tables(MonteCarloAgent(), Path(temp_dir))
            header = path.read_text(encoding="utf-8").splitlines()[0] + "\n"
            path.write_text(header, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no state rows"):
                load_strategy(MonteCarloAgent(), path)
            path.write_text(header + "missing,columns\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "wrong number of columns"):
                load_strategy(MonteCarloAgent(), path)

    def test_cli_resumes_and_can_safely_replace_the_loaded_csv(self) -> None:
        state = BlackjackState(HandCategory.NINE, True, False, 6, CountBucket.POSITIVE)
        original = MonteCarloAgent()
        for reward in (2, -1):
            original.learn((Experience(state, Action.DOUBLE, reward),))

        def train_one(environment: BlackjackEnvironment, agent: MonteCarloAgent) -> tuple[Experience, ...]:
            self.assertEqual(agent.visits[state][Action.DOUBLE], 2)
            self.assertAlmostEqual(agent.q[state][Action.DOUBLE], 0.5)
            episode = (Experience(state, Action.DOUBLE, 2),)
            agent.learn(episode)
            return episode

        with TemporaryDirectory() as temp_dir:
            _, path = save_strategy_tables(original, Path(temp_dir))
            output = StringIO()
            with (
                patch(
                    "sys.argv",
                    ["project010", "--episodes", "1", "--load-strategy", str(path), "--output-dir", temp_dir],
                ),
                patch("sys.stdout", output),
                patch("reinforcement_learning.projects.project010.run_episode", side_effect=train_one),
            ):
                main()
            restored = MonteCarloAgent()
            load_strategy(restored, path)
            self.assertEqual(restored.visits[state][Action.DOUBLE], 3)
            self.assertAlmostEqual(restored.q[state][Action.DOUBLE], 1)
            self.assertIn("states: 1; action visits: 2", output.getvalue())
            self.assertIn(
                "Mean training return (including penalties, excluding discarded rounds): 2.000", output.getvalue()
            )

    def test_cli_load_error_does_not_train_or_create_reports(self) -> None:
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "output"
            with (
                patch(
                    "sys.argv",
                    [
                        "project010",
                        "--load-strategy",
                        str(Path(temp_dir) / "missing.csv"),
                        "--output-dir",
                        str(destination),
                    ],
                ),
                patch("sys.stderr", new_callable=StringIO) as errors,
                patch("reinforcement_learning.projects.project010.run_episode") as train,
                self.assertRaises(SystemExit) as exc,
            ):
                main()
            self.assertEqual(exc.exception.code, 2)
            self.assertIn("Cannot load strategy", errors.getvalue())
            train.assert_not_called()
            self.assertFalse(destination.exists())


class TestTrainingTiming(unittest.TestCase):
    def test_each_ten_thousand_and_partial_block_include_discarded_deals(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output = StringIO()
            with (
                patch("sys.argv", ["project010", "--episodes", "25000", "--output-dir", temp_dir]),
                patch("sys.stdout", output),
                patch("reinforcement_learning.projects.project010.run_episode", return_value=()),
                patch(
                    "reinforcement_learning.projects.project010.perf_counter",
                    side_effect=[10, 12, 14, 14.5, 17, 17.5, 18, 22],
                ),
            ):
                main()
            text = output.getvalue()
            self.assertIn("Episodes 1-10,000 (10,000 dealt): 2.000 s", text)
            self.assertIn("Episodes 10,001-20,000 (10,000 dealt): 2.500 s", text)
            self.assertIn("Episodes 20,001-25,000 (5,000 dealt, partial block): 0.500 s", text)
            self.assertEqual(text.count("(10,000 dealt):"), 2)
            self.assertNotIn("(1,000 dealt):", text)
            self.assertIn("Training time this run: 6.000 s", text)
            self.assertIn("Average time per 1,000 dealt episodes (normalized): 0.240 s", text)
            self.assertIn("Total program time (including load, training, and reports): 12.000 s", text)
            self.assertIn("Discarded dealer-blackjack rounds: 25000", text)
            self.assertIn("Training time this run: 6.000 s", (Path(temp_dir) / "project_10_strategy.md").read_text())

    def test_exact_ten_thousand_has_no_partial_block(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output = StringIO()
            with (
                patch("sys.argv", ["project010", "--episodes", "10000", "--output-dir", temp_dir]),
                patch("sys.stdout", output),
                patch("reinforcement_learning.projects.project010.run_episode", return_value=()),
                patch("reinforcement_learning.projects.project010.perf_counter", side_effect=[0, 1, 3, 3, 3, 4]),
            ):
                main()
            text = output.getvalue()
            self.assertEqual(text.count("(10,000 dealt):"), 1)
            self.assertNotIn("(1,000 dealt):", text)
            self.assertNotIn("partial block", text)
            self.assertIn("Average time per 1,000 dealt episodes (normalized): 0.200 s", text)

    def test_runs_below_ten_thousand_report_only_a_partial_block(self) -> None:
        for episodes, normalized in ((250, "3.000"), (2500, "0.300"), (9999, "0.075")):
            with self.subTest(episodes=episodes), TemporaryDirectory() as temp_dir:
                output = StringIO()
                with (
                    patch("sys.argv", ["project010", "--episodes", str(episodes), "--output-dir", temp_dir]),
                    patch("sys.stdout", output),
                    patch("reinforcement_learning.projects.project010.run_episode", return_value=()),
                    patch("reinforcement_learning.projects.project010.perf_counter", side_effect=[0, 1, 1.75, 2]),
                ):
                    main()
                text = output.getvalue()
                self.assertNotIn("(10,000 dealt):", text)
                self.assertNotIn("(1,000 dealt):", text)
                self.assertIn(f"Episodes 1-{episodes:,} ({episodes:,} dealt, partial block): 0.750 s", text)
                self.assertIn(f"Average time per 1,000 dealt episodes (normalized): {normalized} s", text)
                self.assertIn("Total program time (including load, training, and reports): 2.000 s", text)


class TestStrategyTables(unittest.TestCase):
    def test_unseen_and_untried_greedy_actions_are_not_arbitrary_recommendations(self) -> None:
        state = BlackjackState(HandCategory.NINE, True, False, 6, CountBucket.POSITIVE)
        agent = MonteCarloAgent()
        self.assertEqual(strategy_cell(agent, state), "?")
        agent.learn((Experience(state, Action.HIT, -1),))
        self.assertEqual(strategy_cell(agent, state), "?")  # Untried optimistic tens win.
        agent.learn((Experience(state, Action.STAND, 0),))
        self.assertEqual(strategy_cell(agent, state), "?")  # Untried actions still tie for best.

    def test_sampled_greedy_action_ties_and_incomplete_coverage(self) -> None:
        state = BlackjackState(HandCategory.NINE, True, False, 6, CountBucket.POSITIVE)
        agent = MonteCarloAgent(seed=0)
        agent.learn((Experience(state, Action.DOUBLE, 2),))
        self.assertEqual(strategy_cell(agent, state), "?")
        agent.learn((Experience(state, Action.HIT, 2),))
        self.assertEqual(strategy_cell(agent, state), "?")
        agent.learn((Experience(state, Action.STAND, 1),))
        agent.learn((Experience(state, Action.SPLIT, -100),))
        self.assertEqual(strategy_cell(agent, state), "H/D")

    def test_invalid_learned_choices_are_flagged_not_silently_masked(self) -> None:
        state = BlackjackState(HandCategory.BLACKJACK, False, False, 6, CountBucket.POSITIVE)
        agent = MonteCarloAgent()
        # Synthetic diagnostic values: reporting should not replace the learned policy.
        agent.q[state] = [2, 1, 2, 2]
        agent.visits[state] = [5, 5, 5, 5]
        self.assertEqual(strategy_cell(agent, state), "H!/D!/P!")
        pair = BlackjackState(HandCategory.EIGHT_EIGHT, True, True, 6, CountBucket.POSITIVE)
        agent.q[pair] = [-1, -1, -1, 2]
        agent.visits[pair] = [5, 5, 5, 5]
        self.assertEqual(strategy_cell(agent, pair), "P")

    def test_table_groups_preserve_count_flags_and_dealer_columns(self) -> None:
        agent = MonteCarloAgent()
        for state, action in (
            (BlackjackState(HandCategory.NINE, True, False, 6, CountBucket.POSITIVE), Action.DOUBLE),
            (BlackjackState(HandCategory.NINE, False, False, 6, CountBucket.POSITIVE), Action.HIT),
            (BlackjackState(HandCategory.NINE, True, False, 6, CountBucket.NON_POSITIVE), Action.STAND),
        ):
            for candidate in Action:
                agent.learn((Experience(state, candidate, 1 if candidate is action else -100),))
        text = format_strategy_table(agent)
        self.assertEqual(text.count("## Count"), 6)
        self.assertIn("| Hand / dealer upcard | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |", text)
        for heading, expected in (
            ("Count >0; double yes; split no", "D"),
            ("Count >0; double no; split no", "H"),
            ("Count <1; double yes; split no", "S"),
        ):
            with self.subTest(heading=heading):
                section = text.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
                row = next(line for line in section.splitlines() if line.startswith("| 9 |"))
                cells = [cell.strip() for cell in row.strip("|").split("|")]
                self.assertEqual(cells[5], expected)  # Dealer six, not another upcard.
                self.assertEqual(cells[10], "?")  # Unvisited dealer ace.

    def test_table_rows_omit_impossible_combinations_but_include_unseen_categories(self) -> None:
        self.assertEqual(table_hands(False, True), ())
        self.assertIn(HandCategory.BLACKJACK, table_hands(False, False))
        self.assertNotIn(HandCategory.BLACKJACK, table_hands(True, False))
        self.assertIn(HandCategory.EIGHT, table_hands(False, False))
        self.assertNotIn(HandCategory.EIGHT, table_hands(True, False))
        self.assertIn(HandCategory.FOUR_FOUR, table_hands(True, True))
        self.assertIn(HandCategory.FOUR_FOUR, table_hands(True, False))
        self.assertNotIn(HandCategory.FOUR_FOUR, table_hands(False, False))
        self.assertEqual(len(table_hands(True, True)), 10)
        text = format_strategy_table(MonteCarloAgent())
        self.assertIn("| blackjack | " + " | ".join(["?"] * 10) + " |", text)

    def test_export_contains_visit_counts_q_estimates_and_empty_untried_values(self) -> None:
        agent = MonteCarloAgent()
        state = BlackjackState(HandCategory.NINE, True, False, 6, CountBucket.POSITIVE)
        agent.learn((Experience(state, Action.DOUBLE, 2),))
        with TemporaryDirectory() as temp_dir:
            md_path, csv_path = save_strategy_tables(agent, Path(temp_dir) / "nested", summary=("Seed: 7",))
            self.assertIn("- Seed: 7", md_path.read_text(encoding="utf-8"))
            self.assertNotIn(b"\r", csv_path.read_bytes())
            with csv_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 1160)
            keys = {
                (row["count"], row["can_double"], row["can_split"], row["hand"], row["dealer_upcard"]) for row in rows
            }
            self.assertEqual(len(keys), len(rows))
            learned = next(row for row in rows if int(row["state_visits"]) > 0)
            self.assertEqual(learned["recommendation"], "?")  # Untried tens exceed the learned two.
            self.assertEqual(learned["count"], ">0")
            self.assertEqual(learned["dealer_upcard"], "6")
            self.assertEqual(learned["state_visits"], "1")
            self.assertEqual(learned["visits_double"], "1")
            self.assertEqual(float(learned["q_double"]), 2)
            self.assertEqual(learned["q_hit"], "")
            self.assertEqual(learned["visits_hit"], "0")
            # A new run overwrites the previous report without stale recommendations.
            save_strategy_tables(MonteCarloAgent(), md_path.parent)
            with csv_path.open(newline="", encoding="utf-8") as source:
                self.assertTrue(all(row["recommendation"] == "?" for row in csv.DictReader(source)))

    def test_reporting_is_deterministic_and_does_not_sample_or_modify_the_agent(self) -> None:
        state = BlackjackState(HandCategory.NINE, True, False, 6, CountBucket.POSITIVE)
        agent = MonteCarloAgent(seed=0)
        agent.learn((Experience(state, Action.HIT, 1),))
        before_q = {key: values.copy() for key, values in agent.q.items()}
        before_visits = {key: values.copy() for key, values in agent.visits.items()}
        with patch.object(agent, "choose_action", side_effect=AssertionError("Do not sample the policy")):
            first = format_strategy_table(agent)
            self.assertEqual(first, format_strategy_table(agent))
        self.assertEqual(agent.q, before_q)
        self.assertEqual(agent.visits, before_visits)

    def test_cli_prints_tables_and_saves_to_selected_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output = StringIO()
            with (
                patch("sys.argv", ["project010", "--episodes", "20", "--seed", "7", "--output-dir", temp_dir]),
                patch("sys.stdout", output),
            ):
                main()
            self.assertIn("# Project 10 — Learned Action Tables", output.getvalue())
            self.assertIn("Policy: epsilon-greedy; epsilon: 0.1; initial action value: 10", output.getvalue())
            self.assertIn("## Count >0; double yes; split yes", output.getvalue())
            report = Path(temp_dir) / "project_10_strategy.md"
            self.assertIn(report.read_text(encoding="utf-8"), output.getvalue())
            self.assertTrue((Path(temp_dir) / "project_10_strategy.csv").is_file())


if __name__ == "__main__":
    unittest.main()
