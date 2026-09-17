"""Infinite-deck blackjack with a single player decision per sampled hand."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from itertools import product

import numpy as np

MIN_HAND_VALUE = 3
MAX_HAND_VALUE = 21
N_CONTEXTS = (MAX_HAND_VALUE - MIN_HAND_VALUE + 1) * 2 * 10 * 2


class Action(IntEnum):
    """Available player decisions; values index the action-value table."""

    HIT = 0
    STAND = 1
    DOUBLE = 2


@dataclass(frozen=True, order=True)
class BlackjackState:
    """Observable context; neither the dealer hole card nor future draws leak in."""

    hand_value: int
    is_soft: bool
    dealer_upcard: int
    can_double: bool

    def __post_init__(self) -> None:
        if not MIN_HAND_VALUE <= self.hand_value <= MAX_HAND_VALUE:
            raise ValueError("`hand_value` must be between 3 and 21")
        if not 1 <= self.dealer_upcard <= 10:
            raise ValueError("`dealer_upcard` must be between 1 (ace) and 10")

    @property
    def index(self) -> int:
        """Flatten (total, soft, upcard, can_double), including unused grid cells."""

        return (((self.hand_value - MIN_HAND_VALUE) * 2 + int(self.is_soft)) * 10 + self.dealer_upcard - 1) * 2 + int(
            self.can_double
        )

    @property
    def legal_actions(self) -> tuple[Action, ...]:
        """Doubling is available only on an initial two-card hand."""

        return tuple(Action) if self.can_double else (Action.HIT, Action.STAND)


def hand_value(cards: Sequence[int]) -> tuple[int, bool]:
    """Return the best blackjack total and whether an ace counts as eleven.

    Cards are encoded as 1 (ace) through 10 (including all face cards). At
    most one ace can count as eleven; all others count as one. A bust returns
    its total above 21 and ``False`` rather than raising an exception.
    """

    if not cards or any(card < 1 or card > 10 for card in cards):
        raise ValueError("A hand must contain cards with values from 1 through 10")
    total = sum(cards)
    is_soft = 1 in cards and total + 10 <= MAX_HAND_VALUE
    return total + 10 if is_soft else total, is_soft


def draw_card(rng: np.random.Generator) -> int:
    """Draw with replacement: ace through nine each 1/13, ten-valued cards 4/13."""

    return min(int(rng.integers(1, 14)), 10)


def reachable_states() -> tuple[BlackjackState, ...]:
    """Enumerate non-busted two- and three-card starting contexts.

    The full 3..21 grid contains impossible cells. Three-card starts cover
    every (total, soft) combination reachable with more than two cards.
    """

    states: set[BlackjackState] = set()
    for n_cards in (2, 3):
        for cards in product(range(1, 11), repeat=n_cards):
            total, soft = hand_value(cards)
            if total <= MAX_HAND_VALUE:
                for upcard in range(1, 11):
                    states.add(BlackjackState(total, soft, upcard, n_cards == 2))
    return tuple(sorted(states))


class OneDecisionBlackjack:
    """Deal a context, take exactly one player action, then settle the hand.

    The dealer stands on all 17s, including soft 17. A hit draws one card and
    then forces a stand. A double does the same with a stake of two. There is
    no splitting, surrender, insurance, or dealer peek before the decision.

    An initial two-card 21 pays 3:2 only if the player stands and the dealer
    does not also have blackjack. Dealer blackjack beats other hands, even
    multi-card 21; two standing naturals push. Other wins/losses pay +/- the
    stake. After resolution, another action requires a fresh reset.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = np.random.default_rng() if rng is None else rng
        self.player_hand: list[int] = []
        self.dealer_hand: list[int] = []
        self._active = False

    def reset(self, n_player_cards: int | None = None) -> BlackjackState:
        """Sample a non-busted hand and return only the observable context.

        By default choose two or three starting cards with equal probability.
        Three cards are exploring starts, not another decision in this round.
        Rejection sampling removes busted starts without changing the chosen
        hand length. Pass 2 to deal only ordinary initial blackjack hands.
        """

        if n_player_cards is None:
            n_player_cards = int(self.rng.integers(2, 4))
        if n_player_cards not in (2, 3):
            raise ValueError("`n_player_cards` must be 2 or 3")
        while True:
            self.player_hand = [draw_card(self.rng) for _ in range(n_player_cards)]
            total, soft = hand_value(self.player_hand)
            if total <= MAX_HAND_VALUE:
                break
        self.dealer_hand = [draw_card(self.rng), draw_card(self.rng)]
        self._active = True
        return BlackjackState(total, soft, self.dealer_hand[0], n_player_cards == 2)

    def step(self, action: Action) -> float:
        """Apply one legal decision and return its complete, immediate reward."""

        if not self._active:
            raise RuntimeError("Call reset() before taking an action")
        action = Action(action)
        if action == Action.DOUBLE and len(self.player_hand) != 2:
            raise ValueError("Doubling is allowed only on an initial two-card hand")

        player_total, _ = hand_value(self.player_hand)
        player_natural = len(self.player_hand) == 2 and player_total == 21 and action == Action.STAND
        stake = 2.0 if action == Action.DOUBLE else 1.0
        if action in (Action.HIT, Action.DOUBLE):
            self.player_hand.append(draw_card(self.rng))
            player_total, _ = hand_value(self.player_hand)
        self._active = False
        if player_total > MAX_HAND_VALUE:
            return -stake  # Player bust loses immediately, irrespective of the dealer.

        dealer_total, _ = hand_value(self.dealer_hand)
        dealer_natural = len(self.dealer_hand) == 2 and dealer_total == 21
        if dealer_natural:
            return 0.0 if player_natural else -stake
        if player_natural:
            return 1.5

        while dealer_total < 17:
            self.dealer_hand.append(draw_card(self.rng))
            dealer_total, _ = hand_value(self.dealer_hand)
        if dealer_total > MAX_HAND_VALUE or player_total > dealer_total:
            return stake
        if player_total < dealer_total:
            return -stake
        return 0.0
