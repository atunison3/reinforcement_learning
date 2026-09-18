"""A single-state, two-action blackjack bandit with a fresh two-deck shoe."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reinforcement_learning.blackjack import Action, hand_value

# Each value 1..9 occurs eight times; four ten-valued ranks occur 32 times.
CARD_COUNTS: NDArray[np.int64] = np.array([8] * 9 + [32], dtype=np.int64)
PLAYER_HANDS = ((1, 1), (10, 2), (9, 3), (8, 4), (7, 5), (6, 6))
# Counts of unordered physical-card pairs in a 104-card shoe, conditional on 12.
HAND_WEIGHTS: NDArray[np.float64] = np.array([28, 256, 64, 64, 64, 28], dtype=np.float64)


def remaining_shoe(player_hand: tuple[int, int]) -> NDArray[np.int64]:
    """Remove the player's two cards from two standard decks (no jokers)."""

    if hand_value(player_hand)[0] != 12:
        raise ValueError("The player's two-card hand must total 12")
    counts = CARD_COUNTS.copy()
    for card in player_hand:
        counts[card - 1] -= 1
    return np.repeat(np.arange(1, 11, dtype=np.int64), counts)


@dataclass(frozen=True)
class TwelveRound:
    """One physical deal, allowing counterfactual evaluation of either arm.

    The agent is not given this deal. Its only observation is the reward of
    the selected action. Both counterfactuals start from the same remaining
    shoe: hit consumes one card before the dealer, whereas stand does not.
    """

    player_hand: tuple[int, int]
    dealer_hand: tuple[int, int]
    remaining_cards: NDArray[np.int64]

    def reward(self, action: Action) -> float:
        """Resolve hit-once-then-stand or stand; dealer stands on soft 17."""

        if action not in (Action.HIT, Action.STAND):
            raise ValueError("Only HIT and STAND are available")
        player = list(self.player_hand)
        dealer = list(self.dealer_hand)
        cards = iter(self.remaining_cards)
        if action == Action.HIT:
            player.append(int(next(cards)))
        player_total, _ = hand_value(player)
        if player_total > 21:
            return -1.0
        dealer_total, _ = hand_value(dealer)
        if dealer_total == 21:  # Initial dealer blackjack beats a three-card 21.
            return -1.0
        while dealer_total < 17:
            dealer.append(int(next(cards)))
            dealer_total, _ = hand_value(dealer)
        if dealer_total > 21 or player_total > dealer_total:
            return 1.0
        if player_total < dealer_total:
            return -1.0
        return 0.0


class TwoDeckTwelveBandit:
    """Sample independent games conditioned on an initial player total of 12.

    Hand types are weighted by physical-card multiplicity, not uniformly by
    their six names. Sampling a weighted pair and shuffling the other 102
    cards is equivalent to shuffling all 104 and rejecting non-12 starts.
    Dealer upcard/hole card are dealt next; neither is a learning context.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._shoes = tuple(remaining_shoe(hand) for hand in PLAYER_HANDS)

    def deal(self) -> TwelveRound:
        """Restore both decks, choose a 12, shuffle, and deal two dealer cards."""

        index = int(self.rng.choice(len(PLAYER_HANDS), p=HAND_WEIGHTS / np.sum(HAND_WEIGHTS)))
        shoe = self.rng.permutation(self._shoes[index])
        return TwelveRound(PLAYER_HANDS[index], (int(shoe[0]), int(shoe[1])), shoe[2:])
