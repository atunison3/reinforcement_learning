"""Project 10: finite-shoe blackjack, split-round returns, and Hi-Lo counting.

Run: python -m reinforcement_learning.projects.project010 --episodes 10000
Read: docs/projects/project10.md

All four actions are always offered. Invalid game decisions end the whole
round with -100. Valid rounds pay the sum of all hand outcomes at termination,
so undiscounted Monte Carlo returns credit a split with both descendants.
Dealer blackjack is revealed before any player action; these rounds are
terminated and discarded from learning, without an action penalty.
"""

import argparse
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from itertools import accumulate

from reinforcement_learning.blackjack import hand_value

INVALID_REWARD = -100.0
DECK = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10) * 4


class Action(IntEnum):
    HIT = 0
    STAND = 1
    DOUBLE = 2
    SPLIT = 3


class HandCategory(Enum):
    BELOW_8 = "<8"
    SIX_TWO = "six two"
    FIVE_THREE = "five three"
    FOUR_FOUR = "four four"
    EIGHT = "8 (other composition)"
    NINE = "9"
    TEN = "10"
    ELEVEN = "11"
    TWELVE = "12"
    THIRTEEN = "13"
    FOURTEEN = "14"
    FIFTEEN = "15"
    SIXTEEN = "16"
    ABOVE_16 = ">16"
    ACE_ACE = "ace ace"
    TWO_TWO = "two two"
    THREE_THREE = "three three"
    FIVE_FIVE = "five five"
    SIX_SIX = "six six"
    SEVEN_SEVEN = "seven seven"
    EIGHT_EIGHT = "eight eight"
    NINE_NINE = "nine nine"
    TEN_TEN = "ten ten"
    ACE_TWO = "ace two"
    ACE_THREE = "ace three"
    ACE_FOUR = "ace four"
    ACE_FIVE = "ace five"
    ACE_SIX = "ace six"
    ACE_SEVEN = "ace seven"
    ACE_EIGHT = "ace eight"
    BLACKJACK = "blackjack"


PAIRS = (
    HandCategory.ACE_ACE,
    HandCategory.TWO_TWO,
    HandCategory.THREE_THREE,
    HandCategory.FOUR_FOUR,
    HandCategory.FIVE_FIVE,
    HandCategory.SIX_SIX,
    HandCategory.SEVEN_SEVEN,
    HandCategory.EIGHT_EIGHT,
    HandCategory.NINE_NINE,
    HandCategory.TEN_TEN,
)
SOFT_TOTALS = (
    HandCategory.ACE_TWO,
    HandCategory.ACE_THREE,
    HandCategory.ACE_FOUR,
    HandCategory.ACE_FIVE,
    HandCategory.ACE_SIX,
    HandCategory.ACE_SEVEN,
    HandCategory.ACE_EIGHT,
)


def categorize_hand(cards: Sequence[int], *, from_split: bool = False) -> HandCategory:
    """Map a live hand to the requested buckets, independent of card order.

    Pairs and natural blackjack take precedence over totals. Usable-ace totals
    13..19 use the ace-two..ace-eight buckets, including multi-card hands.
    The two named non-pair eights are composition-specific two-card hands.
    """
    total, soft = hand_value(cards)
    if len(cards) < 2 or total > 21:
        raise ValueError("Only live hands with at least two cards have a decision category")
    if len(cards) == 2:
        if total == 21 and not from_split:
            return HandCategory.BLACKJACK
        if cards[0] == cards[1]:
            return PAIRS[cards[0] - 1]
        if sorted(cards) == [2, 6]:
            return HandCategory.SIX_TWO
        if sorted(cards) == [3, 5]:
            return HandCategory.FIVE_THREE
    if soft and 13 <= total <= 19:
        return SOFT_TOTALS[total - 13]
    if total < 8:
        return HandCategory.BELOW_8
    if total > 16:
        return HandCategory.ABOVE_16
    return HandCategory(str(total) if total != 8 else "8 (other composition)")


class CountBucket(Enum):
    NON_POSITIVE = "<1"
    POSITIVE = ">0"


def hi_lo(card: int) -> int:
    """2..6: +1, 7..9: 0, tens and aces: -1."""
    if not 1 <= card <= 10:
        raise ValueError("Cards must be encoded as 1 (ace) through 10")
    if 2 <= card <= 6:
        return 1
    return -1 if card in (1, 10) else 0


@dataclass(frozen=True)
class BlackjackState:
    hand: HandCategory
    can_double: bool
    can_split: bool
    dealer_upcard: int
    count: CountBucket


@dataclass
class PlayerHand:
    cards: list[int]
    stake: int = 1
    from_split: bool = False

    @property
    def total(self) -> int:
        return hand_value(self.cards)[0]

    @property
    def blackjack(self) -> bool:
        return not self.from_split and len(self.cards) == 2 and self.total == 21


@dataclass(frozen=True)
class StepResult:
    state: BlackjackState | None
    reward: float
    terminated: bool
    hand_rewards: tuple[float, ...] = ()
    invalid_reason: str | None = None


class BlackjackEnvironment:
    """Six decks by default, S17, dealer peek, double after split, up to four hands.

    An episode is a whole round, not an individual split hand. The finite shoe
    and running count persist across rounds; reshuffles happen between rounds.
    Aces can be hit/re-split like other pairs. Split 21 is not natural blackjack.
    """

    actions = tuple(Action)  # Deliberately not masked by the state flags.

    def __init__(
        self,
        decks: int = 6,
        max_hands: int = 4,
        penetration: float = 0.75,
        seed: int | None = None,
    ) -> None:
        if not isinstance(decks, int) or isinstance(decks, bool) or decks < 1:
            raise ValueError("decks must be a positive integer")
        if not isinstance(max_hands, int) or isinstance(max_hands, bool) or not 1 <= max_hands <= 4:
            raise ValueError("max_hands must be an integer from one to four")
        if not 0.0 < penetration < 1.0:
            raise ValueError("penetration must be strictly between zero and one")
        self.decks = decks
        self.max_hands = max_hands
        self.penetration = penetration
        # Reproducible simulation, not cryptography.
        self._rng = random.Random(seed)  # nosec B311
        self._full_shoe = DECK * decks
        # Each finished hand's sum of base card values is at most 31: a
        # non-busted total <=21 followed by at most a ten. Bound the number
        # of cards all player hands plus the dealer could consume, using the
        # smallest cards in a full shoe. Keep that many before starting a round.
        budget = 31 * (max_hands + 1)
        self._round_reserve = sum(total <= budget for total in accumulate(sorted(self._full_shoe)))
        self._shoe: list[int] = []
        self._running_count = 0
        self.shuffle_count = 0
        self._hands: list[PlayerHand] = []
        self._dealer: list[int] = []
        self._hole_revealed = False
        self._current = 0
        self._active = False
        self.discard_reason: str | None = None
        self._shuffle()

    @property
    def running_count(self) -> int:
        return self._running_count

    @property
    def cards_remaining(self) -> int:
        return len(self._shoe)

    def _shuffle(self) -> None:
        self._shoe = list(self._full_shoe)
        self._rng.shuffle(self._shoe)
        self._running_count = 0
        self.shuffle_count += 1

    def _draw(self, *, exposed: bool = True) -> int:
        if not self._shoe:
            raise RuntimeError("Shoe exhausted; a round must never reshuffle mid-hand")
        card = self._shoe.pop()
        if exposed:
            self._running_count += hi_lo(card)
        return card

    def _reveal_hole(self) -> None:
        if not self._hole_revealed:
            self._running_count += hi_lo(self._dealer[1])
            self._hole_revealed = True

    def reset(self, *, shoe: Sequence[int] | None = None) -> BlackjackState | None:
        """Deal and peek before returning a player decision state.

        Deal order: player, player, dealer upcard, dealer hidden hole card.
        With an ace or ten up, peek for blackjack. If found, reveal the hole,
        terminate, set discard_reason="dealer_blackjack", and return None.
        This includes mutual naturals: no action, reward, or learning sample.
        A negative peek leaves the hole hidden and uncounted until settlement.

        Optional shoe is a complete, ordered fresh shoe (first element drawn
        first), useful for reproducible examples/tests. It must have exactly
        the configured decks' card multiplicities, and resets the count.
        """
        if self._active:
            raise RuntimeError("Finish the active round before resetting")
        if shoe is not None:
            if Counter(shoe) != Counter(self._full_shoe):
                raise ValueError("shoe must contain exactly the configured decks' cards")
            self._shoe = list(reversed(shoe))
            self._running_count = 0
            self.shuffle_count += 1
        elif (
            self.cards_remaining <= len(self._full_shoe) * (1.0 - self.penetration)
            or self.cards_remaining < self._round_reserve
        ):
            self._shuffle()
        self._hands = [PlayerHand([self._draw(), self._draw()])]
        self._dealer = [self._draw(), self._draw(exposed=False)]
        self._hole_revealed = False
        self._current = 0
        self.discard_reason = None
        if self._dealer[0] in (1, 10) and hand_value(self._dealer)[0] == 21:
            self._reveal_hole()
            self._active = False
            self.discard_reason = "dealer_blackjack"
            return None
        self._active = True
        return self._state()

    def _state(self) -> BlackjackState:
        hand = self._hands[self._current]
        return BlackjackState(
            hand=categorize_hand(hand.cards, from_split=hand.from_split),
            can_double=len(hand.cards) == 2 and not hand.blackjack,
            can_split=(len(hand.cards) == 2 and hand.cards[0] == hand.cards[1] and len(self._hands) < self.max_hands),
            dealer_upcard=self._dealer[0],
            count=CountBucket.POSITIVE if self.running_count > 0 else CountBucket.NON_POSITIVE,
        )

    def _invalid(self, reason: str) -> StepResult:
        self._reveal_hole()
        self._active = False
        return StepResult(None, INVALID_REWARD, True, invalid_reason=reason)

    def step(self, action: Action) -> StepResult:
        """Return zero until the round ends, then its total net payoff.

        A disallowed game action ends the entire round with -100 (not -100
        plus previous hand outcomes). Unknown action identifiers raise errors.
        """
        if not self._active:
            raise RuntimeError("Call reset() before acting, and after terminal results")
        action = Action(action)
        hand = self._hands[self._current]
        state = self._state()
        if hand.blackjack and action is not Action.STAND:
            return self._invalid("A natural blackjack must stand")
        if action is Action.DOUBLE and not state.can_double:
            return self._invalid("Doubling requires a two-card, non-blackjack hand")
        if action is Action.SPLIT and not state.can_split:
            return self._invalid("Splitting requires a pair and room below max_hands")

        if action is Action.SPLIT:
            card = hand.cards[0]
            # Both child hands receive a visible second card immediately.
            self._hands[self._current : self._current + 1] = [
                PlayerHand([card, self._draw()], from_split=True),
                PlayerHand([card, self._draw()], from_split=True),
            ]
            return StepResult(self._state(), 0.0, False)
        if action in (Action.HIT, Action.DOUBLE):
            hand.cards.append(self._draw())
            if action is Action.DOUBLE:
                hand.stake = 2
            elif hand.total <= 21:
                return StepResult(self._state(), 0.0, False)
        # Stand, double, or bust completes this hand, not necessarily the round.
        self._current += 1
        if self._current < len(self._hands):
            return StepResult(self._state(), 0.0, False)
        return self._settle()

    def _settle(self) -> StepResult:
        self._reveal_hole()
        dealer_total = hand_value(self._dealer)[0]
        # Dealer naturals have already been discarded by reset(), before play.
        if any(hand.total <= 21 and not hand.blackjack for hand in self._hands):
            while dealer_total < 17:
                self._dealer.append(self._draw())
                dealer_total = hand_value(self._dealer)[0]
        rewards: list[float] = []
        for hand in self._hands:
            if hand.total > 21:
                reward = -float(hand.stake)
            elif hand.blackjack:
                reward = 1.5
            elif dealer_total > 21 or hand.total > dealer_total:
                reward = float(hand.stake)
            elif hand.total < dealer_total:
                reward = -float(hand.stake)
            else:
                reward = 0.0
            rewards.append(reward)
        self._active = False
        return StepResult(None, sum(rewards), True, tuple(rewards))


@dataclass(frozen=True)
class Experience:
    state: BlackjackState
    action: Action
    reward: float


class MonteCarloAgent:
    """Every-visit action-value averages with epsilon-greedy, unmasked actions.

    Returns use gamma=1 and stop at the end of the whole round, not at a
    split-hand boundary. This simple state aggregation is not a full MDP state.
    """

    def __init__(self, epsilon: float = 0.1, seed: int | None = None) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        self.epsilon = epsilon
        # Reproducible exploration, not cryptography.
        self._rng = random.Random(seed)  # nosec B311
        self.q: dict[BlackjackState, list[float]] = {}
        self.visits: dict[BlackjackState, list[int]] = {}

    def choose_action(self, state: BlackjackState) -> Action:
        values = self.q.get(state, [0.0] * len(Action))
        if self._rng.random() < self.epsilon:
            return self._rng.choice(tuple(Action))
        best = max(values)
        return self._rng.choice([action for action in Action if values[action] == best])

    def learn(self, episode: Sequence[Experience]) -> None:
        total_return = 0.0
        for experience in reversed(episode):
            total_return += experience.reward
            values = self.q.setdefault(experience.state, [0.0] * len(Action))
            visits = self.visits.setdefault(experience.state, [0] * len(Action))
            action = experience.action
            visits[action] += 1
            values[action] += (total_return - values[action]) / visits[action]


def run_episode(environment: BlackjackEnvironment, agent: MonteCarloAgent) -> tuple[Experience, ...]:
    """Play and learn one round, or return () for a discarded dealer blackjack."""
    state = environment.reset()
    if state is None:
        return ()  # No policy call, fabricated reward, or learning update.
    episode: list[Experience] = []
    while True:
        action = agent.choose_action(state)
        result = environment.step(action)
        episode.append(Experience(state, action, result.reward))
        if result.terminated:
            break
        if result.state is None:
            raise RuntimeError("Nonterminal results require a state")
        state = result.state
    agent.learn(episode)
    return tuple(episode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    environment = BlackjackEnvironment(seed=args.seed)
    agent = MonteCarloAgent(seed=args.seed)
    total = 0.0
    invalid = 0
    discarded = 0
    for _ in range(args.episodes):
        episode = run_episode(environment, agent)
        if not episode:
            discarded += 1
            continue
        reward = sum(experience.reward for experience in episode)
        total += reward
        invalid += reward == INVALID_REWARD
    training_rounds = args.episodes - discarded
    print(f"Dealt rounds: {args.episodes}; training rounds: {training_rounds}; epsilon: {agent.epsilon:g}")
    print(f"Discarded dealer-blackjack rounds: {discarded}")
    mean_return = f"{total / training_rounds:.3f}" if training_rounds else "n/a (no training rounds)"
    print(f"Mean training return (including penalties, excluding discarded rounds): {mean_return}")
    print(f"Invalid-action rounds: {invalid}; observed states: {len(agent.q)}")
    print(f"Shoe shuffles: {environment.shuffle_count}; running count: {environment.running_count}")
    print("Exploratory training statistics are not an estimate of casino profitability.")


if __name__ == "__main__":
    main()
