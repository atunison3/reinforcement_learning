"""Project 10: masked Double Q-learning for finite-shoe blackjack.

Run: python -m reinforcement_learning.projects.project010 --episodes 100000 --epsilon 0.1
Read: docs/projects/project10.md

Exact totals, usable aces, and split-round context distinguish decision states.
Dealer naturals are discarded before decisions. A round includes every split
hand; its aggregate net payoff is delivered only at termination. One-step
Double Q targets use a greedy legal continuation, not subsequent exploratory
returns. Both estimators start at 10. The current epsilon default remains 0.
Markdown and versioned CSV checkpoints are saved every 100,000 dealt rounds
and at completion. Legacy Monte Carlo CSVs cannot be resumed by this model.
"""

import argparse
import csv
import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from functools import lru_cache
from itertools import accumulate, product
from pathlib import Path
from time import perf_counter

from reinforcement_learning.blackjack import hand_value

INITIAL_ACTION_VALUE = 10.0
FORMAT_VERSION = "2"
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
    """Retain composition labels; exact total and usable ace are separate features."""
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


@dataclass(frozen=True, slots=True)
class BlackjackState:
    hand: HandCategory
    can_double: bool
    can_split: bool
    dealer_upcard: int
    count: CountBucket
    total: int
    usable_ace: bool
    max_hands: int = 4
    # Completed split hands need only (final total, stake) for settlement.
    completed_hands: tuple[tuple[int, int], ...] = ()
    # Pending split hands have exactly two exposed cards and unit stakes.
    pending_hands: tuple[tuple[int, int], ...] = ()


def allowed_actions(state: BlackjackState) -> tuple[Action, ...]:
    """Enforce legality, not strategy; hitting a hard 20 remains available."""
    if state.hand is HandCategory.BLACKJACK:
        return (Action.STAND,)
    actions = [Action.HIT, Action.STAND]
    if state.can_double:
        actions.append(Action.DOUBLE)
    if state.can_split:
        actions.append(Action.SPLIT)
    return tuple(actions)


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


class BlackjackEnvironment:
    """Finite shoe, S17, dealer peek, double after split, and re-splitting aces.

    One episode includes all player hands. No reshuffle occurs mid-round.
    The state includes only player-visible information, never the dealer hole.
    """

    actions = tuple(Action)

    def __init__(self, decks: int = 6, max_hands: int = 4, penetration: float = 0.75, seed: int | None = None) -> None:
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
        # Bound cards consumed by all hands, each with base-card sum at most 31.
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
        """Deal player/player/upcard/hole, peek, and discard dealer naturals.

        Optional shoe is a complete physical shoe in draw order. A negative
        peek does not reveal/count the hole card. Discards consume all four cards.
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
        total, usable_ace = hand_value(hand.cards)
        return BlackjackState(
            hand=categorize_hand(hand.cards, from_split=hand.from_split),
            can_double=len(hand.cards) == 2 and not hand.blackjack,
            can_split=(len(hand.cards) == 2 and hand.cards[0] == hand.cards[1] and len(self._hands) < self.max_hands),
            dealer_upcard=self._dealer[0],
            count=CountBucket.POSITIVE if self.running_count > 0 else CountBucket.NON_POSITIVE,
            total=total,
            usable_ace=usable_ace,
            max_hands=self.max_hands,
            completed_hands=tuple((previous.total, previous.stake) for previous in self._hands[: self._current]),
            pending_hands=tuple(
                (min(pending.cards), max(pending.cards)) for pending in self._hands[self._current + 1 :]
            ),
        )

    def step(self, action: Action) -> StepResult:
        """Return zero until settlement; reject illegal calls without side effects."""
        if not self._active:
            raise RuntimeError("Call reset() before acting, and after terminal results")
        action = Action(action)
        state = self._state()
        if action not in allowed_actions(state):
            raise ValueError(f"{action.name} is not allowed in state {state}")
        hand = self._hands[self._current]
        if action is Action.SPLIT:
            card = hand.cards[0]
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
        self._current += 1
        if self._current < len(self._hands):
            return StepResult(self._state(), 0.0, False)
        return self._settle()

    def _settle(self) -> StepResult:
        self._reveal_hole()
        dealer_total = hand_value(self._dealer)[0]
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
    next_state: BlackjackState | None = None  # None means terminal, not a discarded deal.


class DoubleQLearningAgent:
    """Online Double Q-learning, gamma=1, fixed alpha, and masked epsilon-greedy.

    Randomly update A or B. Select a greedy next action using that estimator,
    evaluate it using the other estimator. Behavior uses the mean of A and B.
    Update counts are diagnostics, not Monte Carlo sample-average weights.
    """

    def __init__(self, seed: int | None = None, *, epsilon: float = 0.0, alpha: float = 0.1) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be greater than zero and at most one")
        self.epsilon = epsilon
        self.alpha = alpha
        # Reproducible behavior/update streams, not cryptography.
        self._rng = random.Random(seed)  # nosec B311
        self._update_rng = random.Random(None if seed is None else seed + 1)  # nosec B311
        self.q_a: dict[BlackjackState, list[float]] = {}
        self.q_b: dict[BlackjackState, list[float]] = {}
        self.visits_a: dict[BlackjackState, list[int]] = {}
        self.visits_b: dict[BlackjackState, list[int]] = {}

    def action_values(self, state: BlackjackState) -> list[float]:
        a = self.q_a.get(state, [INITIAL_ACTION_VALUE] * len(Action))
        b = self.q_b.get(state, [INITIAL_ACTION_VALUE] * len(Action))
        return [(left + right) / 2 for left, right in zip(a, b, strict=True)]

    def action_visits(self, state: BlackjackState) -> list[int]:
        a = self.visits_a.get(state, [0] * len(Action))
        b = self.visits_b.get(state, [0] * len(Action))
        return [left + right for left, right in zip(a, b, strict=True)]

    def choose_action(self, state: BlackjackState) -> Action:
        actions = allowed_actions(state)
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return self._rng.choice(actions)
        values = self.action_values(state)
        best = max(values[action] for action in actions)
        return self._rng.choice([action for action in actions if values[action] == best])

    def learn(self, experience: Experience) -> None:
        if experience.action not in allowed_actions(experience.state):
            raise ValueError("Cannot learn an illegal action")
        if not math.isfinite(experience.reward):
            raise ValueError("Reward must be finite")
        state, action = experience.state, experience.action
        for table in (self.q_a, self.q_b):
            table.setdefault(state, [INITIAL_ACTION_VALUE] * len(Action))
        for visits in (self.visits_a, self.visits_b):
            visits.setdefault(state, [0] * len(Action))
        if self._update_rng.random() < 0.5:
            selected, evaluated, visits = self.q_a, self.q_b, self.visits_a
        else:
            selected, evaluated, visits = self.q_b, self.q_a, self.visits_b
        target = experience.reward
        if experience.next_state is not None:
            next_state = experience.next_state
            candidates = allowed_actions(next_state)
            selection_values = selected.get(next_state, [INITIAL_ACTION_VALUE] * len(Action))
            best = max(selection_values[candidate] for candidate in candidates)
            next_action = self._update_rng.choice(
                [candidate for candidate in candidates if selection_values[candidate] == best]
            )
            target += evaluated.get(next_state, [INITIAL_ACTION_VALUE] * len(Action))[next_action]
        selected[state][action] += self.alpha * (target - selected[state][action])
        visits[state][action] += 1


def run_episode(environment: BlackjackEnvironment, agent: DoubleQLearningAgent) -> tuple[Experience, ...]:
    """Update once per transition, before choosing the next behavior action."""
    state = environment.reset()
    if state is None:
        return ()
    episode: list[Experience] = []
    while True:
        action = agent.choose_action(state)
        result = environment.step(action)
        if not result.terminated and result.state is None:
            raise RuntimeError("Nonterminal results require a state")
        experience = Experience(state, action, result.reward, result.state)
        agent.learn(experience)
        episode.append(experience)
        if result.terminated:
            return tuple(episode)
        state = result.state
        if state is None:
            raise RuntimeError("Missing next state")


ACTION_CODES = {Action.HIT: "H", Action.STAND: "S", Action.DOUBLE: "D", Action.SPLIT: "P"}
DEALER_UPCARDS = (*range(2, 11), 1)
TABLE_FLAGS = ((False, False), (True, False), (True, True))
STATE_COLUMNS = [
    "hand",
    "total",
    "usable_ace",
    "can_double",
    "can_split",
    "dealer_upcard",
    "count",
    "max_hands",
    "completed_hands",
    "pending_hands",
]
STRATEGY_COLUMNS = ["format_version", "algorithm", *STATE_COLUMNS, "recommendation", "state_visits"] + [
    field
    for action in Action
    for field in (
        f"q_{action.name.lower()}",
        f"q_a_{action.name.lower()}",
        f"visits_a_{action.name.lower()}",
        f"q_b_{action.name.lower()}",
        f"visits_b_{action.name.lower()}",
    )
]


@lru_cache(maxsize=8)
def _hand_features(from_split: bool) -> frozenset[tuple[HandCategory, int, bool, bool]]:
    """Two/three-card hands cover every live total/softness/category/eligibility.

    More cards introduce no new active-hand features; busts are not decisions.
    """
    features = set()
    for size in (2, 3):
        for cards in product(range(1, 11), repeat=size):
            total, soft = hand_value(cards)
            if total <= 21:
                category = categorize_hand(cards, from_split=from_split)
                features.add((category, total, soft, size == 2 and category is not HandCategory.BLACKJACK))
    return frozenset(features)


def table_states(can_double: bool, can_split: bool, *, max_hands: int = 4) -> tuple[BlackjackState, ...]:
    """All possible single-hand observations, with exact totals (including unseen ones)."""
    states = [
        BlackjackState(hand, double, split, 2, CountBucket.NON_POSITIVE, total, soft, max_hands)
        for hand, total, soft, double in _hand_features(False)
        if (split := (hand in PAIRS and max_hands > 1)) == can_split and double == can_double
    ]
    return tuple(sorted(states, key=lambda state: (state.usable_ace, state.total, state.hand.value)))


def strategy_cell(agent: DoubleQLearningAgent, state: BlackjackState) -> str:
    """Greedy legal mean-Q choice; coverage considers both estimators."""
    allowed = allowed_actions(state)
    counts = agent.action_visits(state)
    if not any(counts[action] for action in allowed):
        return "?"
    values = agent.action_values(state)
    best = max(values[action] for action in allowed)
    actions = [action for action in allowed if values[action] == best]
    if any(counts[action] == 0 for action in actions):
        return "?"
    a = agent.visits_a.get(state, [0] * len(Action))
    b = agent.visits_b.get(state, [0] * len(Action))
    incomplete = any(a[action] == 0 or b[action] == 0 for action in allowed)
    return "/".join(ACTION_CODES[action] for action in actions) + ("*" if incomplete else "")


def format_strategy_table(agent: DoubleQLearningAgent, *, summary: Sequence[str] = ()) -> str:
    """Report exact single-hand strategy; never merge distinct split contexts."""
    lines = ["# Project 10 — Double Q-Learning Strategy", ""]
    lines.extend(f"- {line}" for line in summary)
    contextual = sum(bool(state.completed_hands or state.pending_hands) for state in agent.q_a)
    lines.extend(
        [
            "",
            f"Policy: masked epsilon-greedy; epsilon: {agent.epsilon:g}; alpha: {agent.alpha:g}; gamma: 1; initial Q: 10.",
            "Action masking: enabled (legal actions only). Choices use the mean of the two estimators.",
            "`H` = hit; `S` = stand; `D` = double; `P` = split; `A` = dealer ace.",
            "`?` = unseen legal choices or an untried legal choice leads/ties; `*` = incomplete coverage in either estimator; `/` = ties.",
            "Unmarked choices are not proof of convergence or optimality.",
            "These matrices apply only to a single unsplit hand: no completed or pending hands.",
            f"Split-context states: {contextual}. Their exact contexts and choices are in the CSV; do not reuse these matrices for split children.",
            "",
        ]
    )
    limits = sorted({state.max_hands for state in agent.q_a} or {4})
    for limit in limits:
        for count in CountBucket:
            for can_double, can_split in TABLE_FLAGS:
                rows = table_states(can_double, can_split, max_hands=limit)
                if not rows:
                    continue
                lines.extend(
                    [
                        f"## Count {count.value}; double {'yes' if can_double else 'no'}; split {'yes' if can_split else 'no'}; max hands {limit}",
                        "",
                        "| Hand / dealer upcard | "
                        + " | ".join("A" if card == 1 else str(card) for card in DEALER_UPCARDS)
                        + " |",
                        "| --- | " + " | ".join("---" for _ in DEALER_UPCARDS) + " |",
                    ]
                )
                for prototype in rows:
                    label = f"{'Soft' if prototype.usable_ace else 'Hard'} {prototype.total}"
                    if prototype.hand in (
                        *PAIRS,
                        HandCategory.SIX_TWO,
                        HandCategory.FIVE_THREE,
                        HandCategory.BLACKJACK,
                    ):
                        label += f" ({prototype.hand.value})"
                    cells = [
                        strategy_cell(agent, replace(prototype, count=count, dealer_upcard=card))
                        for card in DEALER_UPCARDS
                    ]
                    lines.append(f"| {label} | " + " | ".join(cells) + " |")
                lines.append("")
    return "\n".join(lines)


def _state_fields(state: BlackjackState) -> list[str]:
    return [
        state.hand.value,
        str(state.total),
        str(state.usable_ace),
        str(state.can_double),
        str(state.can_split),
        str(state.dealer_upcard),
        state.count.value,
        str(state.max_hands),
        json.dumps(state.completed_hands, separators=(",", ":")),
        json.dumps(state.pending_hands, separators=(",", ":")),
    ]


def save_strategy_tables(
    agent: DoubleQLearningAgent, directory: Path, *, summary: Sequence[str] = ()
) -> tuple[Path, Path]:
    """Save exact observed states and both estimators; unobserved matrix cells stay '?'."""
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / "project_10_strategy.md"
    csv_path = directory / "project_10_strategy.csv"
    markdown_path.write_text(format_strategy_table(agent, summary=summary), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(STRATEGY_COLUMNS)
        for state in sorted(agent.q_a, key=_state_fields):
            values = agent.action_values(state)
            counts = agent.action_visits(state)
            fields: list[str | float | int] = [
                FORMAT_VERSION,
                "double_q",
                *_state_fields(state),
                strategy_cell(agent, state),
                sum(counts),
            ]
            for action in Action:
                a, b = agent.visits_a[state][action], agent.visits_b[state][action]
                fields.extend(
                    [
                        values[action] if counts[action] else "",
                        agent.q_a[state][action] if a else "",
                        a,
                        agent.q_b[state][action] if b else "",
                        b,
                    ]
                )
            writer.writerow(fields)
    return markdown_path, csv_path


def _parse_pairs(text: str) -> tuple[tuple[int, int], ...]:
    pairs = json.loads(text)
    if not isinstance(pairs, list) or len(pairs) > 3:
        raise ValueError("hand context must be a list of at most three pairs")
    result = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2 or any(type(value) is not int for value in pair):
            raise ValueError("hand context entries must be integer pairs")
        result.append((pair[0], pair[1]))
    return tuple(result)


def _parse_state(row: dict[str, str]) -> BlackjackState:
    for key in ("usable_ace", "can_double", "can_split"):
        if row[key] not in ("True", "False"):
            raise ValueError(f"{key} must be True or False")
    state = BlackjackState(
        HandCategory(row["hand"]),
        row["can_double"] == "True",
        row["can_split"] == "True",
        int(row["dealer_upcard"]),
        CountBucket(row["count"]),
        int(row["total"]),
        row["usable_ace"] == "True",
        int(row["max_hands"]),
        _parse_pairs(row["completed_hands"]),
        _parse_pairs(row["pending_hands"]),
    )
    hands = 1 + len(state.completed_hands) + len(state.pending_hands)
    if not 1 <= hands <= state.max_hands <= 4 or state.dealer_upcard not in DEALER_UPCARDS:
        raise ValueError("invalid hand limit or dealer upcard")
    if (state.hand, state.total, state.usable_ace, state.can_double) not in _hand_features(hands > 1):
        raise ValueError("impossible hand category, exact total, softness, or double flag")
    if state.can_split != (state.hand in PAIRS and hands < state.max_hands):
        raise ValueError("split flag inconsistent with pair and hand limit")
    if any(not 4 <= total <= 31 or stake not in (1, 2) for total, stake in state.completed_hands):
        raise ValueError("invalid completed total or stake")
    if any(not 1 <= first <= second <= 10 for first, second in state.pending_hands):
        raise ValueError("pending hands require two sorted card values from 1 to 10")
    return state


def load_strategy(agent: DoubleQLearningAgent, path: Path) -> None:
    """Validate versioned Double Q data completely before replacing either table.

    Legacy Monte Carlo means lack exact states and the second estimator: there
    is no faithful automatic conversion. Behavior settings and RNG are not loaded.
    """
    q_a: dict[BlackjackState, list[float]] = {}
    q_b: dict[BlackjackState, list[float]] = {}
    visits_a: dict[BlackjackState, list[int]] = {}
    visits_b: dict[BlackjackState, list[int]] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source, strict=True)
        header = next(reader, None)
        if header is not None and "format_version" not in header and "q_hit" in header:
            raise ValueError(
                "Legacy Monte Carlo CSV is incompatible with exact-state Double Q-learning; start fresh without --load-strategy"
            )
        if header is None or len(header) != len(STRATEGY_COLUMNS) or set(header) != set(STRATEGY_COLUMNS):
            raise ValueError("expected version-2 Double Q strategy CSV columns")
        for fields in reader:
            try:
                if len(fields) != len(header):
                    raise ValueError("wrong number of columns")
                row = {key: value.strip() for key, value in zip(header, fields, strict=True)}
                if row["format_version"] != FORMAT_VERSION or row["algorithm"] != "double_q":
                    raise ValueError("unsupported format version or learning algorithm")
                state = _parse_state(row)
                if state in q_a:
                    raise ValueError("duplicate state")
                qa: list[float] = []
                qb: list[float] = []
                na: list[int] = []
                nb: list[int] = []
                for action in Action:
                    name = action.name.lower()
                    for label, values, counts in (("a", qa, na), ("b", qb, nb)):
                        count = int(row[f"visits_{label}_{name}"])
                        text = row[f"q_{label}_{name}"]
                        if count < 0 or (count == 0) != (text == ""):
                            raise ValueError("Q must be blank exactly when its update count is zero")
                        if action not in allowed_actions(state) and count:
                            raise ValueError("illegal actions cannot have updates in a masked Double Q checkpoint")
                        value = float(text) if text else INITIAL_ACTION_VALUE
                        if not math.isfinite(value):
                            raise ValueError("Q estimates must be finite")
                        values.append(value)
                        counts.append(count)
                    text = row[f"q_{name}"]
                    if (na[-1] + nb[-1] == 0) != (text == ""):
                        raise ValueError("combined Q coverage is inconsistent")
                    if text and (
                        not math.isfinite(float(text)) or not math.isclose(float(text), (qa[-1] + qb[-1]) / 2)
                    ):
                        raise ValueError("combined Q must equal the mean of both estimators")
                if int(row["state_visits"]) != sum(na) + sum(nb) or sum(na) + sum(nb) == 0:
                    raise ValueError("state_visits must equal the positive total update count")
                q_a[state], q_b[state], visits_a[state], visits_b[state] = qa, qb, na, nb
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}: row {reader.line_num}: {exc}") from exc
    agent.q_a, agent.q_b, agent.visits_a, agent.visits_b = q_a, q_b, visits_a, visits_b


def main() -> None:
    program_started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.0, help="Legal-action exploration probability (default: 0)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Double Q step size (default: 0.1)")
    parser.add_argument(
        "--load-strategy", type=Path, help="Resume a version-2 Double Q CSV, not a legacy Monte Carlo CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "projects" / "assets",
        help="Directory for Markdown/CSV (overwritten every 100,000 rounds and at completion)",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    try:
        agent = DoubleQLearningAgent(seed=args.seed, epsilon=args.epsilon, alpha=args.alpha)
    except ValueError as exc:
        parser.error(str(exc))
    loaded_summary: list[str] = []
    if args.load_strategy is not None:
        try:
            load_strategy(agent, args.load_strategy)
        except (OSError, ValueError, csv.Error) as exc:
            parser.error(f"Cannot load strategy: {exc}")
        loaded_visits = sum(sum(agent.action_visits(state)) for state in agent.q_a)
        loaded_summary.append(
            f"Loaded strategy: {args.load_strategy}; states: {len(agent.q_a)}; action visits: {loaded_visits}"
        )
        print(loaded_summary[0], flush=True)
    environment = BlackjackEnvironment(seed=args.seed)
    total = 0.0
    discarded = 0

    def training_summary(dealt: int, training_seconds: float) -> list[str]:
        training_rounds = dealt - discarded
        mean_return = f"{total / training_rounds:.3f}" if training_rounds else "n/a (no training rounds)"
        return [
            *loaded_summary,
            f"Dealt rounds: {dealt}; training rounds: {training_rounds}",
            f"Algorithm: Double Q-learning; epsilon: {agent.epsilon:g}; alpha: {agent.alpha:g}; gamma: 1; initial Q: 10",
            f"Training time this run: {training_seconds:.3f} s",
            f"Average time per 1,000 dealt episodes (normalized): {training_seconds * 1000 / dealt:.3f} s",
            f"Seed: {args.seed}; decks: {environment.decks}; max hands: {environment.max_hands}",
            f"Discarded dealer-blackjack rounds: {discarded}",
            f"Mean training return (excluding discarded rounds): {mean_return}",
            f"Observed states: {len(agent.q_a)}",
            f"Shoe shuffles: {environment.shuffle_count}; running count: {environment.running_count}",
            "Exploratory training statistics are not an estimate of casino profitability.",
        ]

    training_started = block_started = perf_counter()
    for dealt in range(1, args.episodes + 1):
        episode = run_episode(environment, agent)
        if not episode:
            discarded += 1
        else:
            total += sum(experience.reward for experience in episode)
        if dealt % 100000 == 0:
            elapsed = perf_counter() - block_started
            print(f"Episodes {dealt - 9999:,}-{dealt:,} (100,000 dealt): {elapsed:.3f} s", flush=True)
            block_started = perf_counter()
        if dealt % 1000000 == 0 and dealt < args.episodes:
            summary = training_summary(dealt, perf_counter() - training_started)
            markdown_path, csv_path = save_strategy_tables(agent, args.output_dir, summary=summary)
            print(f"Checkpoint after {dealt:,} dealt rounds: saved {markdown_path} and {csv_path}", flush=True)
    training_finished = perf_counter()
    training_seconds = training_finished - training_started
    remainder = args.episodes % 100000
    if remainder:
        print(
            f"Episodes {args.episodes - remainder + 1:,}-{args.episodes:,} "
            f"({remainder:,} dealt, partial block): {training_finished - block_started:.3f} s",
            flush=True,
        )
    summary = training_summary(args.episodes, training_seconds)
    print(format_strategy_table(agent, summary=summary), flush=True)
    markdown_path, csv_path = save_strategy_tables(agent, args.output_dir, summary=summary)
    print(f"Saved {markdown_path}\nSaved {csv_path}", flush=True)
    print(f"Total program time (including load, training, and reports): {perf_counter() - program_started:.3f} s")


if __name__ == "__main__":
    main()
