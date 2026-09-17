"""Exercise 8: Learn blackjack action values by one-decision associative search.

Run ``python -m reinforcement_learning.playground.exercise008 --rounds 1000000``.
A hit is deliberately followed by a forced stand, not another player decision.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from reinforcement_learning.blackjack import (
    MAX_HAND_VALUE,
    MIN_HAND_VALUE,
    N_CONTEXTS,
    Action,
    BlackjackState,
    OneDecisionBlackjack,
    reachable_states,
)
from reinforcement_learning.contextual_bandits import ContextualBanditAgent


def train(rounds: int = 1_000_000, epsilon: float = 0.2, seed: int = 42) -> ContextualBanditAgent:
    """Estimate E[reward | state, action] with epsilon-greedy sample averages.

    Context = (player total, usable ace, dealer upcard, doubling allowed).
    There is one selected action and one update per round; there is no
    bootstrap target, discount, or learned continuation after a player hit.
    The environment has its own RNG; policy selection uses NumPy's global RNG.
    """

    if rounds < 1:
        raise ValueError("`rounds` must be positive")
    agent = ContextualBanditAgent(N_CONTEXTS, len(Action), epsilon)
    np.random.seed(seed)
    environment = OneDecisionBlackjack(np.random.default_rng(seed))
    for _ in range(rounds):
        state = environment.reset()
        action = Action(agent.choose_action(state.index, state.legal_actions))
        reward = environment.step(action)
        agent.update(state.index, action, reward)
    return agent


def save_action_values(agent: ContextualBanditAgent, path: Path) -> None:
    """Export every reachable, legal state-action pair and its sample count.

    An unvisited pair has a blank estimate, not a misleading estimated zero.
    Impossible states and illegal doubles are deliberately absent.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("hand_value", "is_soft", "dealer_upcard", "can_double", "action", "estimated_value", "visits"))
        for state in reachable_states():
            for action in state.legal_actions:
                visits = int(agent.counts[state.index, action])
                value = f"{agent.q[state.index, action]:.6f}" if visits else ""
                writer.writerow(
                    (state.hand_value, state.is_soft, state.dealer_upcard, state.can_double, action.name, value, visits)
                )


def plot_policy(agent: ContextualBanditAgent, path: Path) -> None:
    """Plot the best estimated legal action; flag unreachable/unsampled cells."""

    states = set(reachable_states())
    cmap = ListedColormap(["tab:blue", "tab:orange", "tab:green", "lightgray"]).with_extremes(bad="#555555")
    norm = BoundaryNorm(np.arange(-0.5, 4.5), cmap.N)
    figure, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True, layout="constrained")
    for row, is_soft in enumerate((False, True)):
        for column, can_double in enumerate((True, False)):
            policy = np.full((MAX_HAND_VALUE - MIN_HAND_VALUE + 1, 10), np.nan)
            for total in range(MIN_HAND_VALUE, MAX_HAND_VALUE + 1):
                for upcard in range(1, 11):
                    state = BlackjackState(total, is_soft, upcard, can_double)
                    if state not in states:
                        continue
                    actions = list(state.legal_actions)
                    if np.any(agent.counts[state.index, actions] == 0):
                        best = 3  # Do not claim a learned policy before sampling every legal action.
                    else:
                        best = int(actions[int(np.argmax(agent.q[state.index, actions]))])
                    policy[total - MIN_HAND_VALUE, upcard - 1] = best
            axis = axes[row, column]
            image = axis.imshow(np.ma.masked_invalid(policy), origin="lower", cmap=cmap, norm=norm, aspect="auto")
            axis.set_title(f"{'Soft' if is_soft else 'Hard'} hand; {'can double' if can_double else 'cannot double'}")
            axis.set_xticks(range(10), ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
            axis.set_yticks(range(MAX_HAND_VALUE - MIN_HAND_VALUE + 1), range(MIN_HAND_VALUE, MAX_HAND_VALUE + 1))
            axis.set_xlabel("Dealer upcard")
            axis.set_ylabel("Player hand value")
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), ticks=range(4), shrink=0.7)
    colorbar.ax.set_yticklabels(["Hit, then stand", "Stand", "Double", "Not all actions sampled"])
    figure.legend(handles=[Patch(facecolor="#555555", label="Unreachable state")], loc="outside lower center")
    figure.suptitle("One-Decision Blackjack: Best Estimated Action (not full-game basic strategy)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Train a table, export estimates/counts, and plot the greedy policy."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1_000_000)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "exercises" / "assets",
    )
    args = parser.parse_args()
    agent = train(rounds=args.rounds, epsilon=args.epsilon, seed=args.seed)
    csv_path = args.output_dir / "exercise_8_action_values.csv"
    figure_path = args.output_dir / "exercise_8_policy.png"
    save_action_values(agent, csv_path)
    plot_policy(agent, figure_path)
    legal_pairs = [(state.index, action) for state in reachable_states() for action in state.legal_actions]
    visited = sum(int(agent.counts[state, action] > 0) for state, action in legal_pairs)
    print(
        f"Completed {args.rounds:,} one-action rounds; visited {visited}/{len(legal_pairs)} legal state-action pairs."
    )
    print(f"State-action estimates and visit counts: {csv_path}")
    print(f"Estimated policy (rare states may still be noisy): {figure_path}")


if __name__ == "__main__":
    main()
