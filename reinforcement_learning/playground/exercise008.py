"""Exercise 8: Figure-2.6-style parameter study on the blackjack-12 bandit.

There is one learning state and two arms: hit once then stand, or stand.
Run ``python -m reinforcement_learning.playground.exercise008`` to compare
four methods by average reward over their first 1,000 decisions.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from reinforcement_learning.blackjack import Action
from reinforcement_learning.blackjack_twelve import TwoDeckTwelveBandit

Method = Literal["epsilon_greedy", "gradient", "optimistic", "ucb"]
METHODS: tuple[Method, ...] = ("epsilon_greedy", "gradient", "optimistic", "ucb")
PARAMETERS = tuple(2.0**exponent for exponent in range(-7, 3))
LABELS = {
    "epsilon_greedy": r"$\varepsilon$-greedy ($\varepsilon$)",
    "gradient": r"Gradient bandit ($\alpha$)",
    "optimistic": r"Greedy, optimistic initialization ($Q_0$; $\alpha=0.1$)",
    "ucb": r"UCB ($c$)",
}
COLORS = {"epsilon_greedy": "tab:red", "gradient": "tab:green", "optimistic": "tab:orange", "ucb": "tab:blue"}


class BanditBatch:
    """Independent two-armed learners, vectorized across repeated trials.

    No card information enters this class. Each trial maintains just two
    estimates (or preferences), action counts, and a running reward baseline.
    """

    def __init__(self, trials: int, method: Method, parameter: float, rng: np.random.Generator) -> None:
        if trials < 1:
            raise ValueError("`trials` must be positive")
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        if not np.isfinite(parameter) or parameter <= 0:
            raise ValueError("`parameter` must be positive and finite")
        if method in ("epsilon_greedy", "gradient") and parameter > 1:
            raise ValueError("Epsilon and gradient alpha are capped at 1 in this experiment")
        self.method = method
        self.parameter = parameter
        self.rng = rng
        self.rows = np.arange(trials)
        self.q: NDArray[np.float64] = np.full((trials, 2), parameter if method == "optimistic" else 0.0)
        self.preferences: NDArray[np.float64] = np.zeros((trials, 2))
        self.counts: NDArray[np.int64] = np.zeros((trials, 2), dtype=np.int64)
        self.baseline: NDArray[np.float64] = np.zeros(trials)
        self.steps = 0

    def probabilities(self) -> NDArray[np.float64]:
        """Numerically stable softmax of each trial's two preferences."""

        shifted = self.preferences - np.max(self.preferences, axis=1, keepdims=True)
        weights: NDArray[np.float64] = np.exp(shifted)
        return weights / np.sum(weights, axis=1, keepdims=True)

    def choose_actions(self) -> NDArray[np.int64]:
        """Choose one arm per trial, with uniform tie-breaking."""

        trials = len(self.rows)
        if self.method == "gradient":
            return (self.rng.random(trials) >= self.probabilities()[:, 0]).astype(np.int64)
        scores = self.q.copy()
        if self.method == "ucb":
            scores += self.parameter * np.sqrt(np.log(self.steps + 1) / np.maximum(self.counts, 1))
            scores[self.counts == 0] = np.inf  # Every untried arm takes priority.
        actions = np.argmax(scores, axis=1).astype(np.int64)
        ties = scores[:, 0] == scores[:, 1]
        actions[ties] = self.rng.integers(2, size=int(np.sum(ties)))
        if self.method == "epsilon_greedy":
            explore = self.rng.random(trials) < self.parameter
            actions[explore] = self.rng.integers(2, size=int(np.sum(explore)))
        return actions

    def update(self, actions: NDArray[np.int64], rewards: NDArray[np.float64]) -> None:
        """Learn only from each trial's selected action and observed reward."""

        if actions.shape != self.baseline.shape or rewards.shape != self.baseline.shape:
            raise ValueError("Provide one action and reward per trial")
        if np.any((actions < 0) | (actions > 1)) or not np.all(np.isfinite(rewards)):
            raise ValueError("Actions must be 0 or 1 and rewards must be finite")
        self.counts[self.rows, actions] += 1
        if self.method == "gradient":
            # The baseline uses preceding rewards, excluding the current sample.
            advantage = rewards - self.baseline
            self.preferences -= self.parameter * advantage[:, None] * self.probabilities()
            self.preferences[self.rows, actions] += self.parameter * advantage
        else:
            rate = 0.1 if self.method == "optimistic" else 1.0 / self.counts[self.rows, actions]
            self.q[self.rows, actions] += rate * (rewards - self.q[self.rows, actions])
        self.steps += 1
        self.baseline += (rewards - self.baseline) / self.steps


@dataclass(frozen=True)
class SweepPoint:
    """Reward averaged over all steps and trials at one parameter setting."""

    method: Method
    parameter: float
    average_reward: float
    standard_error: float


@dataclass(frozen=True)
class ExperimentResult:
    """Parameter curves and empirical arm means from the common test deals."""

    points: tuple[SweepPoint, ...]
    action_means: tuple[float, float]
    trials: int
    steps: int
    seed: int


def run_experiments(
    trials: int = 2_000,
    steps: int = 1_000,
    seed: int = 42,
    parameters: tuple[float, ...] = PARAMETERS,
) -> ExperimentResult:
    """Compare all four methods using the same independently shuffled deals.

    A reward table contains both potential outcomes of each physical deal.
    Every learner receives only the selected arm's reward. Reusing deals
    across configurations reduces comparison noise and avoids redealing the
    same distribution 36 times. Algorithms never see future rewards, hand
    composition, or dealer cards, and are freshly initialized per setting.
    """

    if trials < 1 or steps < 1:
        raise ValueError("`trials` and `steps` must be positive")
    if not parameters or any(not np.isfinite(p) or not 1 / 128 <= p <= 4 for p in parameters):
        raise ValueError("Parameters must be finite and between 1/128 and 4")
    if tuple(sorted(set(parameters))) != parameters:
        raise ValueError("Parameters must be unique and increasing")
    configurations = [
        (method, parameter)
        for method in METHODS
        for parameter in parameters
        if method not in ("epsilon_greedy", "gradient") or parameter <= 1
    ]
    seeds = np.random.SeedSequence(seed).spawn(len(configurations) + 1)
    environment = TwoDeckTwelveBandit(np.random.default_rng(seeds[0]))
    rewards = np.empty((steps, trials, 2), dtype=np.int8)
    for step in range(steps):
        for trial in range(trials):
            deal = environment.deal()
            rewards[step, trial] = (deal.reward(Action.HIT), deal.reward(Action.STAND))

    points: list[SweepPoint] = []
    for (method, parameter), child_seed in zip(configurations, seeds[1:]):
        agent = BanditBatch(trials, method, parameter, np.random.default_rng(child_seed))
        total_rewards = np.zeros(trials)
        for step in range(steps):
            actions = agent.choose_actions()
            observed = rewards[step, agent.rows, actions].astype(np.float64)
            agent.update(actions, observed)
            total_rewards += observed
        trial_means = total_rewards / steps
        standard_error = float(np.std(trial_means, ddof=1) / np.sqrt(trials)) if trials > 1 else 0.0
        points.append(SweepPoint(method, parameter, float(np.mean(trial_means)), standard_error))
    action_means = (float(np.mean(rewards[:, :, 0])), float(np.mean(rewards[:, :, 1])))
    return ExperimentResult(tuple(points), action_means, trials, steps, seed)


def save_results(result: ExperimentResult, path: Path) -> None:
    """Write parameter scores, uncertainty, and run settings as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("method", "parameter", "average_reward", "standard_error", "trials", "steps", "seed"))
        for point in result.points:
            writer.writerow(
                (
                    point.method,
                    point.parameter,
                    point.average_reward,
                    point.standard_error,
                    result.trials,
                    result.steps,
                    result.seed,
                )
            )


def plot_results(result: ExperimentResult, path: Path) -> None:
    """Figure-2.6-style parameter plot; bands show +/- one standard error."""

    figure, axis = plt.subplots(figsize=(11, 7))
    for method in METHODS:
        points = [point for point in result.points if point.method == method]
        x = [point.parameter for point in points]
        y = np.array([point.average_reward for point in points])
        errors = np.array([point.standard_error for point in points])
        axis.plot(x, y, marker="o", color=COLORS[method], label=LABELS[method])
        axis.fill_between(x, y - errors, y + errors, color=COLORS[method], alpha=0.15)
    axis.axhline(max(result.action_means), color="black", linestyle="--", label="Best fixed arm (empirical reference)")
    axis.set_xscale("log", base=2)
    axis.set_xticks(PARAMETERS, ["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"])
    axis.set_xlabel(r"Parameter ($\varepsilon$, $\alpha$, $Q_0$, or $c$)")
    axis.set_ylabel(f"Average reward over the first {result.steps:,} steps")
    axis.set_title(f"Blackjack 12: Two-Armed Bandit Parameter Study ({result.trials:,} trials)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the single-state comparison and save the parameter study."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2_000)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "docs" / "exercises" / "assets"
    )
    args = parser.parse_args()
    result = run_experiments(trials=args.trials, steps=args.steps, seed=args.seed)
    csv_path = args.output_dir / "exercise_8_parameter_study.csv"
    figure_path = args.output_dir / "exercise_8_parameter_study.png"
    save_results(result, csv_path)
    plot_results(result, figure_path)
    print(f"Empirical arm means: hit then stand = {result.action_means[0]:.5f}; stand = {result.action_means[1]:.5f}")
    for method in METHODS:
        best = max((point for point in result.points if point.method == method), key=lambda point: point.average_reward)
        print(f"{method}: best sampled parameter = {best.parameter:g}, average reward = {best.average_reward:.5f}")
    print(f"Saved {csv_path}\nSaved {figure_path}")


if __name__ == "__main__":
    main()
