"""Project 9: What does a reward baseline do in a gradient bandit?

Compare five baselines on stationary 10-armed bandits, then repeat with all
rewards shifted by +4. Also illustrate exact gradient variance and a paired
reward-shift invariance experiment. Run as a module to save plots and a CSV.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from reinforcement_learning.gradient_bandits import (
    BaselineKind,
    GradientBanditBatch,
    gradient_moments,
    minimum_variance_baseline,
)


@dataclass(frozen=True)
class BaselineConfiguration:
    name: str
    kind: BaselineKind
    initial_value: float
    color: str


CONFIGURATIONS = (
    BaselineConfiguration("Fixed 0", "fixed", 0.0, "tab:gray"),
    BaselineConfiguration("Fixed -4", "fixed", -4.0, "tab:purple"),
    BaselineConfiguration("Fixed +4", "fixed", 4.0, "tab:orange"),
    BaselineConfiguration("Running average", "average", 0.0, "tab:blue"),
    BaselineConfiguration("Exponential average", "ema", 0.0, "tab:green"),
)
REWARD_OFFSETS = (0.0, 4.0)


@dataclass
class LearningCurves:
    average_reward: NDArray[np.float64]
    optimal_action_percentage: NDArray[np.float64]
    baseline_before_update: NDArray[np.float64]


@dataclass
class ExperimentResult:
    curves: dict[tuple[float, str], LearningCurves]
    optimal_expected_rewards: dict[float, float]
    trials: int
    steps: int
    alpha: float
    beta: float
    seed: int


def run_experiments(
    trials: int = 2_000, steps: int = 1_000, alpha: float = 0.1, beta: float = 0.1, seed: int = 42
) -> ExperimentResult:
    """Compare baselines on paired stationary tasks with reward offsets 0 and 4.

    Each trial samples ten true means from N(0, 1), then holds them fixed.
    Reward noise has standard deviation 1. Agents share true means and
    potential reward samples, but see only their chosen action's reward.
    Baselines are separate for every trial; no rewards are pooled across
    independent learners. The shifted condition reuses the same random draws.
    """

    if trials < 1 or steps < 1:
        raise ValueError("`trials` and `steps` must be positive")
    # Validate learning parameters before sampling or allocating the experiment.
    GradientBanditBatch(1, alpha=alpha, beta=beta)
    true_values = np.random.default_rng(seed).normal(0.0, 1.0, size=(trials, 10))
    optimal_actions = np.argmax(true_values, axis=1)
    curves: dict[tuple[float, str], LearningCurves] = {}
    optimal_expected_rewards: dict[float, float] = {}
    for offset in REWARD_OFFSETS:
        noise_rng = np.random.default_rng(seed + 1)
        policy_rngs = [np.random.default_rng(seed + 2 + index) for index in range(len(CONFIGURATIONS))]
        agents = [
            GradientBanditBatch(trials, 10, alpha, config.kind, config.initial_value, beta) for config in CONFIGURATIONS
        ]
        for config in CONFIGURATIONS:
            curves[offset, config.name] = LearningCurves(np.zeros(steps), np.zeros(steps), np.zeros(steps))
        optimal_expected_rewards[offset] = float(np.mean(np.max(true_values, axis=1))) + offset
        for step in range(steps):
            potential_rewards = true_values + noise_rng.normal(0.0, 1.0, size=(trials, 10)) + offset
            for config, agent, rng in zip(CONFIGURATIONS, agents, policy_rngs):
                actions = agent.choose_actions(rng)
                rewards = potential_rewards[agent.rows, actions]
                curve = curves[offset, config.name]
                curve.average_reward[step] = np.mean(rewards)
                curve.optimal_action_percentage[step] = 100 * np.mean(actions == optimal_actions)
                curve.baseline_before_update[step] = np.mean(agent.baseline)
                agent.update(actions, rewards)
    return ExperimentResult(curves, optimal_expected_rewards, trials, steps, alpha, beta, seed)


def reward_shift_demo(shift: float = 4.0, steps: int = 100, seed: int = 42) -> float:
    """Return the largest policy difference when rewards AND baselines shift.

    Coupled draws make this a trajectory-level check, not just a comparison
    of averages. The shifted learner starts its running baseline at shift,
    so even the first update has exactly the same advantage in real arithmetic.
    """

    if steps < 1 or not np.isfinite(shift):
        raise ValueError("Use positive steps and a finite reward shift")
    agents = (GradientBanditBatch(32), GradientBanditBatch(32, initial_baseline=shift))
    policy_rngs = (np.random.default_rng(seed), np.random.default_rng(seed))
    reward_rng = np.random.default_rng(seed + 1)
    values = np.linspace(-1.0, 1.0, 10)
    largest_difference = 0.0
    for _ in range(steps):
        potential_rewards = values + reward_rng.normal(size=(32, 10))
        for agent, rng, offset in zip(agents, policy_rngs, (0.0, shift)):
            actions = agent.choose_actions(rng)
            agent.update(actions, potential_rewards[agent.rows, actions] + offset)
        difference = float(np.max(np.abs(agents[0].probabilities() - agents[1].probabilities())))
        largest_difference = max(largest_difference, difference)
    return largest_difference


def save_summary(result: ExperimentResult, path: Path) -> None:
    """Export full-horizon and final-window performance with reproducibility settings."""

    window = min(200, result.steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            (
                "reward_offset",
                "baseline",
                "mean_reward",
                "final_window",
                "final_reward",
                "final_optimal_percent",
                "trials",
                "steps",
                "alpha",
                "beta",
                "seed",
            )
        )
        for (offset, name), curve in result.curves.items():
            writer.writerow(
                (
                    offset,
                    name,
                    float(np.mean(curve.average_reward)),
                    window,
                    float(np.mean(curve.average_reward[-window:])),
                    float(np.mean(curve.optimal_action_percentage[-window:])),
                    result.trials,
                    result.steps,
                    result.alpha,
                    result.beta,
                    result.seed,
                )
            )


def plot_learning(result: ExperimentResult, directory: Path) -> None:
    """Save reward/optimal-action curves and the baselines used before updating."""

    directory.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, result.steps + 1)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    baseline_figure, baseline_axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for column, offset in enumerate(REWARD_OFFSETS):
        title = f"True means drawn from N({offset:g}, 1), then held fixed"
        axes[0, column].set_title(title)
        baseline_axes[column].set_title(title)
        for config in CONFIGURATIONS:
            curve = result.curves[offset, config.name]
            axes[0, column].plot(steps, curve.average_reward, color=config.color, label=config.name, linewidth=1)
            axes[1, column].plot(steps, curve.optimal_action_percentage, color=config.color, linewidth=1)
            baseline_axes[column].plot(steps, curve.baseline_before_update, color=config.color, label=config.name)
        axes[0, column].axhline(
            result.optimal_expected_rewards[offset], color="black", linestyle="--", label="Mean optimal action value"
        )
        axes[1, column].set_ylim(0, 100)
        axes[1, column].set_xlabel("Step")
        baseline_axes[column].set_xlabel("Step")
        baseline_axes[column].grid(alpha=0.3)
        for row in (0, 1):
            axes[row, column].grid(alpha=0.3)
    axes[0, 0].set_ylabel("Average reward")
    axes[1, 0].set_ylabel("Optimal action (%)")
    baseline_axes[0].set_ylabel("Mean baseline before preference update")
    axes[0, 0].legend(fontsize="small")
    baseline_axes[0].legend(fontsize="small")
    figure.suptitle(f"Gradient Bandit Baselines: alpha = {result.alpha:g}, EMA beta = {result.beta:g}")
    baseline_figure.suptitle("A stationary bandit can still have a changing policy reward mean")
    figure.tight_layout()
    baseline_figure.tight_layout()
    figure.savefig(directory / "project_9_learning.png", dpi=180, bbox_inches="tight")
    baseline_figure.savefig(directory / "project_9_baselines.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    plt.close(baseline_figure)


def plot_variance_diagnostic(path: Path) -> None:
    """Exact fixed-policy variance: average reward is not always the best baseline."""

    values = np.linspace(-1.0, 1.0, 10)
    policies = (np.full(10, 0.1), np.array([0.01] * 9 + [0.91]))
    titles = ("Uniform policy", "91% probability on the best arm")
    baselines = np.linspace(-4.0, 4.0, 161)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, policy, title in zip(axes, policies, titles):
        variances = [gradient_moments(policy, values, float(baseline))[1] for baseline in baselines]
        optimal = minimum_variance_baseline(policy, values)
        mean_reward = float(policy @ values)
        axis.plot(baselines, variances, color="tab:blue", label="Exact gradient variance")
        axis.axvline(mean_reward, color="black", linestyle="--", label=f"Policy mean reward: {mean_reward:.3f}")
        axis.axvline(optimal, color="tab:green", linestyle=":", label=f"Minimum-variance baseline: {optimal:.3f}")
        axis.set_title(title)
        axis.set_xlabel("Fixed baseline b")
        axis.set_ylabel("Total gradient variance (before multiplying by alpha)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize="small")
    figure.suptitle("Same expected gradient for every baseline; different noise")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2_000)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "docs" / "projects" / "assets"
    )
    args = parser.parse_args()
    result = run_experiments(args.trials, args.steps, args.alpha, args.beta, args.seed)
    save_summary(result, args.output_dir / "project_9_summary.csv")
    plot_learning(result, args.output_dir)
    plot_variance_diagnostic(args.output_dir / "project_9_gradient_variance.png")
    window = min(200, result.steps)
    for (offset, name), curve in result.curves.items():
        print(
            f"offset={offset:g}, {name}: final {window} steps reward={np.mean(curve.average_reward[-window:]):.3f}, "
            f"optimal={np.mean(curve.optimal_action_percentage[-window:]):.2f}%"
        )
    print(f"Reward-shift demo: largest policy difference = {reward_shift_demo(seed=args.seed):.3e}")
    print(f"Saved three plots and summary CSV to {args.output_dir}")


if __name__ == "__main__":
    main()
