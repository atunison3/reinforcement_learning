"""Project 7: Associative search using Sutton and Barto's Exercise 2.10.

Run with ``python -m reinforcement_learning.projects.project007``.
The experiment saves two figures and prints empirical and analytic rewards.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from reinforcement_learning.contextual_bandits import ContextualBanditAgent, optimal_expected_rewards

# Rows are cases A and B; columns are textbook actions 1 and 2.
ACTION_VALUES: NDArray[np.float64] = np.array([[10.0, 20.0], [90.0, 80.0]])
CONTEXT_PROBABILITIES: NDArray[np.float64] = np.array([0.5, 0.5])
EPSILON = 0.1


@dataclass
class LearningCurves:
    """Per-step measurements averaged over independent learning trials."""

    average_reward: NDArray[np.float64]
    optimal_action_percentage: NDArray[np.float64]


def run_experiments(
    trials: int = 1_000,
    steps: int = 1_000,
    epsilon: float = EPSILON,
    reward_std: float = 1.0,
    seed: int = 42,
) -> dict[str, LearningCurves]:
    """Compare learners with and without the current case as a cue.

    The context-aware learner sees the case identity, never the true values.
    The other learner always receives context index zero, even in case B.
    Both observe only their own chosen action's reward. A shared sequence of
    cases and potential rewards pairs the comparison within each trial.

    Reward noise is a simulation choice; Exercise 2.10 specifies only means.
    This function seeds NumPy's global RNG used by ``epsilon_greedy``. A
    separate generator makes the environment independent of policy RNG use.
    """

    if trials < 1 or steps < 1:
        raise ValueError("`trials` and `steps` must be positive")
    if not 0 <= epsilon <= 1:
        raise ValueError("`epsilon` must be between 0 and 1")
    if not np.isfinite(reward_std) or reward_std < 0:
        raise ValueError("`reward_std` must be finite and nonnegative")

    np.random.seed(seed)
    environment_rng = np.random.default_rng(seed)
    results = {name: LearningCurves(np.zeros(steps), np.zeros(steps)) for name in ("Without context", "With context")}
    optimal_actions = np.argmax(ACTION_VALUES, axis=1)

    for _ in range(trials):
        contexts = environment_rng.choice(2, size=steps, p=CONTEXT_PROBABILITIES)
        potential_rewards = ACTION_VALUES[contexts] + environment_rng.normal(0.0, reward_std, size=(steps, 2))
        agents = {
            "Without context": ContextualBanditAgent(n_contexts=1, n_actions=2, epsilon=epsilon),
            "With context": ContextualBanditAgent(n_contexts=2, n_actions=2, epsilon=epsilon),
        }

        for step, raw_context in enumerate(contexts):
            context = int(raw_context)
            for name, agent in agents.items():
                observed_context = context if name == "With context" else 0
                action = agent.choose_action(observed_context)
                reward = float(potential_rewards[step, action])
                agent.update(observed_context, action, reward)
                results[name].average_reward[step] += reward
                # This evaluation uses the hidden case even for the unaware agent.
                results[name].optimal_action_percentage[step] += 100.0 * (action == optimal_actions[context])

    for curves in results.values():
        curves.average_reward /= trials
        curves.optimal_action_percentage /= trials
    return results


def plot_results(
    results: dict[str, LearningCurves],
    output_directory: Path | None = None,
    epsilon: float = EPSILON,
    block_size: int = 20,
) -> None:
    """Save reward and contextual-optimal-action plots, averaging step blocks."""

    if block_size < 1:
        raise ValueError("`block_size` must be positive")
    if output_directory is None:
        output_directory = Path(__file__).resolve().parents[2] / "docs" / "projects" / "assets"
    output_directory.mkdir(parents=True, exist_ok=True)

    without_context, with_context = optimal_expected_rewards(ACTION_VALUES, CONTEXT_PROBABILITIES)
    random_reward = float(CONTEXT_PROBABILITIES @ np.mean(ACTION_VALUES, axis=1))
    contextual_epsilon_reward = (1 - epsilon) * with_context + epsilon * random_reward
    colors = {"Without context": "tab:gray", "With context": "tab:blue"}
    steps = len(next(iter(results.values())).average_reward)
    starts = np.arange(0, steps, block_size)
    ends = np.minimum(starts + block_size, steps)

    for metric, ylabel in (("average_reward", "Average reward"), ("optimal_action_percentage", "Optimal action (%)")):
        figure, axis = plt.subplots(figsize=(10, 6))
        for name, curves in results.items():
            values = curves.average_reward if metric == "average_reward" else curves.optimal_action_percentage
            blocked_values = [float(np.mean(values[start:end])) for start, end in zip(starts, ends)]
            axis.plot(ends, blocked_values, color=colors[name], label=f"{name} (epsilon = {epsilon:g})")

        if metric == "average_reward":
            axis.axhline(with_context, color="black", linestyle="--", label=f"Context-aware optimum: {with_context:g}")
            axis.axhline(
                without_context, color="tab:gray", linestyle="--", label=f"No-context optimum: {without_context:g}"
            )
            axis.axhline(
                contextual_epsilon_reward,
                color="tab:blue",
                linestyle=":",
                label=f"Learned epsilon-greedy target: {contextual_epsilon_reward:g}",
            )
        else:
            axis.axhline(100, color="black", linestyle="--", label="Context-aware optimum: 100%")
            axis.axhline(50, color="tab:gray", linestyle="--", label="No-context expectation: 50%")
            axis.axhline(
                100 * (1 - epsilon / 2), color="tab:blue", linestyle=":", label="Learned epsilon-greedy target"
            )
            axis.set_ylim(0, 105)

        axis.set_xlabel(f"Step (non-overlapping blocks of {block_size}, plotted at block end)")
        axis.set_ylabel(ylabel)
        axis.set_title("Associative Search: Observing the Current Case")
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_directory / f"project_7_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    """Run the reproducible experiment and answer the numeric part of 2.10."""

    without_context, with_context = optimal_expected_rewards(ACTION_VALUES, CONTEXT_PROBABILITIES)
    print(f"Exercise 2.10: optimal expected reward without context = {without_context:g}")
    print(f"Exercise 2.10: optimal expected reward with context = {with_context:g}")
    results = run_experiments()
    plot_results(results)
    print("Empirical performance over the final 200 steps (epsilon = 0.1):")
    for name, curves in results.items():
        print(
            f"  {name}: reward = {np.mean(curves.average_reward[-200:]):.3f}, "
            f"optimal action = {np.mean(curves.optimal_action_percentage[-200:]):.2f}%"
        )


if __name__ == "__main__":
    main()
