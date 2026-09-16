"""Exercise 6: Compare gradient bandit algorithms."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class Environment:
    """A stationary k-armed bandit with Gaussian rewards."""

    def __init__(self, k: int = 10) -> None:
        if k < 1:
            raise ValueError("`k` must be positive")
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal_action = 0

    def reset(self) -> None:
        """Sample the true mean reward of every action."""

        self.action_values = np.random.normal(0.0, 1.0, self.k)
        self.optimal_action = int(np.argmax(self.action_values))

    def step(self, action: int) -> tuple[float, bool]:
        """Return a noisy reward and whether the action is optimal."""

        if not 0 <= action < self.k:
            raise ValueError(f"Invalid action: {action}")
        reward = float(np.random.normal(self.action_values[action], 1.0))
        return reward, action == self.optimal_action


class GradientBanditAgent:
    """A gradient bandit agent with a fixed reward baseline."""

    def __init__(self, k: int = 10, alpha: float = 0.1, baseline: float = 0.0) -> None:
        if k < 1:
            raise ValueError("`k` must be positive")
        if alpha <= 0:
            raise ValueError("`alpha` must be positive")

        self.preferences = np.zeros(k)
        self.alpha = alpha
        self.baseline = baseline

    def probabilities(self) -> NDArray[np.float64]:
        """Return the softmax policy induced by the preferences."""

        shifted_preferences = self.preferences - np.max(self.preferences)
        weights: NDArray[np.float64] = np.exp(shifted_preferences)
        return weights / np.sum(weights)

    def choose_action(self) -> int:
        """Sample an action from the current softmax policy."""

        return int(np.random.choice(len(self.preferences), p=self.probabilities()))

    def update(self, action: int, reward: float) -> None:
        """Apply the gradient bandit preference update."""

        probabilities = self.probabilities()
        reward_error = reward - self.baseline

        self.preferences -= self.alpha * reward_error * probabilities
        self.preferences[action] += self.alpha * reward_error


def run_experiments(
    alphas: tuple[float, ...] = (0.1, 0.4),
    baselines: tuple[float, ...] = (0.0, 4.0),
    trials: int = 2_000,
    steps: int = 1_000,
    k_arms: int = 10,
) -> dict[tuple[float, float], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Run all step-size and baseline combinations.

    Returns a mapping from ``(alpha, baseline)`` to average-reward and
    optimal-action percentage curves.
    """

    rewards = {(alpha, baseline): np.zeros((trials, steps)) for alpha in alphas for baseline in baselines}
    optimal_actions = {(alpha, baseline): np.zeros((trials, steps)) for alpha in alphas for baseline in baselines}

    for trial in range(trials):
        environment = Environment(k_arms)
        environment.reset()

        agents = {
            (alpha, baseline): GradientBanditAgent(k_arms, alpha, baseline)
            for alpha in alphas
            for baseline in baselines
        }

        for step in range(steps):
            for configuration, agent in agents.items():
                action = agent.choose_action()
                reward, optimal = environment.step(action)
                rewards[configuration][trial, step] = reward
                optimal_actions[configuration][trial, step] = optimal
                agent.update(action, reward)

    return {
        configuration: (
            np.mean(configuration_rewards, axis=0),
            100 * np.mean(optimal_actions[configuration], axis=0),
        )
        for configuration, configuration_rewards in rewards.items()
    }


def plot_results(
    results: dict[tuple[float, float], tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> None:
    """Plot optimal-action percentage for all four gradient bandit algorithms."""

    first_curve = next(iter(results.values()))[1]
    steps = np.arange(1, len(first_curve) + 1)
    # Baseline 4 is blue; baseline 0 is orange.
    colors = {0.0: "tab:orange", 4.0: "tab:blue"}
    # Alpha 0.1 is full color; alpha 0.4 is half opacity.
    opacities = {0.1: 1.0, 0.4: 0.5}

    plt.figure(figsize=(10, 6))
    for (alpha, baseline), (_, percent_optimal) in results.items():
        plt.plot(
            steps,
            percent_optimal,
            color=colors[baseline],
            alpha=opacities[alpha],
            label=f"alpha = {alpha}, baseline = {baseline:g}",
        )

    plt.xlabel("Step")
    plt.ylabel("Optimal Action (%)")
    plt.title("Gradient Bandit Optimal Action Selection")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "docs/exercises/assets/exercise_6_gradient_bandit.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main() -> None:
    """Run and display the gradient bandit comparison."""

    np.random.seed(42)
    results = run_experiments(alphas=(0.1, 0.4), baselines=(0.0, 4.0))
    plot_results(results)


if __name__ == "__main__":
    main()
