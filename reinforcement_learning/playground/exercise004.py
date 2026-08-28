"""Exercise 4: Tracking a Nonstationary Problem with optimistic initial values"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class Environment:
    def __init__(self, k: int = 10):
        self.k = k
        self.rewards = np.zeros(k)
        self.optimal_action: int = 0

    def reset(self) -> None:
        """Returns the base state"""

        # Assign the reward means using Normal distrubtion
        self.rewards = np.random.normal(0, 1, self.k)

        # Determine which reward is optimal
        self.optimal_action = int(np.argmax(self.rewards))

    def step(self, action: int) -> tuple[float, bool]:
        """Returns a sampled reward and whether the action was optimal"""

        if action < 0 or action >= self.k:
            raise ValueError(f"Invalid action: {action}")

        reward = np.random.normal(self.rewards[action], 1)
        optimal = action == self.optimal_action

        # Update the moving targets
        self.rewards += np.random.normal(0, 0.01, self.k)

        return float(reward), optimal


class Agent:
    def __init__(self, epsilon: float = 0.1, k: int = 10, step_size: float | None = 0.1, initial_values: float = 0.0):

        if (epsilon < 0) or (epsilon) > 1:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if step_size and ((step_size <= 0) or (step_size > 1)):
            raise ValueError(f"Invalid step size vlaue: {step_size}")

        self.alpha: float | None = None

        if step_size:
            self.alpha = step_size
        else:
            self.n = np.zeros(k)
        self.epsilon = epsilon
        self.k = k
        self.q = np.zeros(k) + initial_values

    def choose_action(self) -> int:
        """Agent chooses action"""

        # Randomly chooses action if exploring
        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.k))

        # Exploit
        max_value = np.max(self.q)
        best_actions = np.flatnonzero(self.q == max_value)

        return int(np.random.choice(best_actions))

    def update(self, action: int, reward: float) -> None:
        """Updates the agent's policy"""

        # Calculate the new average reward
        Q = self.q[action]

        if self.alpha:
            self.q[action] = Q + self.alpha * (reward - Q)
        else:
            self.n[action] += 1
            N = self.n[action]
            self.q[action] = Q + 1 / N * (reward - Q)


def run_experiments(
    alphas: list[float | None], initial_values: list[float], trials: int = 2000, steps: int = 10000, k_arms: int = 10
) -> Any:
    """Run repeated k-armed bandit experiments for multiple alpha values."""

    results = {}

    for alpha in alphas:
        for initial_value in initial_values:
            print(f"\nα = {alpha}")

            average_rewards, percent_optimal = run_experiment(
                trials=trials, steps=steps, alpha=alpha, initial_values=initial_value, k_arms=k_arms
            )

            results[alpha, initial_value] = (
                average_rewards,
                percent_optimal,
            )

    return results


def run_experiment(
    trials: int = 2000, steps: int = 10000, alpha: float | None = 0.1, initial_values: float = 0.0, k_arms: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Run repeated k-armed bandit trials."""

    rewards = np.zeros((trials, steps))
    optimal_actions = np.zeros((trials, steps))

    for trial in range(trials):

        print(f"\r{loading_bar(trial + 1, trials)}", end="")

        env = Environment(k=k_arms)
        agent = Agent(epsilon=0.1, step_size=alpha, initial_values=initial_values, k=k_arms)

        env.reset()

        for step in range(steps):
            action = agent.choose_action()

            reward, optimal = env.step(action)

            rewards[trial, step] = reward
            optimal_actions[trial, step] = optimal

            agent.update(action, reward)

    average_rewards = np.mean(rewards, axis=0)
    percent_optimal = np.mean(optimal_actions, axis=0) * 100

    return average_rewards, percent_optimal


def plot_results(results: dict[float | None, tuple[np.ndarray, np.ndarray]], k_arms: int = 10) -> None:
    """Save comparison plots for multiple alpha values."""

    first_result = next(iter(results.values()))
    steps = np.arange(1, len(first_result[0]) + 1)

    plt.figure(figsize=(10, 6))

    for alpha, (average_rewards, _) in results.items():
        label = "sample average" if alpha is None else f"α = {alpha}"
        plt.plot(
            steps,
            average_rewards,
            label=label,
        )

    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title("Average Reward over 2,000 Trials")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(
        f"docs/exercises/assets/exercise_4_average_reward_k{k_arms}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 6))

    for alpha, (_, percent_optimal) in results.items():
        label = "sample average" if alpha is None else f"α = {alpha}"
        plt.plot(
            steps,
            percent_optimal,
            label=label,
        )

    plt.xlabel("Step")
    plt.ylabel("Optimal Action (%)")
    plt.title("Optimal Action Selection over 2,000 Trials")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(
        f"docs/exercises/assets/exercise_4_optimal_action_percentage_K{k_arms}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def loading_bar(current: int, total: int, width: int = 20) -> str:
    """Return a Unicode loading bar."""

    progress = current / total
    filled = int(progress * width)
    empty = width - filled

    bar = "█" * filled + "░" * empty
    percent = progress * 100

    return f"[{bar}] {percent:6.2f}%"


if __name__ == "__main__":
    trials = 2000
    steps = 1000

    print("Running 10-armed experiment")
    results = run_experiments(alphas=[None, 0.1], trials=trials, steps=steps, initial_values=[0, 5], k_arms=10)

    plot_results(results, k_arms=10)

    print("Running 20-armed experiment")
    results = run_experiments(alphas=[None, 0.1], trials=trials, steps=steps, initial_values=[0, 5], k_arms=20)

    plot_results(results, k_arms=20)
