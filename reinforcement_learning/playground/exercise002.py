"""Exercise 2: 10 armed bandit with a greedy method

In this exercise, a 10 armed bandit will be established using a Gaussian distribution. This is similar
to the 10-armed bandit in Chapter 2 of Sutton's Reinforcement Learning. The goal here is to establish a
agent that chooses to exploit actions by choosing the action with the greatest potential reward.

Methods:
1. Declare the Environment with ten reward means randomly picked from a Gaussian distribution N~(0,1). When
returning a reward to the agent, the reward will again be given using a Gaussian distribution with variance 1
and mean equal to the variance randomly assigned.
2. Establish the Agent. The agent is to choose actions greedily based on average rewards. In scenarios where
actions have a tie, the agent is to randomly choose between actions.
3. Perform an iteration of 1000 steps for the agent to learn the most optimal action.
"""

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
        self.rewards = np.random.normal(0, 1, 10)

        # Determine which reward is optimal
        self.optimal_action = int(np.argmax(self.rewards))

    def step(self, action: int) -> tuple[float, bool]:
        """Returns a sampled reward and whether the action was optimal"""

        if action < 0 or action >= self.k:
            raise ValueError(f"Invalid action: {action}")

        reward = np.random.normal(self.rewards[action], 1)
        optimal = action == self.optimal_action

        return float(reward), optimal


class Agent:
    def __init__(self, epsilon: float = 0.0, k: int = 10):

        if (epsilon < 0) or (epsilon) > 1:
            raise ValueError(f"Invalid epsilon value: {epsilon}")

        self.epsilon = epsilon
        self.k = k
        self.q = np.zeros(k)
        self.n = np.zeros(k)

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

        # Increment the counter
        self.n[action] += 1

        # Calculate the new average reward
        N = self.n[action]
        Q = self.q[action]
        self.q[action] = Q + 1 / N * (reward - Q)


def run_experiments(
    epsilons: list[float],
    trials: int = 2000,
    steps: int = 1000,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Run repeated k-armed bandit experiments for multiple epsilon values."""

    results = {}

    for epsilon in epsilons:
        print(f"\nε = {epsilon}")

        average_rewards, percent_optimal = run_experiment(
            trials=trials,
            steps=steps,
            epsilon=epsilon,
        )

        results[epsilon] = (
            average_rewards,
            percent_optimal,
        )

    return results


def run_experiment(
    trials: int = 2000,
    steps: int = 1000,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run repeated k-armed bandit trials."""

    rewards = np.zeros((trials, steps))
    optimal_actions = np.zeros((trials, steps))

    for trial in range(trials):

        print(f"\r{loading_bar(trial + 1, trials)}", end="")

        env = Environment()
        agent = Agent(epsilon=epsilon)

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


def plot_results(
    results: dict[float, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Save comparison plots for multiple epsilon values."""

    first_result = next(iter(results.values()))
    steps = np.arange(1, len(first_result[0]) + 1)

    plt.figure(figsize=(10, 6))

    for epsilon, (average_rewards, _) in results.items():
        plt.plot(
            steps,
            average_rewards,
            label=f"ε = {epsilon}",
        )

    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title("Average Reward over 2,000 Trials")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(
        "docs/exercises/assets/exercise_2_average_reward.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 6))

    for epsilon, (_, percent_optimal) in results.items():
        plt.plot(
            steps,
            percent_optimal,
            label=f"ε = {epsilon}",
        )

    plt.xlabel("Step")
    plt.ylabel("Optimal Action (%)")
    plt.title("Optimal Action Selection over 2,000 Trials")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(
        "docs/exercises/assets/exercise_2_optimal_action_percentage.png",
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
    results = run_experiments(
        epsilons=[0.0, 0.01, 0.1],
        trials=2000,
        steps=1000,
    )

    plot_results(results)
