"""Project 5: Compare Upper Confidence Bound and epsilon-greedy selection."""

import matplotlib.pyplot as plt
import numpy as np

from reinforcement_learning.policies import epsilon_greedy, upper_confidence_bound


class Environment:
    """A stationary k-armed bandit."""

    def __init__(self, k: int = 10) -> None:
        if k < 1:
            raise ValueError("`k` must be positive")
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal_action = 0

    def reset(self) -> None:
        """Sample the true mean reward for each action."""

        self.action_values = np.random.normal(0.0, 1.0, self.k)
        self.optimal_action = int(np.argmax(self.action_values))

    def step(self, action: int) -> float:
        """Return a noisy reward for an action."""

        if not 0 <= action < self.k:
            raise ValueError(f"Invalid action: {action}")
        return float(np.random.normal(self.action_values[action], 1.0))


class Agent:
    """An action-value agent with either UCB or epsilon-greedy selection."""

    def __init__(
        self,
        k: int = 10,
        strategy: str = "ucb",
        c: float = 2.0,
        epsilon: float = 0.1,
    ) -> None:
        if k < 1:
            raise ValueError("`k` must be positive")
        if strategy not in {"ucb", "epsilon-greedy"}:
            raise ValueError("`strategy` must be 'ucb' or 'epsilon-greedy'")

        self.q = np.zeros(k)
        self.counts = np.zeros(k, dtype=int)
        self.strategy = strategy
        self.c = c
        self.epsilon = epsilon

    def choose_action(self, step: int) -> int:
        """Select an action according to the configured strategy."""

        if self.strategy == "ucb":
            return upper_confidence_bound(self.q, self.counts, step, self.c)
        return epsilon_greedy(self.q, self.epsilon)

    def update(self, action: int, reward: float) -> None:
        """Update the selected action using an incremental sample average."""

        self.counts[action] += 1
        self.q[action] += (reward - self.q[action]) / self.counts[action]


def run_comparison(
    trials: int = 2_000,
    steps: int = 1_000,
    k_arms: int = 10,
    c: float = 2.0,
    epsilon: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return average reward curves for UCB and epsilon-greedy agents."""

    ucb_rewards = np.zeros((trials, steps))
    epsilon_rewards = np.zeros((trials, steps))

    for trial in range(trials):
        # Both agents face the same true action values in each trial.
        ucb_environment = Environment(k_arms)
        ucb_environment.reset()
        epsilon_environment = Environment(k_arms)
        epsilon_environment.action_values = ucb_environment.action_values.copy()
        epsilon_environment.optimal_action = ucb_environment.optimal_action

        ucb_agent = Agent(k_arms, strategy="ucb", c=c)
        epsilon_agent = Agent(k_arms, strategy="epsilon-greedy", epsilon=epsilon)

        for step in range(steps):
            time_step = step + 1

            ucb_action = ucb_agent.choose_action(time_step)
            ucb_reward = ucb_environment.step(ucb_action)
            ucb_rewards[trial, step] = ucb_reward
            ucb_agent.update(ucb_action, ucb_reward)

            epsilon_action = epsilon_agent.choose_action(time_step)
            epsilon_reward = epsilon_environment.step(epsilon_action)
            epsilon_rewards[trial, step] = epsilon_reward
            epsilon_agent.update(epsilon_action, epsilon_reward)

    return np.mean(ucb_rewards, axis=0), np.mean(epsilon_rewards, axis=0)


def plot_results(ucb_rewards: np.ndarray, epsilon_rewards: np.ndarray) -> None:
    """Plot average reward for both action-selection strategies."""

    steps = np.arange(1, len(ucb_rewards) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, ucb_rewards, color="blue", label="UCB (c = 2)")
    plt.plot(steps, epsilon_rewards, color="gray", label="epsilon-greedy (epsilon = 0.1)")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title("UCB vs. Epsilon-Greedy Average Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "docs/projects/assets/project_5_ucb_vs_epsilon_greedy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main() -> None:
    """Run the UCB and epsilon-greedy comparison."""

    np.random.seed(42)
    ucb_rewards, epsilon_rewards = run_comparison(c=2.0, epsilon=0.1)
    plot_results(ucb_rewards, epsilon_rewards)


if __name__ == "__main__":
    main()
