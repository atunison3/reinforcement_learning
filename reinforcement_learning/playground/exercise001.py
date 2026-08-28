"""Exercise 1 in my playground to help understand reinforcement learning.

This module explores a 2 armed bandit. An agent randomly chooses between going left and right.
To its left (0) is a cliff, it falls off and -1 points. To its right (1) is not a cliff, +1.

This code implements a uniformly random policy of determining actions.
"""

from numpy import random


class Environment:
    def reset(self) -> int:
        """Returns the base state"""

        return 1

    def step(self, action: int) -> tuple[int, int, bool]:
        """Returns a reward for the action at a given state"""

        if action == 0:
            # Action is to move left
            return 1, -1, True

        if action == 1:
            # Action is to move right
            return 1, 1, True

        raise ValueError(f"Invalid action: {action}")


class Agent:
    def __init__(self) -> None:
        self.q = {1: [0.0, 0.0]}
        self.n = {1: [0.0, 0.0]}

    def choose_action(self, state: int) -> int:
        """Agent chooses action"""

        if state != 1:
            raise ValueError("Only one state in this game")

        return int(random.choice([0, 1]))

    def update(self, state: int, action: int, reward: int, next_state: int) -> None:
        """Updates the agent's policy"""

        # Increment the counter
        self.n[state][action] += 1

        # Calculate the new average reward
        N = self.n[state][action]
        Q = self.q[state][action]
        self.q[state][action] = Q + 1 / N * (reward - Q)


if __name__ == "__main__":

    env = Environment()
    agent = Agent()

    for _ in range(1000):
        terminated = False
        state = env.reset()

        while not terminated:
            action = agent.choose_action(state)

            next_state, reward, terminated = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state

    print(agent.q)
