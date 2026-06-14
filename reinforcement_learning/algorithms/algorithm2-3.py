"""
Algorithm 2-3: On-policy MC algorithm
1. Input: a policy function pi(a | s, Q_pi(s,a))
2. Initialize Q(s, a) <- 0, for all s in S, a in A
3. loop
4.     Generate full episode trajectory following pi
5.     Initialize G <- 0
6.     loop for each step of episode t = T-1, T-2,...t_0
7.         G <- yG + r
8.         if (s, a) not in (s_0, a_0), (s_1, a_1),...,(s_t-1,a_t-1)
9.             Append G to Returns(s, a)
10.            Q(s,a) <- average(Returns(s,a))
"""

import numpy as np
from collections import defaultdict
from collections.abc import Callable
from statistics import mean

ACTIONS = ["left", "right"]


class CliffWalkTiny:
    """
    Tiny environment:

        cliff <- 1 <-> 2 -> 3

    Start: square 1
    Goal: square 3, reward +1
    Cliff: moving left from square 1, reward -1
    Max steps: 5
    """

    def __init__(self, max_steps: int = 5):
        if not isinstance(max_steps, int):
            raise TypeError(f"Expected max steps to be type int, got type {type(max_steps)}")
        if max_steps <= 1:
            raise ValueError("Max steps needs to be greater than 1")
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> int:
        self.state = 1
        self.steps = 0
        return self.state

    def step(self, action: str) -> tuple[int, int, bool]:
        """
        Returns:
            next_state, reward, done
        """
        self.steps += 1

        # From square 1
        if self.state == 1:
            if action == "left":
                # Fall off cliff
                reward = -1
                done = True
                next_state = -1
            elif action == "right":
                next_state = 2
                reward = 0
                done = False
            else:
                raise ValueError(f"Unknown action: {action}")

        # From square 2
        elif self.state == 2:
            if action == "left":
                next_state = 1
                reward = 0
                done = False
            elif action == "right":
                next_state = 3
                reward = 1
                done = True
            else:
                raise ValueError(f"Unknown action: {action}")

        else:
            raise ValueError(f"Cannot act from terminal state: {self.state}")

        # If max steps expires, terminate with no additional reward.
        if not done and self.steps >= self.max_steps:
            done = True

        self.state = next_state
        return next_state, reward, done


def epsilon_greedy_policy(state: int, Q: dict, epsilon: float = 0.1) -> str:
    """
    On-policy action selection.

    With probability epsilon, choose a random action.
    Otherwise choose the action with the highest Q-value.

    This policy depends on Q, so as Q improves, the behavior policy improves too.
    """
    if np.random.random() < epsilon:
        return np.random.choice(ACTIONS)

    q_values = [Q[(state, action)] for action in ACTIONS]
    max_q = max(q_values)

    # Random tie-breaking among best actions
    best_actions = [action for action in ACTIONS if Q[(state, action)] == max_q]

    return np.random.choice(best_actions)


def generate_episode(env: CliffWalkTiny, policy: Callable, Q: dict) -> list[tuple[int, str, int]]:
    """
    Generate one episode by following policy pi(a | s, Q).

    Returns a list of:
        [(state, action, reward), ...]
    """
    episode = []

    state = env.reset()
    done = False

    while not done:
        action = policy(state, Q)
        next_state, reward, done = env.step(action)

        episode.append((state, action, reward))

        state = next_state

    return episode


def on_policy_mc(env: CliffWalkTiny, policy: Callable, num_episodes: int = 10_000, gamma: float = 1.0) -> tuple[dict, dict]:
    """
    On-policy Monte Carlo algorithm.

    Q(s, a) starts at 0.
    Returns(s, a) starts as an empty list.
    For each episode:
        - Follow pi(a | s, Q)
        - Work backward and compute return G
        - Append G to Returns(s, a)
        - Set Q(s, a) = average Returns(s, a)
    """

    # Q(s, a) initialized to 0
    Q: dict[tuple[int, str], float] = defaultdict(float)

    # returns(s, a) <- []
    returns = defaultdict(list)

    for episode_number in range(num_episodes):
        episode = generate_episode(env, policy, Q)

        G = 0.0

        # Loop backward through episode
        for state, action, reward in reversed(episode):
            G = reward + gamma * G

            returns[(state, action)].append(G)
            Q[(state, action)] = mean(returns[(state, action)])

    return Q, returns


def greedy_action(state: int, Q: dict) -> str:
    """
    Return the best action under the learned Q-values.
    """
    q_values = {action: Q[(state, action)] for action in ACTIONS}
    max_q = max(q_values.values())

    best_actions = [action for action, value in q_values.items() if value == max_q]

    return np.random.choice(best_actions)


def run_greedy_episode(env: CliffWalkTiny, Q: dict) -> tuple[list[tuple[int, str, int, int]], int]:
    """
    Run one episode using the greedy policy learned from Q.
    """
    state = env.reset()
    done = False
    trajectory = []
    total_reward = 0

    while not done:
        action = greedy_action(state, Q)
        next_state, reward, done = env.step(action)

        trajectory.append((state, action, reward, next_state))
        total_reward += reward

        state = next_state

    return trajectory, total_reward


def evaluate_fixed_actions(actions: list[str]) -> tuple[list[tuple[int, str, int, int]], int]:
    """
    Useful for checking a hand-written action sequence, for example:

        right, left, right, left, right

    This should end after 5 moves with 0 total reward.
    """
    env = CliffWalkTiny(max_steps=5)
    state = env.reset()
    total_reward = 0
    trajectory = []

    for action in actions:
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward, next_state))
        total_reward += reward
        state = next_state

        if done:
            break

    return trajectory, total_reward


if __name__ == "__main__":
    np.random.seed(7)

    env = CliffWalkTiny(max_steps=5)

    def pi(state, Q):
        return epsilon_greedy_policy(state, Q, epsilon=0.1)

    Q, returns = on_policy_mc(env=env, policy=pi, num_episodes=20_000, gamma=1.0)

    print("Learned Q-values:")
    for state in [1, 2]:
        for action in ACTIONS:
            print(f"Q({state}, {action:>5}) = {Q[(state, action)]: .3f}")

    print("\nGreedy learned policy:")
    for state in [1, 2]:
        print(f"At square {state}, choose: {greedy_action(state, Q)}")

    print("\nRun one greedy episode after learning:")
    trajectory, total_reward = run_greedy_episode(env, Q)
    for transition in trajectory:
        state, action, reward, next_state = transition
        print(f"state={state}, action={action:>5}, " f"reward={reward}, next_state={next_state}")
    print(f"Total reward: {total_reward}")

    print("\nCheck fixed sequence: right, left, right, left, right")
    fixed_actions = ["right", "left", "right", "left", "right"]
    trajectory, total_reward = evaluate_fixed_actions(fixed_actions)

    for transition in trajectory:
        state, action, reward, next_state = transition
        print(f"state={state}, action={action:>5}, " f"reward={reward}, next_state={next_state}")

    print(f"Total reward: {total_reward}")
