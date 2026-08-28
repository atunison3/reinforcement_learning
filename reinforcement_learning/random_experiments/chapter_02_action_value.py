import numpy as np


class Agent:
    def __init__(self, epsilon: float = 0.5):
        self.epsilon = epsilon
        self.r = [[0.0, 0.0] for _ in range(4)]
        self.n = [[0, 0] for _ in range(4)]

    def choose_action(self, state: int) -> int:
        """Decides which action to take"""

        if np.random.choice([0, 1], p=[1 - self.epsilon, self.epsilon]):
            # Agent chooses to explore
            return int(np.random.choice([0, 1]))
        else:
            s_ = self.r[state]
            if s_[0] == s_[1]:
                # Rewards are equal for each action - need to randomly choose
                return int(np.random.choice([0, 1]))
            else:
                # Agent chooses the action that provides the best reward
                return int(np.argmax(s_))

    def update(self, state: int, reward: float, action: int) -> None:
        """Updates a state's action reward"""

        # Update N(a)
        self.n[state][action] += 1

        # Online value function
        r_avg = self.r[state][action]  # With averaging
        n_a = self.n[state][action]

        self.r[state][action] = r_avg + 1 / n_a * (reward - r_avg)


class Environment:
    def __init__(self):
        self.state = 0
        self.steps = 0

    def get_reward(self, action: int) -> int:
        """Returns a reward"""

        self.steps += 1

        if action == 0:
            action = -1

        self.state += action

        if self.state == -1:
            return -1
        elif self.state == 4:
            return 5
        else:
            return 0

    def reset_env(self):
        self.state = 0
        self.steps = 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("npseed", type=int)
    args = parser.parse_args()

    agent = Agent(epsilon=1)
    env = Environment()

    np.random.seed(args.npseed)
    for trial in range(2000):

        # Declare actions and states
        actions = []

        while env.state >= 0 and env.state < 4 and env.steps < 10:

            # Get the current state
            current_state = env.state

            # Let the agent choose an action
            a = agent.choose_action(current_state)

            # Environment determines the next state and reward
            reward = env.get_reward(a)

            # Update rewards history
            actions.append((current_state, a))

        # Random selection he performed on page 46
        G = reward
        actions.reverse()
        for i, x in enumerate(actions):
            state, action = x
            agent.update(state, G, action)

        # # Averaging
        # G = reward / len(actions)
        # actions.reverse()
        # for i, x in enumerate(actions):
        #     state, action = x
        #     agent.update(state, G, action)

        # # With Discounting
        # gamma = 0.9
        # G = reward
        # actions.reverse()
        # for i, x in enumerate(actions):
        #     state, action = x
        #     agent.update(state, G, action)
        #     G *= gamma

        env.reset_env()
        # agent.epsilon *= 0.9999

    print(f"Averaging: {agent.r}")
