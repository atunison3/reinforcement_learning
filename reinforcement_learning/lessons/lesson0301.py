"""Lesson 3.1: The agent–environment interface.

Run: python -m reinforcement_learning.lessons.lesson0301
Read: docs/lessons/lesson0301.md

A delivery robot chooses between a short trip to a nearby customer and a
longer trip to a distant customer. The environment owns the transition and
reward rules; the agent owns its policy. Fixed policies illustrate the
interface, not a learning algorithm. See the companion lesson for discussion
questions and answers about control, knowledge, representation, and goals.
"""

import random
from dataclasses import dataclass
from enum import Enum


class State(Enum):
    DEPOT = "depot"
    ROAD = "road"
    DONE = "done"


class Action(Enum):
    DELIVER_NEAR = "deliver nearby"
    DRIVE_FAR = "drive toward distant customer"
    FINISH_DELIVERY = "finish distant delivery"


@dataclass(frozen=True)
class Outcome:
    """The environment's response: S_{t+1}, R_{t+1}, and termination."""

    next_state: State
    reward: float
    terminated: bool


class DeliveryEnvironment:
    """One delivery per episode; the task's rules do not depend on the policy."""

    def __init__(self) -> None:
        self.state = State.DEPOT

    def reset(self) -> State:
        self.state = State.DEPOT
        return self.state

    @staticmethod
    def available_actions(state: State) -> tuple[Action, ...]:
        """A(s) can differ between states; no actions follow termination."""
        if state is State.DEPOT:
            return (Action.DELIVER_NEAR, Action.DRIVE_FAR)
        if state is State.ROAD:
            return (Action.FINISH_DELIVERY,)
        return ()

    def step(self, action: Action) -> Outcome:
        if action not in self.available_actions(self.state):
            raise ValueError(f"Action {action.value!r} is unavailable in {self.state.value!r}")
        if action is Action.DELIVER_NEAR:
            outcome = Outcome(State.DONE, 2.0, True)
        elif action is Action.DRIVE_FAR:
            outcome = Outcome(State.ROAD, -1.0, False)
        else:
            outcome = Outcome(State.DONE, 5.0, True)
        self.state = outcome.next_state
        return outcome


class PolicyAgent:
    """A fixed policy pi(a|s), with configurable probability of driving far."""

    def __init__(self, probability_far: float, seed: int = 0) -> None:
        if not 0.0 <= probability_far <= 1.0:
            raise ValueError("probability_far must be between zero and one")
        self.probability_far = probability_far
        # Reproducible policy sampling, not security or cryptography.
        self.rng = random.Random(seed)  # nosec B311

    def policy(self, state: State) -> dict[Action, float]:
        if state is State.DEPOT:
            return {
                Action.DELIVER_NEAR: 1.0 - self.probability_far,
                Action.DRIVE_FAR: self.probability_far,
            }
        if state is State.ROAD:
            return {Action.FINISH_DELIVERY: 1.0}
        raise ValueError("There is no decision at a terminal state")

    def choose_action(self, state: State) -> Action:
        probabilities = self.policy(state)
        return self.rng.choices(list(probabilities), weights=list(probabilities.values()), k=1)[0]


@dataclass(frozen=True)
class Transition:
    """One experience tuple (S_t, A_t, R_{t+1}, S_{t+1})."""

    time: int
    state: State
    action: Action
    outcome: Outcome


def run_episode(agent: PolicyAgent, environment: DeliveryEnvironment) -> list[Transition]:
    """Collect experience without modifying either the policy or task rules."""
    state = environment.reset()
    trajectory: list[Transition] = []
    while state is not State.DONE:
        action = agent.choose_action(state)
        outcome = environment.step(action)
        trajectory.append(Transition(len(trajectory), state, action, outcome))
        # A learning agent could use this transition to update its policy here.
        state = outcome.next_state
    return trajectory


def main() -> None:
    print("Lesson 3.1 — One delivery per episode; maximize total episode reward.")
    environment = DeliveryEnvironment()
    for name, probability_far in (("Immediate-reward policy", 0.0), ("Delayed-reward policy", 1.0)):
        agent = PolicyAgent(probability_far)
        print(f"\n{name}: pi(a|depot) = {agent.policy(State.DEPOT)}")
        trajectory = run_episode(agent, environment)
        for transition in trajectory:
            t = transition.time
            print(
                f"S_{t}={transition.state.value}; A_{t}={transition.action.value}; "
                f"R_{t + 1}={transition.outcome.reward:+g}; "
                f"S_{t + 1}={transition.outcome.next_state.value}"
            )
        print(f"Total episode reward: {sum(item.outcome.reward for item in trajectory):g}")
    print("\nThe agent chooses actions, not rewards or next states.")
    print("Knowing the rules and learning a good policy are different questions.")


if __name__ == "__main__":
    main()
