"""Lesson 3.2: Goals and rewards.

Run: python -m reinforcement_learning.lessons.lesson0302
Read: docs/lessons/lesson0302.md

The intended goal is to recycle a can within six decisions. Compare rewarding
actual recycling with rewarding a proxy: picking up the can. The same physical
rules and fixed policies are evaluated under both reward definitions.

This deterministic example evaluates policies; it does not train an agent.
Its objective is total, undiscounted reward over a fixed finite horizon.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class Can(Enum):
    ON_FLOOR = "on floor"
    HELD = "held"
    RECYCLED = "recycled"


class Action(Enum):
    PICK_UP = "pick up"
    PUT_DOWN = "put down"
    RECYCLE = "recycle"
    WAIT = "wait"


class RewardRule(Enum):
    RECYCLING = "+1 for actual recycling"
    PICKUP = "+1 for each pickup (proxy)"


@dataclass(frozen=True)
class State:
    """Can status and time remaining in this finite-horizon task."""

    can: Can
    steps_remaining: int


@dataclass(frozen=True)
class Transition:
    """One experience: S_t, A_t, R_{t+1}, S_{t+1}."""

    state: State
    action: Action
    reward: float
    next_state: State


class RecyclingEnvironment:
    """One can, an irreversible recycling bin, and an externally set reward rule."""

    def __init__(self, reward_rule: RewardRule, horizon: int = 6) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.reward_rule = reward_rule
        self.horizon = horizon
        self.state = State(Can.ON_FLOOR, horizon)

    def reset(self) -> State:
        self.state = State(Can.ON_FLOOR, self.horizon)
        return self.state

    @staticmethod
    def available_actions(state: State) -> tuple[Action, ...]:
        if state.steps_remaining == 0:
            return ()
        if state.can is Can.ON_FLOOR:
            return (Action.PICK_UP, Action.WAIT)
        if state.can is Can.HELD:
            return (Action.RECYCLE, Action.PUT_DOWN, Action.WAIT)
        return (Action.WAIT,)

    def step(self, action: Action) -> Transition:
        if action not in self.available_actions(self.state):
            raise ValueError(f"Action {action.value!r} is unavailable in {self.state}")
        state = self.state
        can = state.can
        if action is Action.PICK_UP:
            can = Can.HELD
        elif action is Action.PUT_DOWN:
            can = Can.ON_FLOOR
        elif action is Action.RECYCLE:
            can = Can.RECYCLED

        # This rule belongs to the environment, not to the policy.
        rewarded_action = Action.RECYCLE if self.reward_rule is RewardRule.RECYCLING else Action.PICK_UP
        reward = 1.0 if action is rewarded_action else 0.0
        self.state = State(can, state.steps_remaining - 1)
        return Transition(state, action, reward, self.state)


Policy = Callable[[State], Action]


def recycle_policy(state: State) -> Action:
    """Pick up the can, recycle it, then wait until the horizon ends."""
    if state.can is Can.ON_FLOOR:
        return Action.PICK_UP
    if state.can is Can.HELD:
        return Action.RECYCLE
    return Action.WAIT


def repeat_pickup_policy(state: State) -> Action:
    """Pick up and put down the same can repeatedly, never recycling it."""
    if state.can is Can.ON_FLOOR:
        return Action.PICK_UP
    if state.can is Can.HELD:
        return Action.PUT_DOWN
    return Action.WAIT


def run_episode(environment: RecyclingEnvironment, policy: Policy) -> tuple[Transition, ...]:
    """Evaluate a fixed policy; there are no policy updates in this lesson."""
    state = environment.reset()
    trajectory: list[Transition] = []
    while state.steps_remaining > 0:
        transition = environment.step(policy(state))
        trajectory.append(transition)
        state = transition.next_state
    return tuple(trajectory)


def main() -> None:
    print("Lesson 3.2 — Goals and rewards")
    print("Intended goal: recycle the can. Objective: total reward over six decisions.")
    policies: tuple[tuple[str, Policy], ...] = (
        ("Recycle", recycle_policy),
        ("Repeat pickup", repeat_pickup_policy),
    )
    for rule in RewardRule:
        print(f"\nReward rule: {rule.value}")
        for name, policy in policies:
            trajectory = run_episode(RecyclingEnvironment(rule), policy)
            total = sum(transition.reward for transition in trajectory)
            # Measure the actual outcome separately from the reward signal.
            cans_recycled = int(trajectory[-1].next_state.can is Can.RECYCLED)
            print(f"  {name}: return={total:g}; cans recycled={cans_recycled}")
            print("    Actions: " + " -> ".join(transition.action.value for transition in trajectory))
            print("    Rewards: " + ", ".join(f"{transition.reward:g}" for transition in trajectory))
    print("\nA larger return is not evidence of success when the reward measures the wrong thing.")
    print("These policies are hand-written; no learning or convergence is demonstrated.")


if __name__ == "__main__":
    main()
