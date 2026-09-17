"""Tabular contextual-bandit learning and the value of observing context."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from reinforcement_learning.policies import epsilon_greedy


def optimal_expected_rewards(
    action_values: NDArray[np.float64],
    context_probabilities: NDArray[np.float64],
) -> tuple[float, float]:
    """Return optimal expected rewards (without context, with context).

    Rows of ``action_values`` are contexts and columns are actions. Contexts
    must be drawn independently of previous interactions and current actions.
    The context-aware optimum assumes the context is observed before acting.
    These are known-value benchmarks, not estimates available to a learner.
    """

    if action_values.ndim != 2 or 0 in action_values.shape:
        raise ValueError("`action_values` must be a nonempty 2D array")
    if not np.all(np.isfinite(action_values)):
        raise ValueError("`action_values` must be finite")
    if context_probabilities.ndim != 1 or len(context_probabilities) != action_values.shape[0]:
        raise ValueError("`context_probabilities` must have one entry per context")
    if (
        not np.all(np.isfinite(context_probabilities))
        or np.any(context_probabilities < 0)
        or not np.isclose(np.sum(context_probabilities), 1.0)
    ):
        raise ValueError("`context_probabilities` must be nonnegative and sum to 1")

    without_context = float(np.max(context_probabilities @ action_values))
    with_context = float(context_probabilities @ np.max(action_values, axis=1))
    return without_context, with_context


class ContextualBanditAgent:
    """Learn a separate sample-average action-value table for each context.

    Using one context recovers an ordinary nonassociative bandit agent.
    Action selection reuses ``policies.epsilon_greedy`` and its NumPy RNG.
    """

    def __init__(self, n_contexts: int, n_actions: int, epsilon: float = 0.1) -> None:
        if n_contexts < 1 or n_actions < 1:
            raise ValueError("Context and action counts must be positive")
        if not 0 <= epsilon <= 1:
            raise ValueError("`epsilon` must be between 0 and 1")

        self.q: NDArray[np.float64] = np.zeros((n_contexts, n_actions))
        self.counts: NDArray[np.int64] = np.zeros((n_contexts, n_actions), dtype=np.int64)
        self.epsilon = epsilon

    def choose_action(self, context: int, legal_actions: Sequence[int] | None = None) -> int:
        """Select epsilon-greedily among the context's legal actions only.

        Both exploration and exploitation respect ``legal_actions``. Omitting
        it retains the usual behavior in which every action is available.
        """

        if not 0 <= context < self.q.shape[0]:
            raise ValueError(f"Invalid context: {context}")
        if legal_actions is None:
            return epsilon_greedy(self.q[context], self.epsilon)

        actions = np.asarray(legal_actions, dtype=np.int64)
        if actions.ndim != 1 or actions.size == 0:
            raise ValueError("`legal_actions` must be a nonempty 1D sequence")
        if np.any(actions < 0) or np.any(actions >= self.q.shape[1]) or len(np.unique(actions)) != len(actions):
            raise ValueError("`legal_actions` must contain distinct valid action indices")
        selected = epsilon_greedy(self.q[context, actions], self.epsilon)
        return int(actions[selected])

    def update(self, context: int, action: int, reward: float) -> None:
        """Update only the context-action pair that generated the reward."""

        if not 0 <= context < self.q.shape[0]:
            raise ValueError(f"Invalid context: {context}")
        if not 0 <= action < self.q.shape[1]:
            raise ValueError(f"Invalid action: {action}")
        if not np.isfinite(reward):
            raise ValueError("`reward` must be finite")

        self.counts[context, action] += 1
        self.q[context, action] += (reward - self.q[context, action]) / self.counts[context, action]
