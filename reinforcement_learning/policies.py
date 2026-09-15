"""Action-selection strategies for reinforcement learning agents."""

import numpy as np


def epsilon_greedy(values: np.ndarray, epsilon: float) -> int:
    """Choose an action using an epsilon-greedy strategy.

    With probability `epsilon`, choose a random action. Otherwise choose
    uniformly among the actions with the largest estimated value.
    """

    if values.ndim != 1:
        raise ValueError("`values` must be a 1D array")
    if not 0 <= epsilon <= 1:
        raise ValueError("`epsilon` must be between 0 and 1")

    if (epsilon > 0) and (np.random.random() < epsilon):
        return int(np.random.choice(len(values)))

    max_value = np.max(values)
    best_actions = np.flatnonzero(values == max_value)
    return int(np.random.choice(best_actions))


def upper_confidence_bound(values: np.ndarray, counts: np.ndarray, step: int, c: float) -> int:
    """Choose an action using the Upper Confidence Bound (UCB) rule.

    The selected action maximizes

    Q_t(a) + c * sqrt(log(t) / N_t(a))

    where `values` contains `Q_t(a)`, `counts` contains `N_t(a)`, `step` is
    the current time step `t`, and `c` controls the exploration bonus.
    Untried actions are selected before tried actions.
    """

    if values.ndim != 1 or counts.ndim != 1:
        raise ValueError("`values` and `counts` must be 1D arrays")
    if len(values) != len(counts):
        raise ValueError("`values` and `counts` must have the same length")
    if step < 1:
        raise ValueError("`step` must be at least 1")
    if c < 0:
        raise ValueError("`c` must be non-negative")

    untried_actions = np.flatnonzero(counts == 0)
    if len(untried_actions) > 0:
        return int(np.random.choice(untried_actions))

    confidence_bonus = c * np.sqrt(np.log(step) / counts)
    ucb_values = values + confidence_bonus

    max_value = np.max(ucb_values)
    best_actions = np.flatnonzero(ucb_values == max_value)
    return int(np.random.choice(best_actions))
