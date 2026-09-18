"""Gradient-bandit learning and analytical reward-baseline diagnostics."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

BaselineKind = Literal["fixed", "average", "ema"]


class GradientBanditBatch:
    """Independent softmax learners with fixed, running-mean, or EMA baselines.

    Rows are independent trials, not observations shared between learners.
    Each baseline is computed from preceding rewards and updated only after
    the preference update. It never depends on the current sampled action.
    """

    def __init__(
        self,
        trials: int,
        k: int = 10,
        alpha: float = 0.1,
        baseline_kind: BaselineKind = "average",
        initial_baseline: float = 0.0,
        beta: float = 0.1,
    ) -> None:
        if trials < 1 or k < 2:
            raise ValueError("Use at least one trial and two actions")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("`alpha` must be positive and finite")
        if baseline_kind not in ("fixed", "average", "ema"):
            raise ValueError("Unknown baseline kind")
        if not np.isfinite(initial_baseline):
            raise ValueError("`initial_baseline` must be finite")
        if not np.isfinite(beta) or not 0 < beta <= 1:
            raise ValueError("`beta` must be in (0, 1]")
        self.preferences: NDArray[np.float64] = np.zeros((trials, k))
        self.baseline: NDArray[np.float64] = np.full(trials, initial_baseline, dtype=np.float64)
        self.rows = np.arange(trials)
        self.alpha = alpha
        self.baseline_kind = baseline_kind
        self.beta = beta
        self.steps = 0

    def probabilities(self) -> NDArray[np.float64]:
        """Stable softmax: subtracting each row's maximum preserves its policy."""

        shifted = self.preferences - np.max(self.preferences, axis=1, keepdims=True)
        weights: NDArray[np.float64] = np.exp(shifted)
        return weights / np.sum(weights, axis=1, keepdims=True)

    def choose_actions(self, rng: np.random.Generator) -> NDArray[np.int64]:
        """Sample one action per trial, with no separate epsilon exploration."""

        cumulative = np.cumsum(self.probabilities(), axis=1)
        actions: NDArray[np.int64] = np.sum(rng.random((len(self.rows), 1)) >= cumulative, axis=1, dtype=np.int64)
        # Guard against a last cumulative probability just below one by roundoff.
        actions[actions >= self.preferences.shape[1]] = self.preferences.shape[1] - 1
        return actions

    def update(self, actions: NDArray[np.int64], rewards: NDArray[np.float64]) -> None:
        """Apply alpha * (reward - old baseline) * (one_hot(action) - policy)."""

        if actions.shape != self.baseline.shape or rewards.shape != self.baseline.shape:
            raise ValueError("Provide one action and reward per trial")
        if np.any(actions < 0) or np.any(actions >= self.preferences.shape[1]):
            raise ValueError("Invalid action")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("Rewards must be finite")
        advantage = rewards - self.baseline
        self.preferences -= self.alpha * advantage[:, None] * self.probabilities()
        self.preferences[self.rows, actions] += self.alpha * advantage
        self.steps += 1
        if self.baseline_kind != "fixed":
            rate = 1.0 / self.steps if self.baseline_kind == "average" else self.beta
            self.baseline += rate * (rewards - self.baseline)


def _validate_policy(probabilities: NDArray[np.float64], action_values: NDArray[np.float64]) -> None:
    if probabilities.ndim != 1 or probabilities.size == 0 or action_values.shape != probabilities.shape:
        raise ValueError("Provide matching nonempty 1D probabilities and action values")
    if not np.all(np.isfinite(action_values)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities and action values must be finite")
    if np.any(probabilities < 0) or not np.isclose(np.sum(probabilities), 1.0):
        raise ValueError("Probabilities must be nonnegative and sum to one")


def gradient_moments(
    probabilities: NDArray[np.float64],
    action_values: NDArray[np.float64],
    baseline: float,
    reward_std: float = 1.0,
) -> tuple[NDArray[np.float64], float]:
    """Return E[g] and trace(Cov[g]) at a fixed policy, before multiplying by alpha.

    Here g = (R - baseline) * (one_hot(A) - probabilities), with reward mean
    action_values[A] and common conditional standard deviation reward_std.
    Known action values are used for this diagnostic only, never by learners.
    """

    _validate_policy(probabilities, action_values)
    if not np.isfinite(baseline) or not np.isfinite(reward_std) or reward_std < 0:
        raise ValueError("Use a finite baseline and finite nonnegative reward standard deviation")
    mean_reward = float(probabilities @ action_values)
    expected_gradient = probabilities * (action_values - mean_reward)
    scores = np.eye(len(probabilities)) - probabilities
    squared_norms = np.sum(scores**2, axis=1)
    second_moment = float((probabilities * squared_norms) @ ((action_values - baseline) ** 2 + reward_std**2))
    variance = max(0.0, second_moment - float(expected_gradient @ expected_gradient))
    return expected_gradient, variance


def minimum_variance_baseline(probabilities: NDArray[np.float64], action_values: NDArray[np.float64]) -> float:
    """Oracle scalar baseline minimizing the total variance of the score gradient.

    This weights rewards by squared score norms, not just action probabilities.
    With a deterministic policy the gradient is zero for any baseline; return
    the expected reward in that degenerate case.
    """

    _validate_policy(probabilities, action_values)
    scores = np.eye(len(probabilities)) - probabilities
    weights = probabilities * np.sum(scores**2, axis=1)
    total = float(np.sum(weights))
    return float(weights @ action_values / total) if total > 0 else float(probabilities @ action_values)
