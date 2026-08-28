# Playground Exercise 2

Module: `reinforcement_learning.playground.exercise002`

A *k*-armed Gaussian bandit similar in spirit to the Chapter 2 10-armed testbed. Agents select actions greedily or with ε-greedy exploration and update sample-average action values.

## `Environment`

```python
class Environment:
    def __init__(self, k: int = 10): ...
    def reset(self) -> None: ...
    def step(self, action: int) -> tuple[float, bool]: ...
```

### `__init__`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `k` | `10` | Number of arms |

Initializes `rewards` to zeros and `optimal_action` to `0` until `reset()`.

### `reset`

Samples true action means from \(\mathcal{N}(0, 1)\) into `self.rewards` (currently always length 10 via `np.random.normal(0, 1, 10)`), then sets `optimal_action` to the index of the maximum mean.

Returns `None`.

### `step`

```python
step(self, action: int) -> tuple[float, bool]
```

- Raises `ValueError` if `action < 0` or `action >= k`
- Samples reward from \(\mathcal{N}(\mu_a, 1)\) where \(\mu_a = \texttt{self.rewards[action]}\)
- Returns `(reward, optimal)` where `optimal` is whether `action == optimal_action`

## `Agent`

```python
class Agent:
    def __init__(self, epsilon: float = 0.0, k: int = 10): ...
    def choose_action(self) -> int: ...
    def update(self, action: int, reward: float) -> None: ...
```

### `__init__`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `epsilon` | `0.0` | Exploration probability |
| `k` | `10` | Number of arms |

Raises `ValueError` if `epsilon < 0` or `epsilon > 1`.

Initializes:

- `q` — zeros length `k` (action-value estimates)
- `n` — zeros length `k` (counts)

### `choose_action`

```python
choose_action(self) -> int
```

- With probability `epsilon`, returns a uniform random action in `0 .. k-1`
- Otherwise returns a random tie-break among actions with maximum `q`

### `update`

```python
update(self, action: int, reward: float) -> None
```

Incremental sample-average update for the chosen action:

\[
Q_a \leftarrow Q_a + \frac{1}{N_a}(R - Q_a)
\]

## `run_experiment`

```python
run_experiment(
    trials: int = 2000,
    steps: int = 1000,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]
```

For each trial: create environment and agent, call `env.reset()`, then for `steps` iterations choose an action, step, record reward/optimality, and update the agent.

Returns:

1. `average_rewards` — mean reward per step across trials, shape `(steps,)`
2. `percent_optimal` — percent of trials that chose the optimal action at each step, shape `(steps,)`

Prints a Unicode progress bar while running.

## `run_experiments`

```python
run_experiments(
    epsilons: list[float],
    trials: int = 2000,
    steps: int = 1000,
) -> dict[float, tuple[np.ndarray, np.ndarray]]
```

Runs `run_experiment` once per ε and returns a mapping:

```text
epsilon -> (average_rewards, percent_optimal)
```

## `plot_results`

```python
plot_results(
    results: dict[float, tuple[np.ndarray, np.ndarray]],
) -> None
```

Writes two PNG files (relative to the process working directory):

- `docs/exercises/assets/exercise_2_average_reward.png`
- `docs/exercises/assets/exercise_2_optimal_action_percentage.png`

## `loading_bar`

```python
loading_bar(current: int, total: int, width: int = 20) -> str
```

Returns a string such as `[████░░░░]  40.00%` used by `run_experiment`.

## Script behavior

```bash
python -m reinforcement_learning.playground.exercise002
```

Runs:

```python
run_experiments(epsilons=[0.0, 0.01, 0.1], trials=2000, steps=1000)
plot_results(results)
```

## Example

```python
from reinforcement_learning.playground.exercise002 import (
    plot_results,
    run_experiments,
)

results = run_experiments(epsilons=[0.0, 0.1], trials=100, steps=500)
plot_results(results)
```

Saved figures are also embedded under [Exercise Plots](../exercises/exercise.md).
