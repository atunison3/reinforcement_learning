# Usage

There is no console script entry point in `pyproject.toml`. Use the package by importing modules or running playground files as scripts.

## Import the playground modules

```python
from reinforcement_learning.playground.exercise001 import Agent, Environment
from reinforcement_learning.playground.exercise002 import (
    Agent as BanditAgent,
    Environment as BanditEnvironment,
    run_experiment,
    run_experiments,
    plot_results,
)
```

## Exercise 1 — 2-action environment

Train a random policy for 1000 episodes and print learned action values:

```python
from reinforcement_learning.playground.exercise001 import Agent, Environment

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
```

Or run the module directly from the repository root:

```bash
python -m reinforcement_learning.playground.exercise001
```

## Exercise 2 — 10-armed bandit

Run repeated trials for several ε values and write comparison plots under `docs/exercises/assets/`:

```python
from reinforcement_learning.playground.exercise002 import plot_results, run_experiments

results = run_experiments(
    epsilons=[0.0, 0.01, 0.1],
    trials=2000,
    steps=1000,
)
plot_results(results)
```

Or run the module directly:

```bash
python -m reinforcement_learning.playground.exercise002
```

`plot_results` saves:

- `docs/exercises/assets/exercise_2_average_reward.png`
- `docs/exercises/assets/exercise_2_optimal_action_percentage.png`

Run from the repository root so those relative paths resolve correctly.

## Single ε experiment

```python
from reinforcement_learning.playground.exercise002 import run_experiment

average_rewards, percent_optimal = run_experiment(
    trials=2000,
    steps=1000,
    epsilon=0.1,
)
```

`average_rewards` and `percent_optimal` are NumPy arrays of length `steps`.

## Next steps

- Module-level API details: [Package Reference](../documentation/package.md)
- Exercise 2 plots in the docs: [Exercise Plots](../exercises/exercise.md)
