# Reinforcement Learning Playground

A Python package and notebook-free playground for working through reinforcement learning ideas while reading *Reinforcement Learning* by Phil Winder.

## What is reinforcement_learning?

This repository records study notes, small bandit environments, and agent implementations. The installable package lives under `reinforcement_learning/` and currently focuses on multi-armed bandit playgrounds used to build intuition for rewards, action values, exploration, and exploitation.

## Why use it?

Use this project when you want to:

- Experiment with simple RL environments in plain Python.
- Compare greedy and ε-greedy action selection on a 10-armed Gaussian bandit.
- Keep study notes, plots, and code in one place.
- Run the same formatting, lint, type-check, security, and unit-test gates locally and in CI.

## Features

- **Playground Exercise 1** — 2-action cliff/safe environment with a uniformly random policy and incremental action-value updates.
- **Playground Exercise 2** — *k*-armed Gaussian bandit with ε-greedy agents, repeated trials, and saved performance plots.
- **Packaging** — setuptools project (`pyproject.toml`) installable with pip.
- **Quality gates** — black, ruff, bandit, mypy, and `unittest` via pre-commit and GitHub Actions.
- **Docsify site** — this documentation set under `docs/`.

## Quick Example

```python
from reinforcement_learning.playground.exercise001 import Agent, Environment

env = Environment()
agent = Agent()

state = env.reset()
action = agent.choose_action(state)
next_state, reward, terminated = env.step(action)
agent.update(state, action, reward, next_state)

print(agent.q)
```

## Documentation

For installation, usage, module reference, and contributor workflow, see the
[full documentation](documentation/README.md).

## Contributing

Developers who want to extend the package should start with the
[contributor guide](contributing/README.md).
