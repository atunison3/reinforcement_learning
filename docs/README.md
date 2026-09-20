# Reinforcement Learning

A Python package for studying reinforcement learning through concept lessons, responses to Sutton and Barto's *Reinforcement Learning: An Introduction*, and independent projects.

## Repository areas

- **Lessons** — focused explanations of reinforcement-learning concepts and supporting code.
- **Exercises** — reserved for responses to actual Sutton and Barto exercises; currently empty.
- **Projects** — personal experiments and systems, including bandits and [blackjack card counting](projects/project10.md).
- **Notes** — chapter-level study notes.

The installable package lives under `reinforcement_learning/`; the Docsify site lives under `docs/`.

## Quick example

```python
from reinforcement_learning.projects.project001 import Agent, Environment

env = Environment()
agent = Agent()
state = env.reset()
action = agent.choose_action(state)
next_state, reward, terminated = env.step(action)
agent.update(state, action, reward, next_state)
```

## Explore

- [Lessons](lessons.md)
- [Exercises](exercises/README.md)
- [Projects](projects.md)
- [Documentation overview](documentation/README.md)
- [Getting started](getting-started/installation.md)
