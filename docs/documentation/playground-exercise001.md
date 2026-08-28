# Playground Exercise 1

Module: `reinforcement_learning.playground.exercise001`

A minimal 2-armed setup used to practice the environment / agent loop.

- Action `0` (left): cliff, reward `-1`, episode ends
- Action `1` (right): safe, reward `+1`, episode ends
- Policy: uniformly random choice between `{0, 1}`
- Learning: incremental sample-average update of action values

## `Environment`

```python
class Environment:
    def reset(self) -> int: ...
    def step(self, action: int) -> tuple[int, int, bool]: ...
```

### `reset`

Returns the base state `1`.

### `step`

```python
step(self, action: int) -> tuple[int, int, bool]
```

Applies an action and returns `(next_state, reward, terminated)`.

| Action | Result |
|--------|--------|
| `0` | `(1, -1, True)` |
| `1` | `(1, 1, True)` |
| other | raises `ValueError` |

## `Agent`

```python
class Agent:
    def __init__(self) -> None: ...
    def choose_action(self, state: int) -> int: ...
    def update(self, state: int, action: int, reward: int, next_state: int) -> None: ...
```

### State

On init:

- `q = {1: [0.0, 0.0]}` — estimated values for actions `0` and `1` in state `1`
- `n = {1: [0.0, 0.0]}` — visit counts per action

### `choose_action`

```python
choose_action(self, state: int) -> int
```

- Requires `state == 1`; otherwise raises `ValueError`
- Returns `0` or `1` via `numpy.random.choice`

### `update`

```python
update(self, state: int, action: int, reward: int, next_state: int) -> None
```

Increments the action count and updates the sample average:

\[
Q \leftarrow Q + \frac{1}{N}(R - Q)
\]

`next_state` is accepted for interface symmetry and is not used in the update body.

## Script behavior

Running the module as `__main__` creates an environment and agent, runs 1000 episodes, and prints `agent.q`.

```bash
python -m reinforcement_learning.playground.exercise001
```

## Example

```python
from reinforcement_learning.playground.exercise001 import Agent, Environment

env = Environment()
agent = Agent()

state = env.reset()
action = agent.choose_action(state)
next_state, reward, done = env.step(action)
agent.update(state, action, reward, next_state)
```
