# Package Reference

## Project metadata

| Field | Value |
|-------|--------|
| Name | `reinforcement_learning` |
| Version | `0.1.0` |
| License | Apache-2.0 |
| Python | `>=3.14` |
| Build backend | `setuptools.build_meta` |

Source: `pyproject.toml`.

## Import root

```python
import reinforcement_learning
```

`reinforcement_learning/__init__.py` is currently empty. Import playground modules explicitly:

```python
from reinforcement_learning.playground import exercise001
from reinforcement_learning.playground import exercise002
```

## Modules

| Module | Description |
|--------|-------------|
| `reinforcement_learning.playground.exercise001` | 2-action environment, random agent, sample-average Q updates |
| `reinforcement_learning.playground.exercise002` | *k*-armed Gaussian bandit, ε-greedy agent, multi-trial runner, plotting |

See:

- [Playground Exercise 1](playground-exercise001.md)
- [Playground Exercise 2](playground-exercise002.md)

## Dependencies

### Declared in `pyproject.toml`

- **Runtime (`[project].dependencies`)**: none
- **Optional dev (`[project.optional-dependencies].dev`)**: `bandit`, `black`, `mypy`, `pre-commit`, `ruff`

### Used by playground code

These are imported by the playground modules and appear in `requirements.txt`:

- `numpy`
- `matplotlib`

Install them when running Exercise 1 or Exercise 2.

## Tooling defaults

From `pyproject.toml`:

- **black** / **ruff** line length: `120`
- **ruff** / **black** target: `py314`
- **mypy**: `strict = true`, package `reinforcement_learning`

## Entry points

There are no `[project.scripts]` console entry points. Run modules with `python -m ...` or import them from Python.
