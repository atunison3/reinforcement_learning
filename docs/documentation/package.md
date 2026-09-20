# Package Reference

## Project metadata

| Field | Value |
| ------- | -------- |
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

`reinforcement_learning/__init__.py` is currently empty. Import project modules explicitly:

```python
from reinforcement_learning.projects import project001
from reinforcement_learning.projects import project002
```

## Modules

| Module | Description |
| -------- | ------------- |
| `reinforcement_learning.projects.project001` | 2-action environment, random agent, sample-average Q updates |
| `reinforcement_learning.projects.project002` | *k*-armed Gaussian bandit, ε-greedy agent, multi-trial runner, plotting |
| `reinforcement_learning.projects.project010` | Finite-shoe blackjack, split-round returns, Hi-Lo count states, Monte Carlo learning |

See:

- [Project 1](project001.md)
- [Project 2](project002.md)
- [Project 10 — Card Counting](../projects/project10.md)

## Dependencies

### Declared in `pyproject.toml`

- **Runtime (`[project].dependencies`)**: none
- **Optional dev (`[project.optional-dependencies].dev`)**: `bandit`, `black`, `mypy`, `pre-commit`, `ruff`

### Used by project code

These are imported by the project modules and appear in `requirements.txt`:

- `numpy`
- `matplotlib`

Install them when running Project 1 or Project 2.

## Tooling defaults

From `pyproject.toml`:

- **black** / **ruff** line length: `120`
- **ruff** / **black** target: `py314`
- **mypy**: `strict = true`, package `reinforcement_learning`

## Entry points

There are no `[project.scripts]` console entry points. Run modules with `python -m ...` or import them from Python.
