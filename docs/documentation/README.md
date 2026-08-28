# Documentation

Detailed documentation for the `reinforcement_learning` package.

## Contents

- [Package Reference](package.md) — layout, version, and dependencies
- [Playground Exercise 1](playground-exercise001.md) — 2-action cliff/safe bandit
- [Playground Exercise 2](playground-exercise002.md) — *k*-armed Gaussian bandit and experiment helpers
- [Exercise Plots](../exercises/exercise.md) — saved Exercise 2 figures

## Package layout

```text
reinforcement_learning/
├── __init__.py
└── playground/
    ├── exercise001.py
    └── exercise002.py
```

Public learning code currently lives under `reinforcement_learning.playground`.

## Related project areas

| Path | Role |
|------|------|
| `tests/` | `unittest` discovery root |
| `docs/` | Docsify documentation site |
| `notes/` | Chapter study notes (Markdown) |
| `.pre-commit-config.yaml` | Local hooks (black, ruff, bandit, mypy, unittest) |
| `.github/workflows/ci.yml` | CI checks on `main` |
