# Documentation

Detailed documentation for the `reinforcement_learning` package.

## Contents

- [Package Reference](package.md) — layout, version, and dependencies
- [Project 1](project001.md) — 2-action cliff/safe bandit
- [Project 2](project002.md) — *k*-armed Gaussian bandit and experiment helpers
- [Book exercises](../exercises/README.md) — reserved for textbook exercise responses
- [Project results](../projects.md) — personal experiments, explanations, and saved figures
- [Lessons](../lessons.md) — concept-focused lesson material
- [Projects](../projects.md) — independent project space

## Package layout

```text
reinforcement_learning/
├── lessons/       # concept-focused lessons
├── exercises/     # Sutton and Barto exercise responses
└── projects/      # independent projects and experiments
```

The nine existing experiments live under `reinforcement_learning.projects`, named `project001.py` through `project009.py`. Their numbering identifies personal projects, not textbook exercises. `reinforcement_learning.exercises` is reserved for actual book responses; lesson material has its own area.

## Related project areas

| Path | Role |
| ------ | ------ |
| `tests/` | `unittest` discovery root |
| `docs/` | Docsify documentation site |
| `notes/` | Chapter study notes (Markdown) |
| `.pre-commit-config.yaml` | Local hooks (black, ruff, bandit, mypy, unittest) |
| `.github/workflows/ci.yml` | CI checks on `main` |
