# Contributor Guide

This section is for people changing the source code, tests, or documentation.

## What contributors typically do

1. Clone the repository and create a virtual environment.
2. Install the package in editable mode plus development tools.
3. Enable pre-commit hooks.
4. Make changes under `reinforcement_learning/` or `tests/`.
5. Run lint, type checks, security scans, and unit tests.
6. Update Docsify pages under `docs/` when behavior or setup changes.

## Guides

- [Development Setup](development-setup.md) — environment, editable install, pre-commit, Docsify preview
- [Testing](testing.md) — unittest, ruff, bandit, mypy, CI parity

## Project conventions

- Package code: `reinforcement_learning/`
- Tests: `tests/` discovered with `python -m unittest discover -s tests -v`
- Formatter: black, line length 120
- Linter: ruff
- Types: mypy (strict for the package per `pyproject.toml`)
- Security: bandit on `reinforcement_learning` and `tests`
- License: Apache-2.0
