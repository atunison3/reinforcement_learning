# Testing

The project uses Python’s standard library **`unittest`** framework (not pytest).

## Unit tests

From the repository root, with the package installed (editable recommended):

```bash
python -m unittest discover -s tests -v
```

This matches pre-commit and CI.

### Targeted runs

```bash
python -m unittest tests.test_playground.test_exercise002 -v
python -m unittest tests.test_playground.test_exercise002.TestExercise002.test_01 -v
```

Test packages live under `tests/test_playground/`.

## Linting and formatting

Commands used by local tooling / CI:

```bash
ruff check reinforcement_learning tests
black --line-length=120 reinforcement_learning tests
```

black is configured in `pyproject.toml` with `line-length = 120` and `target-version = ["py314"]`.

## Type checking

```bash
mypy reinforcement_learning tests
```

`pyproject.toml` enables `strict = true` for package `reinforcement_learning` and sets `python_version = "3.14"`.

## Security

```bash
bandit -r reinforcement_learning tests
```

## All pre-commit hooks

```bash
pre-commit run --all-files
```

## CI parity

`.github/workflows/ci.yml` runs on push and pull request to `main` with Python 3.14 and:

1. Installs filtered `requirements.txt` plus `pip install -e .`
2. `ruff check reinforcement_learning tests`
3. `bandit -r reinforcement_learning tests`
4. `mypy reinforcement_learning tests`
5. `python -m unittest discover -s tests -v`

Run the same commands locally before opening a PR.
