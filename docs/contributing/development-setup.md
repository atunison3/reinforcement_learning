# Development Setup

## Clone

```bash
git clone https://github.com/atunison3/reinforcement_learning.git
cd reinforcement_learning
```

## Virtual environment

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Use Python 3.14+ to match `requires-python` and CI.

## Editable install

Editable mode reflects local source changes without reinstalling after every edit:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Development dependencies

### Option A — optional extras from `pyproject.toml`

```bash
python -m pip install -e ".[dev]"
```

This installs: `bandit`, `black`, `mypy`, `pre-commit`, `ruff`.

### Option B — pinned `requirements.txt`

CI filters out editable VCS lines, then installs requirements and the local package:

```bash
grep -v '^-e ' requirements.txt > /tmp/rl-requirements.txt
python -m pip install -r /tmp/rl-requirements.txt
python -m pip install -e .
```

### Windows (PowerShell) for Option B

```powershell
Get-Content requirements.txt | Where-Object { $_ -notmatch '^-e ' } | Set-Content $env:TEMP\rl-requirements.txt
python -m pip install -r $env:TEMP\rl-requirements.txt
python -m pip install -e .
```

Playground modules also need **NumPy** and **Matplotlib** (included in `requirements.txt`).

## Pre-commit

Hooks are defined in `.pre-commit-config.yaml` and use `language: system`, so tools must be on your `PATH` (your activated `.venv`).

```bash
pre-commit install
pre-commit run --all-files
```

Hooks include:

| Hook | Role |
|------|------|
| trailing-whitespace / end-of-file-fixer | Whitespace cleanup on `reinforcement_learning/` and `tests/` |
| black | Format Python (`--line-length=120`) |
| ruff | Lint Python under those trees |
| bandit | Security scan of `reinforcement_learning` and `tests` |
| mypy | Type-check matching files |
| unittest | `python -m unittest discover -s tests -v` |

## Preview documentation

This site is static Docsify under `docs/`. Serve the folder with any static file server.

### Using docsify-cli (if installed)

```bash
npm install -g docsify-cli
docsify serve docs
```

### Using Python’s HTTP server

### macOS / Linux

```bash
cd docs
python -m http.server 3000
```

### Windows

```powershell
cd docs
python -m http.server 3000
```

Open `http://localhost:3000` in a browser.

## Repository map for contributors

```text
reinforcement_learning/   # package source
tests/                    # unit tests
docs/                     # Docsify site
notes/                    # chapter notes
.pre-commit-config.yaml   # local hooks
.github/workflows/ci.yml  # CI pipeline
pyproject.toml            # packaging and tool config
requirements.txt          # pinned environment packages
```
