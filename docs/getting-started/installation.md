# Installation

These steps install the package for **use** (running playground modules and importing the library).

## Requirements

- Python **3.14** or newer (`requires-python = ">=3.14"` in `pyproject.toml`)
- `pip`
- Runtime libraries used by the playground code: **NumPy** and **Matplotlib** (listed in `requirements.txt`; not declared under `[project].dependencies` in `pyproject.toml`)

The project is not assumed to be published on PyPI. Install from a local clone.

## Clone the repository

```bash
git clone https://github.com/atunison3/reinforcement_learning.git
cd reinforcement_learning
```

## Create and activate a virtual environment

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

## Install the package and runtime dependencies

Install the local package, then install pinned dependencies from `requirements.txt` (skipping any editable VCS line if present):

```bash
python -m pip install --upgrade pip
python -m pip install .
grep -v '^-e ' requirements.txt > /tmp/rl-requirements.txt
python -m pip install -r /tmp/rl-requirements.txt
```

### Windows (PowerShell)

```powershell
python -m pip install --upgrade pip
python -m pip install .
Get-Content requirements.txt | Where-Object { $_ -notmatch '^-e ' } | Set-Content $env:TEMP\rl-requirements.txt
python -m pip install -r $env:TEMP\rl-requirements.txt
```

Alternatively, install NumPy and Matplotlib directly after the package:

```bash
python -m pip install .
python -m pip install numpy matplotlib
```

## Verify the installation

```bash
python -c "from reinforcement_learning.playground import exercise001, exercise002; print('ok')"
```

If the import succeeds, the package is available on your `PYTHONPATH` / site-packages.
