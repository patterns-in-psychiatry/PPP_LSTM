# Psychosis Prognosis Predictor (PPP)

A recurrent neural network architecture for continuous and uncertainty-aware
treatment outcome prediction in first-episode psychosis.

## Stage 1: Python 3.12 core

The new `ppp` package currently provides the installation foundation for the
benchmark. Scenario construction, preprocessing, validation and model adapters
will be added incrementally. No clinical results have been reproduced.

Use a fresh Python 3.12 environment. The core does not require TensorFlow, Keras,
clinical data or model downloads. The resolved development environment is pinned
in `requirements-core.lock` (including transitive dependencies and build tools).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.lock
python -m pip install --no-build-isolation --no-deps -e '.[dev]'
python -m pip check
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate
it with `.venv\Scripts\Activate.ps1` in PowerShell.

The lock records the tested Python 3.12 Linux environment; other operating systems
require validation. To deliberately update dependencies, edit direct pins in
`pyproject.toml`, resolve in a clean environment, regenerate the lock with
`python -m pip freeze --exclude-editable --all` (excluding pip itself), and rerun
the checks. Do not mix the old and new requirements in one environment.

## Original scripts

`LSTM_script.py`, `LSTM_script_one_site_out.py`, `Datasets/`, `Models/`,
`Utilities/`, `Plotting/` and the original `requirements.txt` remain unchanged.
They are not installed as part of `ppp-benchmark` and have not been ported to
the new environment. Git history preserves the original implementation; there
is no duplicated `legacy/` directory. Linting and formatting cover only the new
package, tests and project configuration.

## Development steps

1. **Completed: package foundation.** Installable Python 3.12 core, pinned
   environment, Ruff configuration and one isolated package-import smoke test.
2. **Next: data contracts.** S1–S6, outcome definitions, missing labels,
   temporal availability and patient/site alignment, with minimal focused tests.
3. **Later: benchmark execution.** Fold-local preprocessing, shared validation,
   metrics and model adapters. Scientific corrections will be documented separately.
