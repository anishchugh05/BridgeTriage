# Massachusetts Bridge Inspection Planner

This repository contains the Streamlit decision-support application for prioritizing additional Massachusetts bridge inspections. Production mode uses the version-controlled real Massachusetts bridge panel, the repository's real model service, deterministic planner, and measured backtest engine. Demo mode is an explicitly selected, fabricated, visibly labeled alternative; it is never a production fallback.

## Current production release

- **Public repository:** [https://github.com/anishchugh05/hackachusetts-2027](https://github.com/anishchugh05/hackachusetts-2027)
- **Deployment branch:** [`person-a-app`](https://github.com/anishchugh05/hackachusetts-2027/tree/person-a-app)
- **Production commit:** [`8d93c21`](https://github.com/anishchugh05/hackachusetts-2027/commit/8d93c21cb67b1d2acea8aa35da4d930a30509d96)
- **Validation:** the [GitHub Actions `Validate` run](https://github.com/anishchugh05/hackachusetts-2027/actions/runs/40135153272) for that commit completed successfully. Local validation reported **99 tests passing**, and the production smoke check passed in about **17 seconds**.
- **Hosted-app status:** this guide does not claim a public Streamlit URL is available yet. Use the Community Cloud checklist below to create or verify a deployment.

## Teammate handoff: pull and run

Run all commands from the repository root after cloning. For a fresh checkout of the deployment branch:

```text
git clone --branch person-a-app --single-branch https://github.com/anishchugh05/hackachusetts-2027.git
cd hackachusetts-2027
```

For an existing clone, update the same branch without rewriting local history:

```text
git fetch origin
git switch person-a-app
git pull --ff-only origin person-a-app
```

The supported runtime is **Python 3.11**, as declared in `.python-version`. Create and activate an isolated environment.

**PowerShell (Windows):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

**POSIX shell (Linux/macOS):**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

`python --version` must report Python 3.11.x. Install the exact, hashed runtime lock and launch the application from the repository root:

```text
python -m pip install --require-hashes -r requirements.txt
streamlit run app/app.py
```

No mode setting is needed for production because production is the default.

## Runtime modes and provenance

`BRIDGE_APP_MODE` accepts only the exact, case-sensitive values `production` and `demo`.

- If `BRIDGE_APP_MODE` is absent, the application runs in production mode.
- If it is exactly `production`, the application explicitly runs in production mode.
- If it is exactly `demo`, the application runs in demo mode with fabricated data and calculations. Every data-derived view and download is preceded by `DEMO MODE - FABRICATED DATA`.
- Empty, whitespace-padded, differently cased, or otherwise invalid values activate neither mode. The application displays a user-safe runtime-mode error before any data-derived output.

Production mode uses only these real, version-controlled components:

- **Real_Dataset:** `bridges/panel.parquet`
- **Real_Model_Service:** the repository `model` package
- **Real_Planner:** the repository `planner` package
- **Backtest_Engine:** the real historical backtest exported by `planner`

Production never substitutes `app/stubs.py`, fabricated data, demo calculations, or any other fallback. If a production data, schema, model, planner, or backtest component fails, the application stays in production mode, keeps demo mode deactivated, and displays a user-safe error for the failed component. Output that depends on the failure is suppressed while unrelated validated output may remain available.

To launch explicit production mode on PowerShell:

```powershell
$env:BRIDGE_APP_MODE = "production"
streamlit run app/app.py
```

To launch the fabricated demo on PowerShell:

```powershell
$env:BRIDGE_APP_MODE = "demo"
streamlit run app/app.py
```

On Linux or macOS, set the mode for one command instead:

```bash
BRIDGE_APP_MODE=production streamlit run app/app.py
BRIDGE_APP_MODE=demo streamlit run app/app.py
```

## Production data asset

Production requires exactly this version-controlled runtime data asset:

```text
bridges/panel.parquet
```

The application resolves the asset relative to the repository, not the caller's working directory. A clean production deployment requires **no data-generation step**, **no runtime bridge-data download**, and **no external data service**. Files such as `bridges/fake.parquet` and any data-generation or fetch scripts are not production data sources.

## Test and smoke validation

Install test tooling separately after the runtime lock:

```text
python -m pip install --require-hashes -r requirements.txt
python -m pip install --require-hashes -r requirements-test.txt
```

Run dependency/import validation and the automated test suite from the repository root:

```text
python scripts/validate_dependencies.py
python -m pytest
```

Run the bounded production smoke check on PowerShell:

```powershell
$env:BRIDGE_APP_MODE = "production"
python scripts/smoke_app.py
```

Or on Linux/macOS:

```bash
BRIDGE_APP_MODE=production python scripts/smoke_app.py
```

The smoke check uses the checked-in real panel, blocks unexpected external network access, and enforces a 120-second initial-render limit.

## Streamlit Community Cloud deployment checklist

Create or configure the app with these exact values:

- [ ] **Repository:** `https://github.com/anishchugh05/hackachusetts-2027`
- [ ] **Branch:** `person-a-app`
- [ ] **Main file path:** `app/app.py`
- [ ] **Python version (Advanced settings):** `3.11`
- [ ] **Secrets:** leave the secrets field empty; the application requires no credentials, API keys, or `.streamlit/secrets.toml`
- [ ] **Environment:** leave `BRIDGE_APP_MODE` unset so the deployment starts in production mode
- [ ] **Pre-deployment check:** run `python scripts/validate_dependencies.py`, `python -m pytest`, and `BRIDGE_APP_MODE=production python scripts/smoke_app.py` against the revision being deployed

Community Cloud installs the root `requirements.txt` runtime lock and launches `app/app.py`; no editable package installation, data preparation, bridge-data generation, or bridge-data download is required. A successful initial production render identifies Real_Dataset, Real_Model_Service, Real_Planner, and Backtest_Engine and shows the current-ranking headline and map without the demo indicator.
