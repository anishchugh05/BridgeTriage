# Massachusetts Bridge Inspection Planner

This repository contains the Streamlit decision-support application for prioritizing additional Massachusetts bridge inspections. Production mode uses the version-controlled real Massachusetts bridge panel, the repository's real model service, deterministic planner, and measured backtest engine. Demo mode is an explicitly selected, fabricated, visibly labeled alternative; it is never a production fallback.

## Clean-checkout local setup

Run all commands from the repository root. The supported runtime is **Python 3.11**, as declared in `.python-version`.

```text
git clone https://github.com/<github-owner>/<repository>.git
cd <repository>
git checkout <branch>
```

Create and activate an isolated Python 3.11 environment.

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

`python --version` must report Python 3.11.x. Install the exact, hashed runtime lock:

```text
python -m pip install --require-hashes -r requirements.txt
```

Start the application from the repository root:

```text
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

To launch explicit production mode:

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

If a production data, schema, model, planner, or backtest component fails, the application remains in production mode, keeps demo mode deactivated, and displays a user-safe error for the failed component. It never substitutes fabricated data or demo calculations. Output that depends on the failed component is suppressed, while unrelated validated output may remain available.

## Production data asset

Production requires exactly this version-controlled runtime data asset:

```text
bridges/panel.parquet
```

The application resolves the asset relative to the repository, not the caller's working directory. A clean production deployment requires **no data-generation step**, **no runtime bridge-data download**, and **no external data service**. Files such as `bridges/fake.parquet` and any data-generation or fetch scripts are not production data sources.

## Test and smoke validation

Install test tooling separately, after the runtime lock:

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

The smoke check uses the checked-in real panel and requires no external network request.

## Streamlit Community Cloud deployment

Create the app in Streamlit Community Cloud with these settings:

| Setting | Value |
| --- | --- |
| Repository | `<github-owner>/<repository>` |
| Branch | `<branch>` |
| Main file path | `app/app.py` |
| Python version (Advanced settings) | `3.11` |

Leave `BRIDGE_APP_MODE` unset so the cloud deployment starts in production mode. Community Cloud installs the root `requirements.txt` runtime lock and launches `app/app.py`; no editable package installation or data preparation is required.

This application requires **no secrets, credentials, API keys, or `.streamlit/secrets.toml` entries**. Leave the Community Cloud secrets field empty. The required real dataset is already version-controlled at `bridges/panel.parquet`, and production startup performs no generation or network download.

Before deploying, run the test and production smoke commands above against the revision selected by `<branch>`. A successful initial production render identifies Real_Dataset, Real_Model_Service, Real_Planner, and Backtest_Engine and shows the current-ranking headline and map without the demo indicator.
