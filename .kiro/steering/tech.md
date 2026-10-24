# Tech Stack

Derived from the existing repo (`requirements.txt`, `fetch_years.py`,
`bridges/`, git state).

## Language & runtime

- **Python** is the primary language (observed: Python 3.14 locally; code uses
  only stdlib + the libs below, so any 3.10+ should work).
- A small amount of **Node.js** exists only as an alternate fetcher
  (`bridges/fetch.js`); it is not required for the pipeline.

## Dependencies (`requirements.txt`)

- `pandas`, `numpy` — data wrangling and the panel table.
- `scikit-learn` — the decline model in Part B.
- `ortools` — constrained selection in the planner (Part C).
- `matplotlib` — the proof chart.
- `streamlit` — the `app/` UI, built last.

Install:

```bash
pip install -r requirements.txt
```

## Data source

- FHWA National Bridge Inventory, delimited Massachusetts files:
  `https://www.fhwa.dot.gov/bridge/nbi/<year>/delimited/MA<yy>.txt`
- `fetch_years.py` downloads the biennial years 2012, 2014, 2016, 2018, 2020,
  2022 into `data/` and matches NBI columns by **item-number suffix** (e.g.
  `_008` for structure id) so minor naming differences across years don't break
  ingestion.
- Raw files are read with `dtype=str, encoding="latin-1"` — NBI is fixed-schema
  latin-1 text; keep everything as strings until explicitly parsed so `"N"` and
  blanks survive.
- `bridges/data.csv` is a reduced 2022 snapshot (5252 rows) for quick
  inspection only — the real pipeline pulls fresh yearly files.

## Conventions

- Match NBI columns by suffix, never by hard-coded full column name.
- Read raw NBI files as strings, but parse them into a typed panel in Part A
  (numeric 0–9 ratings with `"N"` → `NaN`, datetime `insp_date`, boolean flags,
  decimal-degree lat/lon). Don't ship string ratings downstream.
- Downloaded raw data is git-ignored (see `.gitignore`): `data/*.txt`,
  `data/*.csv`, `raw/`.

## Commands

```bash
# fetch biennial NBI MA files (2012–2022) + sanity report
python fetch_years.py

# refresh a single 2022 snapshot
python bridges/fetch.py      # or: node bridges/fetch.js

# run the app (once app/ exists)
streamlit run app/<entrypoint>.py
```
