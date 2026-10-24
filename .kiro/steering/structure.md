# Project Structure

Maps the repo layout to the team contract (A = data, B = model, C = planner).

```
hackachusetts-2027/
├── fetch_years.py          # Part A — downloads biennial NBI MA 2012–2022, sanity report
├── requirements.txt        # shared dependencies
├── .gitignore              # ignores raw data (data/*.txt, *.csv, raw/)
│
├── data/                   # Part A — data lives here
│   └── processed/
│       └── panel.parquet   # A's output: one row per bridge per year (contract)
│
├── model/                  # Part B — decline model + scoring
│   ├── __init__.py
│   └── score.py            # fit_decline(panel, start, end), score(table, rates, asof)
│
├── planner/                # Part C — picker, baselines, backtest, proof chart
│
├── app/                    # built together last (Streamlit UI)
│
└── bridges/                # starter kit / reference data (not the live pipeline)
    ├── README.md           # describes the 2022 MA snapshot
    ├── data.csv            # reduced 2022 snapshot, 5252 rows, for eyeballing
    ├── fetch.py            # single-year fetch (Python)
    └── fetch.js            # single-year fetch (Node, alternate)
```

## Ownership map

| Area           | Owner  | Produces / provides                                  |
| -------------- | ------ | ---------------------------------------------------- |
| `fetch_years.py`, `data/` | Part A | `data/processed/panel.parquet`           |
| `model/`       | Part B | `fit_decline(...)`, `score(...)`                     |
| `planner/`     | Part C | picker, baselines, backtest, proof chart             |
| `app/`         | joint  | UI, assembled after A/B/C stabilize                  |

## Where things go

- New ingestion / cleaning logic → `fetch_years.py` and `data/` (Part A only).
- The learned model and anything that outputs the scored table → `model/`
  (Part B only). Keep the `fit_decline` / `score` signatures from the contract.
- Selection, baselines, backtest, and charts → `planner/` (Part C only).
- `bridges/` is reference material — do not build the live pipeline there.
- Keep cross-part communication to the two contract hand-offs: A's parquet
  schema and B's `score` output schema.

## Conventions

- One module's output is another's input only through the contract schemas.
  Don't reach across parts into internal helpers.
- Processed artifacts go under `data/processed/`; raw downloads stay in `data/`
  and are git-ignored.
```
