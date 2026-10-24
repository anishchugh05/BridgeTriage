# Team Contract: parallel ownership

Three people build in parallel. The contract below is the interface between
them — as long as everyone honors their outputs and signatures, the parts snap
together without coordination. `app/` is built together later, after A/B/C land.

## Part A — data (owns `fetch_years.py`, `data/`)

Responsible for turning raw FHWA NBI files into one clean panel.

**Panel years:** 2012, 2016, 2018, 2022. The matched four-year intervals are
2012→2016 and 2018→2022; never fit the two-year 2016→2018 gap.

**Output:** `data/processed/panel.parquet` — one row per bridge per year.

**Columns (exact names, types, and parsing):**

| column             | type                | meaning / parsing                                             |
| ------------------ | ------------------- | ------------------------------------------------------------- |
| `id`               | string              | NBI structure id (item 8), stable across years                |
| `year`             | int                 | NBI publication year of the row                               |
| `county`           | string              | county name                                                   |
| `year_built`       | int (nullable)      | original construction year                                    |
| `year_recon`       | int (nullable)      | reconstruction year (may be missing)                          |
| `material`         | string              | structure material / kind                                     |
| `deck`             | float, 0–9          | deck condition (item 58); NBI `"N"` → `NaN` (not applicable)  |
| `super`            | float, 0–9          | superstructure condition (item 59); `"N"` → `NaN`             |
| `sub`              | float, 0–9          | substructure condition (item 60); `"N"` → `NaN`               |
| `culvert`          | float, 0–9          | culvert condition (item 62); `"N"` → `NaN`                    |
| `adt`              | int (nullable)      | average daily traffic                                         |
| `insp_date`        | datetime            | last inspection; parsed from NBI MMYY format                  |
| `insp_freq_mo`     | int (nullable)      | required inspection frequency, months                         |
| `fracture_critical`| bool                | fracture-critical flag                                        |
| `underwater`       | bool                | underwater-inspection flag                                    |
| `lat`              | float               | latitude, decimal degrees                                     |
| `lon`              | float               | longitude, decimal degrees (negative in Massachusetts)        |
| `crosses`          | string              | what the bridge crosses (for rail/water consequence)          |
| `insp_cost`        | float (optional)    | estimated extra-inspection cost; may be absent — see Part B   |

Type and parsing rules that bind Part A:
- `deck` / `super` / `sub` / `culvert` are numeric 0–9. NBI `"N"` converts to
  `NaN`, and `NaN` means **not applicable** — never 0, never a low rating.
- `insp_date` is a real datetime parsed from NBI's `MMYY` format.
- `lat` / `lon` are decimal degrees; longitude is negative (western hemisphere).
- `fracture_critical` and `underwater` are booleans.
- `insp_cost` is optional. If Part A cannot supply a real inspection price,
  leave it absent / `NaN`; Part B uses its labeled drone-flight estimate.

General rules that bind Part A:
- Raw NBI files stay strings only until Part A parses them. The **panel is
  typed** per the table above — do not ship string ratings downstream.
- Never drop or synthesize bridges. Missing stays missing (`NaN`).
- One row per `(id, year)` — no duplicates.

## Part B — model (owns `model/`)

Consumes A's panel. Provides two functions:

```python
def fit_decline(panel, start, end):
    """Learn decline rates from panel rows in [start, end]. MUST NOT read any
    year > end — that would leak the future into a backtest pick."""
    ...  # returns `rates`

def score(table, rates, asof):
    """Score each bridge as of `asof` using learned `rates`.
    Returns a table with columns:
      id, p_worse, consequence, insp_cost, flight_minutes,
      expected_drops_per_10k, score, reason
    where consequence_norm = consequence / median(consequence),
    expected_drops_per_10k = p_worse * consequence_norm / insp_cost * 10000,
    score equals expected_drops_per_10k, and `reason` is one line."""
    ...
```

Rules that bind Part B:
- `fit_decline` honors the no-lookahead rule via its `end` argument.
- `score` always emits a non-empty `reason` for every row.
- When no real `insp_cost` is present, `score` uses the explicitly labeled
  drone-flight estimate from `model.flight_cost`, and returns its estimated
  `flight_minutes`. `improvement_cost` is never used as inspection cost.

## Part C — planner (owns `planner/`)

Consumes B's scored table. Provides:
- **picker** — selects bridges in descending score order until a **dollar
  budget** is used up (summing `insp_cost`). Also supports a **fixed count N**
  mode (top N by score). Budget mode is the default framing; N is the simpler
  fixed-count variant.
- **baselines** — worst-rating-first and oldest-first pickers.
- **backtest** — fit the decline model on **2012→2016**, **pick with 2018**,
  then check which bridges actually had a safety drop by 2022. Compare planner
  vs each baseline.
- **proof chart** — visualize the comparison.

Rules that bind Part C:
- Backtest fits on the four-year 2012→2016 interval and picks as of 2018.
  Never fit the two-year 2016→2018 gap, and never leak 2022 into the pick.

## Shared boundaries

- A→B hand-off is the parquet schema above. B→C hand-off is the `score` output
  schema. Changing either column list is a contract change — announce it.
- `app/` (Streamlit) is joint work once A, B, C are stable.
