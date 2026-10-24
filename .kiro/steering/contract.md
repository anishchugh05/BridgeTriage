# Team Contract: parallel ownership

Three people build in parallel. The contract below is the interface between
them — as long as everyone honors their outputs and signatures, the parts snap
together without coordination. `app/` is built together later, after A/B/C land.

## Part A — data (owns `fetch_years.py`, `data/`)

Responsible for turning raw FHWA NBI files into one clean panel.

**Fetches NBI years:** 2012, 2014, 2016, 2018, 2020, 2022 (every other year).
These biennial snapshots give the model enough history to learn decline without
leaking into the pick year.

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
- `insp_cost` is optional. If Part A cannot estimate it, leave it absent /
  `NaN`; Part B substitutes a placeholder cost (see below).

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
      id, p_worse, consequence, insp_cost, score, reason
    where score = p_worse * consequence / insp_cost and `reason` is a
    one-line human explanation."""
    ...
```

Rules that bind Part B:
- `fit_decline` honors the no-lookahead rule via its `end` argument.
- `score` always emits a non-empty `reason` for every row.
- When `insp_cost` is missing (`NaN` or absent), `score` substitutes a
  placeholder cost so the ranking still computes — and the `reason` notes that
  the cost was estimated with a placeholder.

## Part C — planner (owns `planner/`)

Consumes B's scored table. Provides:
- **picker** — selects bridges in descending score order until a **dollar
  budget** is used up (summing `insp_cost`). Also supports a **fixed count N**
  mode (top N by score). Budget mode is the default framing; N is the simpler
  fixed-count variant.
- **baselines** — worst-rating-first and oldest-first pickers.
- **backtest** — fit the decline model on **2012→2016** (or **2014→2018**) and
  **pick with 2018**, then check which bridges actually declined by 2022.
  Compare planner vs each baseline.
- **proof chart** — visualize the comparison.

Rules that bind Part C:
- Backtest fits on years up to the pick year only — e.g.
  `fit_decline(panel, 2012, 2016)` then pick as of 2018, or
  `fit_decline(panel, 2014, 2018)` then pick as of 2018. Nothing after the pick
  year may leak in.

## Shared boundaries

- A→B hand-off is the parquet schema above. B→C hand-off is the `score` output
  schema. Changing either column list is a contract change — announce it.
- `app/` (Streamlit) is joint work once A, B, C are stable.
