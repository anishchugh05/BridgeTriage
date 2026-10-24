# BridgeTriage — Project Documentation

> Comprehensive documentation for the `final` branch. This is the agreed final
> version. It describes what the system actually does, how the pieces connect,
> how to run it, and where its boundaries are. Companion docs:
> [architecture.md](architecture.md), [model_card.md](model_card.md),
> [stories.md](stories.md), and [../model/README.md](../model/README.md).

## 1. What BridgeTriage is

BridgeTriage turns a versioned FHWA National Bridge Inventory (NBI) panel for
Massachusetts into a budget-aware inspection-prioritization workflow. Routine
inspections are fixed by federal rules; the choice BridgeTriage helps make is
narrow and concrete:

> Given a limited budget for *extra* inspections, which bridges should get that
> scrutiny to buy the most safety value per dollar?

It is a decision-support tool, not a safety authority. It does not issue
findings, replace statutory inspection cycles, or produce vendor quotes. Every
probability, cost, and flight time it emits is a **planning estimate**, clearly
labeled as such.

## 2. The pipeline at a glance

```
bridges/panel.parquet        (canonical NBI panel, Part A)
        │  model/data.py  load_panel()  — adapt names, preserve missingness
        ▼
model/  target → fit → score   (model/validate.py selects, model/run.py writes)
        │  data/processed/scores_2018.parquet, scores_2022.parquet
        ▼
planner/ exclusion → pick → backtest   (planner/run_export.py)
        │  results/backtest.csv, results/summary.json
        ▼
app/     Streamlit prototype (app/app.py)  — exploration UI
```

Two commands produce all analytical evidence:

```bash
python -m model.run          # writes data/processed/scores_{2018,2022}.parquet
python -m planner.run_export # writes results/backtest.csv + results/summary.json
```

## 3. Data layer (`bridges/`, `model/data.py`)

**`bridges/panel.parquet`** — the canonical runtime source: 20,757 rows ×
21 columns, 5,506 unique `structure_id`s, one row per `(structure_id, year)`
for years **2012, 2016, 2018, 2022**. The schema is fixed and owned by
`bridges/README.md`:

```
structure_id, year, county, material, age, adt, deck, super, sub,
worst_rating, insp_freq_months, fracture_critical, underwater, year_recon,
rebuilt, lat, lon, carries, crosses, improvement_cost, insp_date
```

Critical semantics (enforced downstream, never worked around):

- `worst_rating` = min(deck, super, sub); **null means not applicable**, not a
  condition of zero. ~300 null-rating rows per year (culverts, spillways).
- `rebuilt` marks reconstruction/condition improvement — excluded from
  deterioration analysis (counts: 2012=0, 2016=518, 2018=325, 2022=394).
- `year_recon == 0` means never rebuilt.
- `improvement_cost` is item-094 **repair cost** and is prohibited as an
  inspection cost.

**`model/data.py` `load_panel(path=None)`** reads the panel and returns an
adapted copy (it never mutates `bridges/`): renames `structure_id→id`,
`insp_freq_months→insp_freq_mo`; derives `year_built = year − age`; maps
`year_recon==0 → NaN`; parses `insp_date`; deliberately does **not** map
`improvement_cost → insp_cost`. It also exports the source/scored boundary used
by the planner and app: `SOURCE_COLUMNS`, the `SourceDataError` family, and
`validate_scored_table` / `reconcile_scored_table`. Key constant:
`FOUR_YEAR_PAIRS = [(2012, 2016), (2018, 2022)]` — the two-year 2016→2018 gap is
never used for fitting or backtest.

## 4. Model layer (`model/`)

### The target — `model/target.py`

One shared definition, used by both the model and the backtest, so training and
evaluation agree on what a "drop" is. A **safety drop** between two years is:

- crossing into poor: `end ≤ 4 AND start ≥ 5`, **or**
- already poor and worsening: `start ≤ 4 AND end ≤ start − 1`.

Slips that stay above 4 (e.g. 9→7) are **not** drops. A missing grade on either
side is not a drop. (`is_safety_drop`, `is_safety_drop_series`.)

### Scoring — `model/score.py`

`score(table, rates, asof)` produces one row per eligible bridge. Rows with a
null condition are skipped (not applicable). The output has exactly **11
columns**:

| column | meaning |
|---|---|
| `id` | NBI structure number |
| `p_worse` | probability of a safety drop over a 12-month window |
| `consequence` | `log1p(ADT)` × rail (1.5) / water (1.3) / poor-condition (1.4) multipliers, floored at 0.1 |
| `insp_cost` | real inspection cost if provided, else the drone-flight estimate |
| `flight_minutes` | estimated drone flight time |
| `expected_drops_per_10k` | `p_worse × (consequence / cohort median) / insp_cost × 10000` |
| `score` | equal to `expected_drops_per_10k` — the single number the planner ranks on |
| `voi` | value of information, `p_worse × (1 − p_worse) × consequence_norm` |
| `decision` | `fix_track`, `fly`, or `monitor` |
| `reason` | one-line technical explanation |
| `reason_human` | plain-language explanation for a non-technical reader |

The scoring formula, stated once:

```
consequence_norm = consequence / median(consequence over the scored cohort)
score = expected_drops_per_10k = p_worse × consequence_norm / insp_cost × 10,000
```

`p_worse` is produced two ways depending on the selected model:
- **Logistic / baseline:** the fitted horizon probability is rescaled directly
  to 12 months — `1 − (1 − p_horizon)^(12 / horizon_months)`.
- **Grouped (`DeclineRates`):** a constant-hazard recency path over
  `months_since_insp + 12` months (the grouped model has no recency feature).

### The models

- **`fit_decline(panel, start, end)` → `DeclineRates`** — grouped empirical
  P(drop) by material × age-band × condition, with shrinkage (`SHRINKAGE_K=20`)
  toward coarser parent groups. Rejects any interval ≠ 4 years.
- **`fit_logistic(panel, start, end, use_months_since_insp=True)` →
  `LogisticModel`** — sklearn logistic regression. Features: condition one-hot
  (grades 0–9), material one-hot, age, log ADT, inspection frequency,
  fracture-critical, underwater, and optionally months-since-inspection.
- **`fit_condition_baseline(...)` → `ConditionBaseline`** — condition-only drop
  rate, the comparison floor.

### Cost, consequence, explainability

- **`model/flight_cost.py`** — `flight_cost(row)` returns
  `(insp_cost_usd, flight_minutes, reason_fragment)`. Cost =
  `(1500 + 8 × length_m) × access_factor + 2 × travel_km` (access factor 1.4 for
  rail/water); minutes = `20 + 0.5 × length_m + 2 × travel_km × 1.2` (round-trip
  travel). Length uses a panel column if present, else a material-median
  labeled "length estimated". Travel is haversine to the nearest depot
  (Boston / Worcester / Springfield), default 60 km when coordinates are
  missing. All outputs are labeled estimates.
- **`model/voi.py`** — value-of-information post-pass. `add_voi(scored)` adds
  `voi` and `decision`. Per cohort: `fix_track` when `p_worse ≥ 0.25`
  (likely already dropped — a flight tells us nothing new); `fly` for the top
  20% of VOI among sub-threshold bridges (most uncertainty resolved per dollar);
  `monitor` otherwise. `simulate_flight(scored_row, outcome)` resolves `p_worse`
  to 1.0 (confirmed) or 0.0 (refuted) and stamps `insp_date` to today, so the
  app can re-rank after a flight.
- **`model/materials.py`** — `material_name(code)` maps NBI 043A codes
  ("3"→"Steel") for readable reasons.

### Selection & validation — `model/validate.py`

Five-fold cross-validation **grouped by bridge id** over the 2012→2016 training
horizon (`CV_SEED=1234`). It scores four candidates — baseline, grouped,
logistic without months-since-inspection, logistic with it — reporting AUC and
log loss per fold plus a 5-bin out-of-fold calibration table. `select_model`
applies a **simplicity-aware rule**: prefer the simpler model unless a more
complex one beats the current best mean AUC by more than one CV-fold standard
deviation. The currently selected model is **logistic without
months-since-inspection**; full methodology, coefficients, and limitations are
in [model_card.md](model_card.md).

### Runners

- **`python -m model.run`** — selects + fits on 2012→2016, scores 2018 and 2022
  as of April 1 of each year, writes `data/processed/scores_{2018,2022}.parquet`
  (verified 4,891 and 4,927 rows × 11 columns).
- **`python -m model.stories`** — writes [stories.md](stories.md): top-5 picks
  that actually dropped by 2022 vs top-5 worst-rating-first picks.
- **`model/fake_data.py`** — synthetic panel for tests / `validate.py` fallback.
  Never used in published results.

## 5. Planner layer (`planner/`)

### Backtest engine — `planner/backtest.py`

- `adapt_panel(...)` → `(df_2018, df_2022)` with a flat placeholder
  `inspect_cost`.
- `apply_exclusion_and_label(...)` builds **one shared candidate pool** used by
  every method, so no method gains an advantage from the filter. Exclusions, in
  order: `duplicate_id`, `missing_2022`, `rebuilt`, `nan_min_2018`,
  `nan_min_2022`, `missing_yearbuilt_adt`. Each kept bridge is labeled
  `worse = is_safety_drop(min_2018, min_2022)`.
- Cost rule (the only one): `marginal_cost` / `cost_accounting` — a county
  mobilization fee is charged **once per selected county**.
- `pick(scores, budget, candidates, pool_cost_info, county_fee, …)` — greedy
  budgeted selection; `rank_by_value_per_cost` divides score by inspection cost
  (model and oracle). Validates that scores are finite (and non-negative for
  value-per-cost).
- Baselines: `score_worst_first`, `score_oldest_first`, `score_random(seed)`,
  and `score_oracle(outcomes)` (an upper bound that sees realized outcomes —
  **never a deployable policy**).
- `run_backtest(...)` runs all methods across budgets `[0.01, 0.02, 0.05,
  0.10, 0.20]`; `bootstrap_gap_ci(...)` gives 95% CIs on each method's recall
  gap versus worst-first (outcome-sampling variance only).

### Run scripts

| command | produces |
|---|---|
| `python -m planner.run_export` | **`results/backtest.csv`** + **`results/summary.json`** (the published evidence) |
| `python -m planner.run_model` | console recall pivots + `/tmp` plots (model vs baselines) |
| `python -m planner.run_real` | baseline-only run on the real panel, console + `/tmp` plots |
| `python -m planner.run_ci` | bootstrap gap CIs to console |

`run_export.load_model_scores()` strictly validates the 2018 score file
(requires `id` + `score`, finite unique ids, full candidate coverage) before
the backtest runs.

## 6. Evidence artifacts (`results/`, `data/processed/`)

Current checked-in results:

**`results/summary.json`** — candidate pool `4420`, decliners `137`,
exclusions `{rebuilt: 393, nan_min_2018: 310, missing_2022: 91, nan_min_2022: 1,
duplicate_id: 0, missing_yearbuilt_adt: 0}`.

**`results/backtest.csv`** — recall by method and budget (fraction of the pool's
actual decliners caught). Selected rows:

| budget | worst_first | oldest_first | model | oracle (ceiling) |
|---|---|---|---|---|
| 0.05 | 0.066 | 0.036 | 0.088 | 1.000 |
| 0.10 | 0.255 | 0.066 | 0.212 | 1.000 |
| 0.20 | 0.467 | 0.161 | 0.423 | 1.000 |

Read honestly: the model beats both baselines at small budgets (0.01–0.05) and
beats oldest-first at every budget, but at 0.10–0.20 worst-first edges the model
on this pool, and the recall-gap bootstrap CIs vs worst-first straddle zero. The
oracle is a diagnostic ceiling, not a policy. The takeaway is "competitive and
honestly measured," not "dominant." See [stories.md](stories.md) for the
per-bridge narrative where the model's top picks did decline and worst-first's
did not.

**`data/processed/scores_2018.parquet`** (4,891 × 11) and
**`scores_2022.parquet`** (4,927 × 11) carry the 11 score columns above.

## 7. Application layer (`app/`) — prototype status

`app/` contains `app.py, config.py, data.py, errors.py, services.py, stubs.py,
style.py`. Run with:

```bash
python -m streamlit run app/app.py
```

**Honest status:** `app/app.py` is a **prototype** that renders a pydeck map and
tabs (Map, Budget, Backtest, Bridge detail, Hazards, Export, Methods). It tries
to import the real model/planner and, on any failure, falls back to
`app/stubs.py` (fabricated output) with a visible "stub model" pill. In the
current `final` branch that fallback triggers, because:

- `app.py` imports `from planner.pick import pick, backtest`, but `pick` lives
  in `planner/backtest.py` — there is no `planner/pick` module; and
- it calls `score(cur)` with one argument, while the real
  `model.score.score` needs `(table, rates, asof)`.

So the UI currently shows **stub data**, not the real pipeline. The production
building blocks exist but are not wired into `app.py`: `config.py` (exact
`BRIDGE_APP_MODE` production/demo parsing), `errors.py` (public-safe error
registry), and `services.py` (deterministic validated caching). The README and
[architecture.md](architecture.md) document the target composition that removes
the catch-all fallback.

Two separate loaders exist by design: `model/data.py:load_panel` (renames to
`id`, for the model) and `app/data.py:load_panel` (keeps `structure_id`, falls
back to `bridges/fake.parquet` when the real panel is absent).

The authoritative way to regenerate analytical artifacts today is the two CLI
commands in §2 — not the UI.

## 8. Running everything

Setup (Python **3.11** per `.python-version`):

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements.txt
python -m pip install --require-hashes -r requirements-test.txt
```

Analytical pipeline:

```bash
python -m model.run           # scores_2018.parquet, scores_2022.parquet
python -m planner.run_export  # results/backtest.csv, results/summary.json
python -m model.stories       # docs/stories.md  (optional)
python -m model.validate      # CV / selection / calibration report (optional)
```

Prototype UI (stub fallback, see §7):

```bash
python -m streamlit run app/app.py
```

## 9. Tests & quality gate

```bash
python -m pytest -q                    # behavioral / model / planner / contract tests
python scripts/validate_dependencies.py # dependency + interpreter + import gate
```

`pytest.ini` collects from `planner` and `tests` (`test_*.py` and
`properties.py`). Coverage includes the model (`tests/test_model.py`), VOI
(`tests/test_voi.py`), flight cost (`tests/test_flight_cost.py`), planner
invariants (`planner/properties.py`), and boundary/cache/dependency units under
`tests/unit/`. The README records an 81-passing baseline.

**Known release blocker:** `scripts/validate_dependencies.py` fails closed
because `planner/backtest.py` imports `matplotlib` without an approved
dependency mapping. That is the gate doing its job; resolve it (declare the
dependency or move the plotting import) before calling a release green.

## 10. Trust rules (non-negotiable)

1. Real FHWA Massachusetts data only in analysis; fabricated rows live only in
   explicit test/demo fixtures.
2. No lookahead: fit on 2012→2016, rank as of 2018, only then check 2022. The
   two-year 2016→2018 gap is never used for four-year fitting.
3. Missing ≠ bad: null condition ratings are not applicable, never zeros;
   rebuilt and non-comparable rows are excluded consistently for every method.
4. Costs are estimates: `improvement_cost` is never used as inspection price;
   flight cost and duration are disclosed planning estimates.
5. Every score is explainable: technical `reason` and plain-language
   `reason_human` accompany every row.
6. Artifacts are reproducible: paths, score years, and budgets are fixed in
   source control.
7. Failures are explicit: typed errors, strict schemas, exact runtime-mode
   parsing, and a dependency gate — surfaced, not hidden.

## 11. Known limitations

- Massachusetts-specific; do not transfer without refitting.
- The safety-drop event is rare (137 decliners in a 4,420-bridge backtest pool),
  so model-vs-baseline margins are modest and CIs are wide.
- The panel currently has no structure-length column, so drone-cost length is a
  material-median estimate (reasons say "length estimated").
- Backtest bootstrap CIs capture outcome-sampling variance for fixed
  selections, not model-selection or deployment variance.
- The Streamlit UI is a prototype on stub data; a successful model/backtest run
  does not certify it.
