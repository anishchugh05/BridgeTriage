# `model/` — BridgeTriage decline model (Part B)

This package turns Part A's cleaned Massachusetts NBI panel into a ranked,
explainable inspection plan. It predicts which bridges are most likely to have
quietly slipped into poor condition, weighs that against traffic/consequence and
an estimated drone-flight cost, and emits a decision for each bridge. It is the
"B" in the A (data) → B (model) → C (planner) split.

For the formal target definition, validation numbers, and limitations, see
[`docs/model_card.md`](../docs/model_card.md). For the picks-vs-reality proof,
see [`docs/stories.md`](../docs/stories.md).

## What it produces

`score(table, model, asof)` returns one row per bridge with:

| column | meaning |
|---|---|
| `id` | NBI structure number |
| `p_worse` | estimated probability the bridge has had a safety drop |
| `consequence` | traffic/crossing-weighted importance |
| `insp_cost` | real inspection cost if provided, else estimated drone-flight cost |
| `flight_minutes` | estimated drone flight time |
| `expected_drops_per_10k` | `p_worse × (consequence / median) / insp_cost × 10000` |
| `score` | equal to `expected_drops_per_10k` (the one number Part C ranks on) |
| `voi` | value of information, `p_worse × (1 − p_worse) × consequence_norm` |
| `decision` | `fix_track`, `fly`, or `monitor` |
| `reason` | one-line technical explanation |
| `reason_human` | plain-language explanation for a non-technical reader |

All costs, times, and probabilities are **planning estimates, not official
measurements**.

## Decisions

- **fix_track** — `p_worse ≥ 0.25`. Likely already in poor condition; a flight
  wouldn't tell us anything new, so queue a fix.
- **fly** — uncertain (`p_worse < 0.25`) and VOI in the cohort's top 20%. A
  flight resolves the most uncertainty per dollar.
- **monitor** — everything else. Watch, don't spend a flight yet.

`simulate_flight(scored_row, outcome)` returns the bridge's updated
`(p_worse, insp_date)` after an inspection confirms (`→ 1.0`) or refutes
(`→ 0.0`) the drop, with the inspection date refreshed to the flight day — so the
app can show the plan changing after a flight.

## Modules

| file | role |
|---|---|
| `data.py` | `load_panel()` adapts Part A's `bridges/panel.parquet` |
| `target.py` | `is_safety_drop(start, end)` — the shared prediction target |
| `score.py` | `fit_decline`, `fit_logistic`, `score`, condition/consequence/reason logic |
| `voi.py` | value-of-information, decision labels, `simulate_flight` |
| `flight_cost.py` | `flight_cost(row)` drone cost/time estimate |
| `materials.py` | NBI 043A material code → name |
| `validate.py` | 5-fold grouped CV, model selection, calibration |
| `run.py` | fits the selected model and writes `data/processed/scores_<year>.parquet` |
| `stories.py` | builds `docs/stories.md` (picks vs. 2022 reality) |
| `fake_data.py` | synthetic panel so the model runs before real data lands |

## How to run

All commands use the repo's virtualenv at `.venv`.

```bash
# fit + score 2018 and 2022, writing data/processed/scores_<year>.parquet
.venv/bin/python -m model.run

# cross-validation, model selection, calibration, coefficients
.venv/bin/python -m model.validate

# regenerate the picks-vs-reality proof doc
.venv/bin/python -m model.stories

# tests
.venv/bin/python -m pytest tests/test_model.py tests/test_voi.py tests/test_flight_cost.py -q
```

`model.run` and `model.stories` depend on `bridges/panel.parquet` (owned by
Part A). `model.validate` falls back to synthetic data if the panel is absent.

## Model selection (summary)

Four candidates are compared by 5-fold CV grouped by bridge id: a per-condition
baseline, a grouped material × age-band × condition model, and logistic
regression with and without `months_since_insp`. The selector prefers the
simpler model unless a more complex one beats it by more than one std of CV AUC.
On the current panel it selects **logistic without `months_since_insp`**
(CV AUC 0.819). Full numbers are in the model card.

## Rules this package honors

- Real FHWA NBI Massachusetts data only; never invent bridges or ratings.
- Never fit on years after the pick year (no lookahead); never fit the two-year
  2016→2018 gap.
- NBI null `worst_rating` means not-applicable, never a rating of 0 — such
  bridges are excluded.
- Every scored row carries a non-empty reason; costs are labeled estimates.
- Nothing under `bridges/` is modified by this package.
