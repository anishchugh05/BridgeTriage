# Design — Decline Model (Part B)

Documents the design as implemented in `model/`.

## Overview

```
model/
├── __init__.py     # public API
├── target.py       # is_safety_drop — the shared prediction target
├── materials.py    # NBI 043A code -> name, used in reasons
├── flight_cost.py  # labeled drone cost/duration estimate
├── data.py         # load_panel — adapts Part A's panel
├── score.py        # grouped + logistic models and scoring (core)
├── fake_data.py    # synthetic contract-shaped panel
├── validate.py     # grouped 5-fold CV, selection, OOF calibration
└── run.py          # writes 2018/2022 score parquet files
```

Data flows: Part A panel → `load_panel` → (`fit_decline` | `fit_logistic`) →
model chosen by held-out AUC → `score` → ranked table → Part C. The target
`is_safety_drop` is shared with Part C's backtest.

## Components

### `DeclineRates` (dataclass)

Holds fitted safety-drop probabilities at three granularities plus a global
fallback:

- fine: `("mat", material, age_band, grade)`
- mid: `("mc", material, grade)`
- coarse: `("c", grade)`

`prob_for(material, age_band, condition)` resolves finest-first. Grouped
scoring converts the selected horizon probability to a monthly hazard; no
monotonic grade constraint is imposed for the safety-drop target.

### The target — `target.is_safety_drop`

A pair is a safety drop when `end ≤ 4 and start ≥ 5` (crosses into poor) or
`start ≤ 4 and end ≤ start − 1` (already poor, got worse). High-grade slips
that stay above 4 are not drops. `fit_decline`, `fit_logistic`, validate, and
Part C all label from this one function (and its vectorized twin).

### `fit_decline(panel, start, end)` — grouped model

1. `_build_pairs`: filter to `start`/`end` years (no lookahead), pair by `id`,
   drop null-condition and rebuilt/improved pairs, label with `is_safety_drop`.
2. Estimate P(drop) per group (material × age band × condition) with shrinkage
   toward the coarser parent (`SHRINKAGE_K = 20`). No monotonicity clamp.
3. Store probabilities over the fit horizon; grouped scoring converts them to
   a monthly hazard: `h = -ln(1 - p) / horizon_months`. Logistic scoring instead
   scales its probability directly to 12 months.

### `fit_logistic(panel, start, end, train_ids=None)` — logistic model

Same pairs/target. Builds a design matrix of condition one-hot (grades 0–9),
material one-hot (training vocabulary), age, log1p(ADT), insp_freq_mo,
fracture_critical, underwater; fits sklearn `LogisticRegression`. Exposes the
same `interval_prob(rec)` / `horizon_months` interface as the grouped model.

### months_since_insp feature

Months from `insp_date` to the reference date (April 1 of the row's year for a
pair, or the `asof` date at scoring), capped at 60. Falls back to
`insp_freq_mo` only when `insp_date` is missing, with the fallback flagged in
the reason. Toggleable in the logistic model so CV can measure its value.

### Model selection (grouped 5-fold CV)

`validate.cross_validate` runs 5-fold CV grouped by bridge id (a bridge is
never split across folds) and reports mean ± std AUC/log loss for four models:
grouped, logistic(no msi), logistic(+ msi), baseline. `select_model` walks the
complexity order and adopts a more complex model only when its mean AUC beats
the current pick by more than the current pick's std — otherwise the simpler
model wins. The chosen family is refit on all bridges. `score` works with
whichever model it's handed, passing `asof` when the model accepts it.

### `flight_cost(row)` — drone inspection estimate

Returns `(insp_cost_usd, flight_minutes, reason_fragment)`. Cost is
`(1500 + 8 × length_m) × access_factor + 2 × one_way_km`; access factor is 1.4
for water/rail. Duration is `20 + 0.5 × length_m + 2 × one_way_km × 1.2`, so
travel is round-trip. Travel uses the nearest Boston, Worcester, or Springfield
depot; missing coordinates use 60 km. The function accepts future meter-length
columns and otherwise uses material-level planning medians. Every fallback and
all outputs are labeled as estimates.

### `score(table, rates, asof)`

Per bridge:
- **p_worse (logistic/baseline)** — scale the fitted horizon probability
  directly to 12 months: `1 - (1 - p)^(12 / horizon_months)`. Do not reapply
  inspection-age hazard to logistic predictions.
- **p_worse (grouped)** — retain the constant-hazard path over
  `months_since_insp + 12`; grouped rates do not contain recency as a feature.
- **consequence** — `log1p(ADT)`, × rail (1.5) / water (1.3) multipliers via
  word-boundary regex, × 1.4 when condition ≤ 5; floored at 0.1.
- **insp_cost / flight_minutes** — a real positive `insp_cost` when supplied;
  otherwise `flight_cost(row)` estimates drone cost from base, bridge length,
  water/rail access, and nearest-depot travel. Duration includes round-trip
  travel. Missing length/coordinates use explicitly labeled estimates.
- **expected_drops_per_10k** — `p_worse × (consequence / median consequence)
  / insp_cost × 10000`.
- **score** — exactly equal to `expected_drops_per_10k` for Part C.
- **reason** — one line noting material/age/condition, forecast/risk window,
  crossings, inspection recency, poor-condition weight, and a labeled drone
  estimate fragment.

Returns exactly eight columns, NaN-free; rows with null `worst_rating` are
not applicable and are omitted.

## Key design decisions

- **Safety-drop target over raw grade slips.** The model predicts crossings
  into condition ≤ 4, or further deterioration once already there. High-grade
  slips that remain above 4 are intentionally negative examples; no monotonic
  grade constraint is imposed.
- **Missing inspection date is moderate, not stale.** Treating an unknown date
  as 10 years old would wrongly float data-poor bridges to the top; one
  required cycle is the neutral assumption.
- **Word-boundary crossing match.** `RR\b` matches standalone "RR" and
  suffixes like "BMRR" but not CHERRY/TERRACE; water tokens are whole-word.

## Error handling

- `end <= start` → `ValueError`.
- Empty paired sample → flat global prior (0.1), still returns valid rates.
- Missing/NaN fields degrade gracefully to fallbacks; output is always NaN-free.

## Testing strategy

`tests/test_model.py` asserts the six behaviors in Requirement 6.
`model/validate.py` provides the quantitative check (log loss, AUC vs the
per-condition baseline) on either real or synthetic data.
