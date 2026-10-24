# Design — Decline Model (Part B)

Documents the design as implemented in `model/`.

## Overview

```
model/
├── __init__.py     # public API
├── target.py       # is_safety_drop — the shared prediction target
├── materials.py    # NBI 043A code -> name, used in reasons
├── data.py         # load_panel — adapts Part A's panel
├── score.py        # grouped + logistic models and scoring (core)
├── fake_data.py    # synthetic contract-shaped panel
└── validate.py     # 80/20 split, metrics for grouped/logistic/baseline
```

Data flows: Part A panel → `load_panel` → (`fit_decline` | `fit_logistic`) →
model chosen by held-out AUC → `score` → ranked table → Part C. The target
`is_safety_drop` is shared with Part C's backtest.

## Components

### `DeclineRates` (dataclass)

Holds learned monthly hazards at three granularities plus a global fallback:

- fine:   `("mat", material, age_band, grade)`
- mid:    `("mc", material, grade)`
- coarse: `("c", grade)`
- `global_hazard` when nothing matches.

`hazard_for(material, age_band, condition)` resolves finest-first. Because the
fine level is densified over all grades 0–9 per (material, age band) during the
fit, any trained (material, age band) resolves at the fine level, so adjacent
grades are never compared across different fallback levels — this is what keeps
`p_worse` monotone in grade at scoring time.

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
3. Store probabilities over the fit horizon; `score` converts to a monthly
   hazard: `h = -ln(1 - p) / horizon_months`.

### `fit_logistic(panel, start, end, train_ids=None)` — logistic model

Same pairs/target. Builds a design matrix of condition one-hot (grades 0–9),
material one-hot (training vocabulary), age, log1p(ADT), insp_freq_mo,
fracture_critical, underwater; fits sklearn `LogisticRegression`. Exposes the
same `interval_prob(rec)` / `horizon_months` interface as the grouped model.

### Model selection

`validate.select_model` fits both on the 80% split, compares held-out AUC, and
returns the better family refit on all bridges. `score` works with whichever
model it's handed.

### `score(table, rates, asof)`

Per bridge:
- **p_worse** — `1 - exp(-hazard × months)`, where `months = months_since_insp
  + 12`. Missing `insp_date` falls back to `insp_freq_mo` (else 24 months),
  never a long-ago assumption.
- **consequence** — `log1p(ADT)`, × rail (1.5) / water (1.3) multipliers via
  word-boundary regex, × 1.4 when condition ≤ 5; floored at 0.1.
- **insp_cost** — column value, else a placeholder by underwater /
  fracture-critical flags.
- **score** — `p_worse × consequence / insp_cost`.
- **reason** — one line noting material/age/condition, risk window, crossings,
  poor-condition weight, unknown inspection date, and placeholder cost.

Returns exactly the six contract columns, NaN-free.

## Key design decisions

- **Monotonicity via densify-then-isotonic, in the data-correct direction.**
  Raw empirical rates wobble from sparse cells; the invariant is enforced in
  the fit. The direction is set by the real MA data: P(drop) rises with grade
  (grade 3 ~0%, grade 9 ~93% over 2012→2016), so the clamp makes P(drop)
  non-decreasing in grade, not non-increasing.
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
