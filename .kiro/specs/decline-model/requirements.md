# Requirements — Decline Model (Part B)

This spec documents the decline model and scoring that already live in
`model/`. It is written after the fact to capture the behavior that is built
and tested, so Parts A and C can rely on a stable interface.

## Introduction

Part B turns Part A's bridge panel into a ranked risk table. For each bridge it
estimates the probability the structure is worse now than its last recorded
rating (`p_worse`), weighs that by how much a failure would matter
(`consequence`), and divides by the cost of an extra inspection to produce a
risk-reduction-per-dollar `score`. The planner (Part C) consumes this table.

## Requirements

### Requirement 1 — Learn decline from history without lookahead

**User story:** As the planner, I want decline rates learned only from data up
to the pick year, so a backtest can't cheat by peeking at the future.

#### Acceptance criteria
1. WHEN `fit_decline(panel, start, end)` is called THEN it SHALL read rows only
   from years `start` and `end`.
2. WHEN rows exist for years after `end` THEN the learned rates SHALL be
   identical to a panel without those rows.
3. WHEN `end <= start` THEN `fit_decline` SHALL raise `ValueError`.

### Requirement 2 — Pair bridges and exclude rebuilds

**User story:** As a modeler, I want to measure true deterioration, not repairs.

#### Acceptance criteria
1. WHEN a bridge appears in both years THEN it SHALL be paired by `id`.
2. WHEN the end-year `rebuilt` flag is true (or, absent that column,
   `year_recon` changed — 0 meaning never) THEN that pair SHALL be excluded.
3. WHEN a bridge's condition improved between the two years THEN it SHALL be
   excluded (treated as repaired/rebuilt).

### Requirement 2b — Shared safety-drop target

**User story:** As Part C, I want the exact same "drop" definition the model
trains on, so the backtest measures the same event.

#### Acceptance criteria
1. A single function `is_safety_drop(start, end)` SHALL define the target and
   be importable by Part C.
2. A pair SHALL be a drop WHEN `end <= 4 AND start >= 5`, OR
   `start <= 4 AND end <= start - 1`.
3. A high-grade slip that stays above 4 (e.g. 9→7, 8→5) SHALL NOT be a drop.
4. A pair with a missing start or end grade SHALL NOT be a drop.

### Requirement 3 — Condition from NBI components, honoring "N"

#### Acceptance criteria
1. WHEN deck/super/sub are present THEN condition SHALL be their minimum.
2. WHEN deck/super/sub are all NaN (NBI "N") THEN condition SHALL fall back to
   culvert.
3. WHEN all four are NaN THEN condition SHALL be NaN and the bridge SHALL use
   the global fallback rate.
4. NBI "N" (NaN) SHALL NEVER be treated as a rating of 0.

### Requirement 4 — Models, months_since_insp feature, and CV selection

#### Acceptance criteria
1. A **grouped** model (`fit_decline`) SHALL estimate P(safety drop) by
   material × age band × condition, each small group shrunk toward
   material × condition, then condition alone, then a global rate. No
   monotonicity clamp is applied.
2. A **logistic** model (`fit_logistic`) SHALL predict the safety drop from
   condition (one-hot), material (one-hot), age, log ADT, insp_freq_mo,
   fracture_critical, underwater, and optionally `months_since_insp`.
3. `months_since_insp` SHALL be the months from `insp_date` to April 1 of the
   row's year (the NBI snapshot date), capped at 60, falling back to
   `insp_freq_mo` only when `insp_date` is missing. `score` SHALL compute it to
   the `asof` date and show "last inspected N mo ago" in the reason (flagging
   the fallback when the date is missing).
4. `validate.py` SHALL run 5-fold cross-validation **grouped by bridge id** and
   report mean ± std of AUC and log loss for: grouped, logistic without
   `months_since_insp`, logistic with `months_since_insp`, and the per-condition
   baseline. It SHALL also print a five-quantile-bin calibration table for the
   selected model using out-of-fold predictions (mean predicted vs actual).
5. `select_model` SHALL pick by mean CV AUC, preferring the simpler model
   unless a more complex one wins by more than one standard deviation.
   Complexity order: baseline < grouped < logistic(no msi) < logistic(+ msi).
6. A grouped-model horizon probability SHALL convert to a per-month hazard
   via `h = -ln(1 - p) / horizon_months` and apply over inspection recency +
   12 months. A logistic-model probability SHALL instead scale directly to a
   12-month horizon via `1 - (1-p)^(12/horizon_months)`, without hazard
   re-application.

### Requirement 5 — Score each bridge

#### Acceptance criteria
1. For a logistic model, `p_worse` SHALL equal
   `1 - (1-p)^(12/horizon_months)` with no recency hazard re-application.
   For the grouped model, it SHALL retain the hazard path over
   `months_since_insp + 12`.
2. WHEN `insp_date` is missing THEN months SHALL fall back to `insp_freq_mo`,
   else a 24-month default, and the reason SHALL include "inspection date
   unknown".
3. `consequence` SHALL derive from `log(1 + ADT)`, multiplied for rail or water
   crossings and given extra weight when condition ≤ 5.
4. Rail/water crossing detection SHALL use word boundaries, so "RR" does not
   match inside words like CHERRY or TERRACE, while "RR", "BMRR", "RAILROAD",
   "RAILWAY", and "MBTA" do match.
5. `insp_cost` SHALL come from the column when a positive real value is
   present. Otherwise it SHALL use `flight_cost(row)`: $1,500 base + $8 per
   meter, ×1.4 for water/rail access, plus $2 per one-way kilometer from the
   nearest Boston/Worcester/Springfield depot. Missing coordinates use 60 km;
   missing length uses a material median and is labeled "length estimated".
6. `flight_minutes` SHALL equal `20 + 0.5 × length_m + 2 × travel_km × 1.2`
   (round-trip travel) and be labeled as an estimate in the reason.
7. `consequence_norm` SHALL equal consequence divided by its cohort median;
   `expected_drops_per_10k` SHALL equal
   `p_worse × consequence_norm / insp_cost × 10000`.
8. `score` SHALL equal `expected_drops_per_10k` exactly.
9. Every row SHALL carry a non-empty one-line `reason` with the drone cost,
   duration, nearest-depot travel, and any estimated inputs.
10. `score` SHALL return exactly `id, p_worse, consequence, insp_cost,
   flight_minutes, expected_drops_per_10k, score, reason`, with no NaNs; null
   `worst_rating` rows SHALL be omitted.

### Requirement 6 — Runnable before Part A, validated, and batch-scored

#### Acceptance criteria
1. The module SHALL provide a synthetic panel generator producing the contract
   columns and types for years 2012–2022.
2. Validation SHALL use five folds grouped by bridge id over 2012→2016 and
   report model metrics plus selected-model calibration.
3. `model/run.py` SHALL select/refit on 2012→2016, score 2018 and 2022 as of
   April 1 of each year, and write `data/processed/scores_2018.parquet` and
   `scores_2022.parquet`.
4. The test suite SHALL cover the target, rebuild/no-lookahead rules, both
   horizon-conversion paths, normalized score formula, calibration bins,
   missing-date/crossing handling, and batch output.
