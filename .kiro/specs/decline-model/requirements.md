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
   baseline.
5. `select_model` SHALL pick by mean CV AUC, preferring the simpler model
   unless a more complex one wins by more than one standard deviation.
   Complexity order: baseline < grouped < logistic(no msi) < logistic(+ msi).
6. A horizon drop probability SHALL convert to a per-month hazard via
   `h = -ln(1 - p) / horizon_months` so inspection recency modulates `p_worse`.

### Requirement 5 — Score each bridge

#### Acceptance criteria
1. `p_worse` SHALL equal `1 - exp(-hazard × (months since insp_date + 12))`.
2. WHEN `insp_date` is missing THEN months SHALL fall back to `insp_freq_mo`,
   else a 24-month default — NEVER a long-ago assumption that inflates rank —
   and the reason SHALL include "inspection date unknown".
3. `consequence` SHALL derive from `log(1 + ADT)`, multiplied for rail or water
   crossings and given extra weight when condition ≤ 5.
4. Rail/water crossing detection SHALL use word boundaries, so "RR" does not
   match inside words like CHERRY or TERRACE, while "RR", "BMRR", "RAILROAD",
   "RAILWAY", and "MBTA" do match.
5. `insp_cost` SHALL come from the column when present, otherwise a placeholder
   based on the underwater and fracture-critical flags.
6. `score` SHALL equal `p_worse × consequence / insp_cost`.
7. Every row SHALL carry a non-empty one-line `reason`, and the reason SHALL
   note when a placeholder cost was used.
8. `score` SHALL return exactly the columns `id, p_worse, consequence,
   insp_cost, score, reason` with no NaN values.

### Requirement 6 — Runnable before Part A, and validated

#### Acceptance criteria
1. The module SHALL provide a synthetic panel generator producing the contract
   columns and types for years 2012–2022.
2. A validation harness SHALL fit on 2012→2016, test on 2014→2018, and report
   log loss and AUC against a per-condition-only baseline.
3. The test suite SHALL cover: grade monotonicity, rebuild exclusion,
   no-lookahead, clean score output, the missing-date fallback, and
   word-boundary crossing detection.
