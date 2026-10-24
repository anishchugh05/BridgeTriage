# Backtest Harness Requirements

Single source of truth for the backtest harness. design.md and tasks.md must match this document exactly.

## Introduction

A backtest harness to evaluate bridge inspection prioritization strategies. Given 2018 bridge inventory data (condition ratings, age, traffic), apply different selection methods to choose bridges for inspection within a fixed budget, then measure how well each method predicts 2022 condition decline using real outcomes.

## Glossary

**Bridge Rating** — Integer condition score 0–9; NaN indicates "not applicable" (recorded as "N" in source data).

**Worse** — `worse = is_safety_drop(min_2018, min_2022) = (min_2022 < min_2018) and (min_2022 <= 4)` on integer mins (each min ignores NaN). If either min is NaN, the bridge is EXCLUDED from the pool. A decline that stays at 5 or above (e.g., 7→5) is NOT worse. `is_safety_drop` is defined locally in planner/backtest.py; it is NOT imported from model.

**Candidate Pool** — Bridges remaining after all exclusions. The filter uses 2022 information (rebuilt, missing, NaN ratings) and is applied identically to every method, so no method gains an advantage.

**Score** — Predicted priority. Convention for ALL score functions: **higher score = picked first**. Score is never an evaluation weight.

**Budget** — Fraction of the candidate pool's total inspection cost: 1%, 2%, 5%, 10%, 20%.

**Evaluation Weight** — `log1p(adt)` per bridge. Unweighted counts always reported alongside weighted ones.

**COUNTY_FEE** — Fixed charge per county, deducted once per county on its first selected bridge. Same for all methods.

## Requirements

### R1: Data Input Format

#### Acceptance Criteria

1. 2018 input table columns (order agnostic): `structure_id` (str), `county` (str), `year_built` (int), `adt` (float ≥ 0), `deck`, `super`, `sub` (integer 0–9 or NaN), `inspect_cost` (float > 0). No `score` column: scores are produced by scoring functions, never read from input.

2. 2022 outcomes table columns: `structure_id` (str, must match 2018), `deck`, `super`, `sub` (integer 0–9 or NaN), `rebuilt` (bool or 0/1).

3. NaN ratings are preserved; no silent conversion to 0 or 9.

4. Load-time validation raises `ValueError` if any `inspect_cost` is NaN or <= 0, or any `adt` < 0. (NaN `adt` is NOT a load error; such bridges are excluded at pool time, R2.)

### R2: Bridge Exclusion Rules

#### Acceptance Criteria

1. Exclude a bridge if ANY of:
   - `rebuilt = True` in 2022; OR
   - `structure_id` absent from 2022; OR
   - `min_2022` is NaN; OR
   - `min_2018` is NaN; OR
   - `year_built` is NaN or <= 0; OR
   - `adt` is NaN; OR
   - its `structure_id` is duplicated within either year's table (duplicates are counted, reported, and ALL rows with a duplicated id are excluded).

2. The filter uses 2022 information and is applied identically to every method, including the oracle.

3. Report the candidate pool size AND exclusion counts by reason: missing in 2022, rebuilt, NaN min 2018, NaN min 2022, missing year_built/adt, duplicate structure_id. Never modify the input tables; filter during analysis only.

4. Bridges replaced under a NEW structure_id appear as "missing" in 2022 (old ID absent, new ID unmatched to any 2018 row) and are excluded exactly like any other missing bridge. No attempt is made to match renamed IDs.

### R3: Deterioration Detection

#### Acceptance Criteria

1. `min_2018 = min(deck_2018, super_2018, sub_2018)` ignoring NaN; same for 2022.

2. `worse = is_safety_drop(min_2018, min_2022) = (min_2022 < min_2018) and (min_2022 <= 4)`. A decline that stays at 5 or above (e.g., 7→5) is NOT worse.

3. NaN min on either side → excluded per R2, never labeled.

4. `is_safety_drop` is defined locally in planner/backtest.py; it is NOT imported from model.

### R4: Cost Accounting — one cost rule

#### Acceptance Criteria

1. ONE cost rule, implemented as `marginal_cost(county, inspect_cost, counties_seen, county_fee) -> float`: returns `inspect_cost + county_fee` if `county not in counties_seen`, else `inspect_cost`.

2. `cost_accounting(selected_ids, candidates, county_fee) -> (total_cost, n_counties)` iterates selected ids in order, accumulating via `marginal_cost`. `pick()` uses `marginal_cost` incrementally (O(n log n) overall); no other cost logic exists anywhere.

3. Total selection cost = `sum(inspect_cost) + count(unique counties selected) * county_fee`.

4. Budget denominator is the CANDIDATE POOL only: `total_cost_pool = sum(inspect_cost over pool) + count(unique pool counties) * county_fee`. Available spend = `budget_fraction * total_cost_pool`.

5. Budget comparisons are exact with relative tolerance 1e-9 (`cost <= available * (1 + 1e-9)`); no absolute epsilon.

6. Cost units: `inspect_cost`, `county_fee`, and budget arithmetic share one unit. On fake data the unit is abstract; on the real run the unit is "placeholder units" (flat `inspect_cost = 1`, `COUNTY_FEE = 5` — labeled assumptions), and the unit is stated wherever costs are printed.

### R5: pick() Function

#### Acceptance Criteria

1. Canonical signature (defined only here; design and tasks reference it):

   `pick(scores, budget, candidates, pool_cost_info, county_fee, rank_by_value_per_cost=False, tiebreak="adt", rng=None) -> list[str]`

   Returns structure_ids in selection order.

2. `scores`: dict structure_id -> float. Higher = picked first, all methods, no exceptions.

3. Validation: raise `ValueError` unless ALL scores are finite (no NaN/inf, both modes); when `rank_by_value_per_cost=True`, additionally raise `ValueError` if any score is negative; raise `ValueError` if `tiebreak="random"` and `rng is None`.

4. Ranking:
   - `rank_by_value_per_cost=False` (all baselines): key = score.
   - `rank_by_value_per_cost=True` (model scores and oracle): key = `score / inspect_cost`.

5. Tie-break:
   - `tiebreak="adt"` (default): sort by (key DESC, adt DESC), stable.
   - `tiebreak="random"`: shuffle the candidate order with `rng` FIRST, then stable sort by key only (equal keys keep shuffled order).

6. Greedy: iterate in rank order, tentatively add each bridge via `marginal_cost`, keep it if `cost_so_far + marginal <= available * (1 + 1e-9)`.

7. Candidates missing from `scores` are skipped (log warning). Scores for non-candidates are ignored.

### R6: Baseline — worst_first

#### Acceptance Criteria

1. `score = -min_2018`. Rank by score only (`rank_by_value_per_cost=False`).

2. Tie-break behavior comes from pick's `tiebreak` argument (R5.5).

### R7: Baseline — oldest_first

#### Acceptance Criteria

1. `score = -year_built`. Rank by score only.

2. Tie-break behavior comes from pick's `tiebreak` argument (R5.5).

### R8: Baseline — random_avg

#### Acceptance Criteria

1. Score = uniform random draw per candidate from `np.random.default_rng(seed)`. Rank by score only (scores are a.s. distinct; tie-break has no effect).

2. Production runs: 200 seeds (`RANDOM_SEEDS = range(200)`). Property tests: 20 seeds (`RANDOM_SEEDS_TEST = range(20)`).

3. Aggregate per budget: mean, std, min, max of recall and weighted recall; keep all per-seed rows.

### R9: Oracle

#### Acceptance Criteria

1. `score = log1p(adt)` if worse, else 0. Ranked by value per cost (`rank_by_value_per_cost=True`) — best achievable weighted recall per dollar under the budget rule.

2. Report count and weight of true decliners caught at each budget.

### R10: Evaluation Metrics

#### Acceptance Criteria

1. Headline (per method, per budget): **recall** = TP / count of worse bridges in pool; **weighted recall** = (sum log1p(adt) over selected worse) / (sum log1p(adt) over all pool worse).

2. Supporting: TP, FP, weighted TP, weighted FP, count_selected, cost_total.

3. All metrics within the candidate pool. Unweighted and weighted reported separately; never mixed; `score` never a weight.

### R11: run_backtest

#### Acceptance Criteria

1. Canonical signature (defined only here):

   `run_backtest(df_2018, df_2022, county_fee, budgets=[0.01, 0.02, 0.05, 0.10, 0.20], model_scores=None, tiebreaks=("adt",)) -> DataFrame`

2. One shared candidate pool. For each budget: worst_first, oldest_first, random_avg (200 production seeds), oracle; plus `pick(model_scores, ..., rank_by_value_per_cost=True)` when `model_scores` is given.

3. Tiebreak handling: for each entry in `tiebreaks`, deterministic baselines (worst_first, oldest_first) run once with that tiebreak. When `"random"` is included, those baselines instead run over `RANDOM_SEEDS_TEST` (20 seeds) and report the MEAN as ONE row with `tiebreak="random"`. The default call `tiebreaks=("adt",)` yields exactly one worst_first row per budget (and one row per other deterministic method).

4. One row per (method, budget [, seed, tiebreak]); random_avg also contributes aggregate rows (mean/std/min/max).

### R12: plot

#### Acceptance Criteria

1. Two headline plots (no precision–recall curves — budget fixes selection size, so PR curves are degenerate):

   a. **Budget vs. Recall** (unweighted). One line per method; oracle dashed; random_avg shaded band = mean ± BAND_K·std with **BAND_K = 3** (same constant everywhere, including the sanity test).
   
   b. **Budget vs. Weighted Recall**. Same construction.

2. Bootstrap 95% CI on each method's gap vs worst_first, computed by `bootstrap_gap_ci(...)` returning (lo, hi) per method and budget. Procedure: selections stay FIXED (selected_ids from the actual run); per replicate, draw multinomial weights over candidate-pool row indices (`rng.multinomial(n_pool, uniform)`), recompute recall numerator AND denominator as weighted sums, then the gap. Plot 1 CI uses unweighted recall; plot 2 CI uses weighted recall. 1000 replicates (`BOOTSTRAP_REPS`), fixed rng seed `BOOTSTRAP_SEED = 0`. worst_first's gap CI vs itself is exactly [0, 0].

3. The report must state: because selections are fixed, these CIs capture outcome-sampling variance only, not selection variance.

4. `output_path` is a PREFIX: plot writes `{prefix}_recall.png` and `{prefix}_weighted.png` and returns the two figures. Legend, grid, axis labels, title.

### R13: make_fake

#### Acceptance Criteria

1. Signature: `make_fake(n, seed, signal, noise) -> (df_2018, df_2022, truth)`; df_2018/df_2022 match R1; `truth` is a per-structure_id table with columns `latent` and `p_decline`.

2. Every generated bridge has `min_2018 >= 1` and at least one non-NaN 2018 rating.

3. Feature term (fixed definition):
   - `f = 2.0 * z(9 - min_2018) + 1.0 * z(age) + 0.5 * z(log1p(adt))`, where `age = 2018 - year_built` and `z(x)` standardizes over the dataset (mean 0, std 1).
   - `latent = BASE_RATE_LOGIT + signal * f + noise * N(0, 1)`.
   - `p_decline = logistic(latent)`.

4. `noise = 0`: requires `signal > 0`, else `ValueError`. `worse` = the top `round(TARGET_RATE * n)` bridges by latent, `TARGET_RATE = 0.15` — deterministic, exactly ~15% by construction.

5. `noise > 0`: `worse ~ bernoulli(p_decline)`.

6. Rating evolution (consistent with the safety-drop label by construction): worse bridges with `min_2018 >= 5` get `min_2022` uniform in 0..4; worse bridges with `min_2018 <= 4` fall by `drop ∈ 1..min(3, min_2018)`. Non-worse bridges are NEVER safety drops: `min_2018 <= 4` → `min_2022 >= min_2018`; `min_2018 >= 5` → `min_2022 >= 5` (a bridge at 5 can never be labeled non-worse by landing on 4). 2022 rating columns are regenerated consistent with the new min.

7. `BASE_RATE_LOGIT` is calibrated by simulation so that `signal=0, noise=1` yields a realized decline rate within [0.12, 0.22] (R14.8). Stored as a module constant.

8. Output includes plausible distributions: integer ratings 0–9 with some NaN, positive inspect_cost, ~14 counties, some rebuilt and some missing bridges so exclusion rules are exercised.

### R14: Correctness Properties

Property tests use `RANDOM_SEEDS_TEST = range(20)` for random_avg (not the 200 production seeds). All test functions take no arguments.

#### Acceptance Criteria

1. **Budget Compliance** — every selection: `total_cost <= available * (1 + 1e-9)`, over 100 make_fake datasets, all methods, all budgets.

2. **No Duplicate Selections** — `len(selected) == len(set(selected))` for every method and budget.

3. **Deterioration Monotonicity** — (a) exhaustively check the min step over all 11³ (deck, super, sub) triples (0–9 plus NaN) vs a reference implementation; (b) check the safety-drop/exclusion comparison over all 11×11 (min_2018, min_2022) combos including NaN against `is_safety_drop = (end < start) and (end <= 4)`; (c) run 20,000 random triple pairs vectorized through `apply_exclusion_and_label` and compare against the reference. Assert `worse` iff `(min_2022 < min_2018) and (min_2022 <= 4)`; NaN min either side → absent from pool.

4. **Oracle Upper Bound (soft aggregate, weighted only)** — over 50 make_fake seeds: `mean(oracle weighted recall) >= mean(each other method's weighted recall) - 0.01` at each budget. Per-seed violations are reported as a table; a single-seed violation does NOT fail the test. Also run on real data when available. Unweighted dominance is not claimed or tested.

5. **Exclusion Consistency** — no excluded id in any selection; pool = input − (all R2 reasons); excluded bridges not in the budget denominator.

6. **Sanity — noise=0 near oracle ("ceiling (uses truth)")** — `make_fake(signal=1, noise=0)`, model score = `truth.p_decline * log1p(adt)`, labeled **"ceiling (uses truth)"**. Procedure: run once, record the realized model/oracle weighted-recall ratio per budget, REPORT the ratios, then set the assertion bar from those realized ratios (not a fixed constant). Scores are non-negative (R5.3 satisfied).

7. **Sanity — signal=0 near random** — oracle excluded; baselines run with `tiebreak="random"`. Leg B is the gate: for each of 20 make_fake seeds (signal=0, noise=1), compute `d = method recall − that dataset's random_avg mean` (random_avg over the 20 test seeds); assert `|mean(d)| <= BAND_K * std(d) / sqrt(20)` — computed for BOTH unweighted recall AND weighted recall. BAND_K = 3, the same constant used in R12.1a.

8. **Base-Rate Calibration** — over 20 make_fake seeds with `signal=0, noise=1`, the realized decline rate = `mean(worse)` (safety-drop label) over the pool returned by `apply_exclusion_and_label` is within [0.12, 0.22]. This is measurable because the pipeline label equals the drawn label by construction (R14.9).

9. **Generator Consistency** — for every bridge in the pool, the worse label computed by the pipeline (`apply_exclusion_and_label` via `is_safety_drop`) equals the generator's drawn worse (from `truth`/generator internals), across seeds and (signal, noise) settings.

10. **Safety-Drop Parity** — compare the local `is_safety_drop` with `model.is_safety_drop` over all 11×11 inputs (0–9 plus NaN for start and end); the test is SKIPPED if `model` is not importable.

## Assumptions

1. COUNTY_FEE is a fixed constant (parameter, not computed dynamically).

2. Costs are static (2018 basis). The panel's `improvement_cost` column is an improvement cost, not an inspection cost — it is NOT used. For the real run, `inspect_cost = 1` (flat) and `COUNTY_FEE = 5`, both labeled as assumptions in "placeholder units"; with flat inspect_cost, value-per-cost ranking equals raw-score ranking, and the report must note this.

3. No bridges are added between datasets; bridges missing from 2022 are excluded.

4. NaN ratings are independent across components.

5. 2022 data is ground truth.

6. Seed ranges: `RANDOM_SEEDS = range(200)` random_avg production; `FAKE_SEEDS = range(50)` oracle property; `SANITY_SEEDS = range(20)` sanity Leg B and base-rate test; `RANDOM_SEEDS_TEST = range(20)` random_avg in tests; `BOOTSTRAP_SEED = 0`.

7. The candidate filter legitimately uses 2022 information because it defines the evaluation universe, not the prediction signal; identical for all methods.

8. Real-data adapter: `adapt_panel(path="bridges/panel.parquet", flat_inspect_cost=1.0)`. The parquet is never mutated; columns are selected by name only; year 2012 and 2016 rows are ignored. `year_built = 2018 - age`. `rebuilt` is taken directly from the panel's `rebuilt` column (A's definition: the bridge improved or was reconstructed) and is NOT recomputed. structure_id whitespace is stripped in both years.
