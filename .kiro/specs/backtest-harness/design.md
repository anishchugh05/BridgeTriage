# Backtest Harness Design

Implements requirements.md (the single source of truth). Where behavior is specified there, this document only shows how.

## System Overview

```
2018 data ─┐
           ├→ load_and_validate → apply_exclusion_and_label → (candidates, outcomes, pool_cost_info)
2022 data ─┘                                                            │
                         scoring (worst/oldest/random/oracle/model) ←───┘
                                   │
                pick() [marginal_cost, tiebreak adt|random, budget-bounded]
                                   │
                          evaluate_metrics → run_backtest → plot (+bootstrap_gap_ci)

make_fake(n, seed, signal, noise) → (df_2018, df_2022, truth)  [test data source]
adapt_nbi(MA18.txt, MA22.txt) → (df_2018, df_2022)             [real data source]
```

## Data Model

### candidates (DataFrame, produced by exclusion pass)
Indexed by `structure_id` (str). Columns exactly:
```
county (str), inspect_cost (float), adt (float), year_built (int), min_2018 (float)
```

### outcomes (dict)
```
structure_id -> {min_2018: float, min_2022: float, worse: bool, adt: float}
```

### pool_cost_info
```
{'total_cost_pool': float,   # sum(inspect_cost over pool) + n_pool_counties * county_fee
 'unique_counties': set[str],
 'exclusion_counts': {missing_2022: int, rebuilt: int, nan_min_2018: int,
                      nan_min_2022: int, missing_yearbuilt_adt: int,
                      duplicate_id: int}}
```

### EvaluationMetrics
```
tp, fp: int;  weighted_tp, weighted_fp: float
recall, weighted_recall: float        # HEADLINE
count_selected, count_pool, count_worse_in_pool: int;  cost_total: float
```

## Core Algorithms

### Algorithm 1: load_and_validate(df_2018, df_2022)
1. Validate 2018 columns: structure_id, county, year_built, adt, deck, super, sub, inspect_cost (no `score`).
2. Validate 2022 columns: structure_id, deck, super, sub, rebuilt.
3. Ratings → float; "N"/empty → NaN; remaining values must be integers 0–9.
4. structure_id → str in both. Build 2022 index for lookup.
5. ValueError if any inspect_cost is NaN or <= 0, or any adt < 0. (NaN adt passes load; excluded at pool time.)
6. Return (df_2018, df_2022, index).

### Algorithm 2: apply_exclusion_and_label(df_2018, df_2022, index, county_fee)
Filter uses 2022 info; identical for all methods (documented in any report).
```
0. Count duplicated structure_ids within each year; exclude ALL rows whose
   id is duplicated (record count under 'duplicate_id').
For each 2018 row:
  skip + count reason: missing in 2022 | rebuilt | min_2018 NaN |
  min_2022 NaN | year_built NaN or <= 0 or adt NaN ('missing_yearbuilt_adt')
  worse = (min_2022 <= min_2018 - 1)
Build candidates (indexed by structure_id; cols county, inspect_cost, adt,
year_built, min_2018), outcomes dict, and
pool_cost_info = {'total_cost_pool': sum(inspect_cost) + n_unique_counties*county_fee,
                  'unique_counties': {...},
                  'exclusion_counts': {...}}     # POOL ONLY, never all bridges
Bridges replaced under a new ID fall out as 'missing in 2022' (R2.4).
```

### Algorithm 3: Cost — one rule, two functions
```
def marginal_cost(county, inspect_cost, counties_seen, county_fee):
    return inspect_cost + (county_fee if county not in counties_seen else 0)

def cost_accounting(selected_ids, candidates, county_fee):
    seen = set(); total = 0
    for id in selected_ids:                      # selection order
        total += marginal_cost(candidates.county[id], candidates.inspect_cost[id],
                               seen, county_fee)
        seen.add(candidates.county[id])
    return total, len(seen)
```
pick() uses marginal_cost incrementally; NOTHING else computes costs.
Units: abstract on fake data; "placeholder units" on the real run
(inspect_cost = 1, COUNTY_FEE = 5, labeled assumptions); unit stated wherever costs print.

### Algorithm 4: pick
Canonical signature (requirements R5.1):
`pick(scores, budget, candidates, pool_cost_info, county_fee, rank_by_value_per_cost=False, tiebreak="adt", rng=None) -> list[str]`
```
1. Validate: all scores finite, else ValueError.
   rank_by_value_per_cost and any score < 0 → ValueError.
   tiebreak="random" and rng is None → ValueError.
2. available = budget * pool_cost_info['total_cost_pool']
3. key = score/inspect_cost if rank_by_value_per_cost else score
4. Order:
   tiebreak="adt":     sort by (key DESC, adt DESC), stable
   tiebreak="random":  permute candidate order with rng FIRST,
                       then stable sort by key DESC only
5. selected = []; seen = set(); spent = 0
   for id in ordered ids:                               # O(n log n)
       m = marginal_cost(county[id], inspect_cost[id], seen, county_fee)
       if spent + m <= available * (1 + REL_TOL):        # REL_TOL = 1e-9
           selected.append(id); spent += m; seen.add(county[id])
6. return selected
```

### Algorithm 5: Scoring (all: higher = picked first)
```
worst_first:  score = -min_2018     (rank by score; tiebreak via pick arg)
oldest_first: score = -year_built   (rank by score; tiebreak via pick arg)
random:       score = rng.random()  (rank by score; rng = default_rng(seed);
                                     200 seeds production, 20 in tests)
oracle:       score = log1p(adt) if worse else 0   (rank by value per cost)
model:        caller dict via pick(..., rank_by_value_per_cost=True)
```

### Algorithm 6: evaluate_metrics(selected_ids, outcomes, adt)
```
worse_ids = {id: o.worse};  S = set(selected_ids)
tp = |S ∩ worse_ids|; fp = |S| − tp
weighted_tp/fp = sums of log1p(adt); total_worse_weight likewise
recall = tp / |worse_ids|            (0.0 if none)
weighted_recall = weighted_tp / total_worse_weight   (0.0 if 0)
```

### Algorithm 7: run_backtest
Canonical signature (requirements R11.1):
`run_backtest(df_2018, df_2022, county_fee, budgets=[0.01, 0.02, 0.05, 0.10, 0.20], model_scores=None, tiebreaks=("adt",)) -> DataFrame`
```
candidates, outcomes, pool = apply_exclusion_and_label(...)
for budget in budgets:
    for tb in tiebreaks:
        if tb == "adt":
            worst_first, oldest_first: 1 run each (tiebreak="adt")
        if tb == "random":
            worst_first, oldest_first: run over RANDOM_SEEDS_TEST (20 seeds);
            report the MEAN as ONE row with tiebreak="random"
    random_avg (200 production seeds; per-seed rows + mean/std/min/max rows)
    oracle (1 run)
    model via pick(..., rank_by_value_per_cost=True) if model_scores given
Default call tiebreaks=("adt",) → exactly one worst_first row per budget.
Columns: method, budget, seed, tiebreak, tp, fp, weighted_tp, weighted_fp,
         recall, weighted_recall, count_selected, cost_total, is_aggregate, stat
```

### Algorithm 8: plot(results_df, output_path=None) + bootstrap_gap_ci(...)
Two figures only (PR curves are degenerate under fixed budgets):
- Fig 1: budget vs recall; Fig 2: budget vs weighted recall.
- Method lines; oracle dashed; random_avg band = mean ± BAND_K·std (BAND_K = 3,
  same constant as sanity property 7).
- `bootstrap_gap_ci(selections, outcomes, adt, metric) -> {method: {budget: (lo, hi)}}`:
  selections FIXED; per replicate w = rng.multinomial(n_pool, [1/n_pool]*n_pool)
  over pool row indices; numerator and denominator as weighted sums;
  gap vs worst_first. Fig 1 CI uses unweighted recall, Fig 2 uses weighted.
  BOOTSTRAP_REPS = 1000, rng = default_rng(BOOTSTRAP_SEED = 0).
  worst_first's gap CI vs itself is exactly (0, 0).
- output_path is a PREFIX: write {prefix}_recall.png and {prefix}_weighted.png;
  return the two figures.
- Report states: fixed selections → CIs capture outcome-sampling variance only.

### Algorithm 9: make_fake(n, seed, signal, noise) -> (df_2018, df_2022, truth)
```
rng = default_rng(seed)
Generate n bridges: ~14 counties, year_built 1850–2021, adt ~ lognormal,
ratings int 0–9 (~15% NaN per column, at least one non-NaN per bridge),
inspect_cost > 0. Enforce min_2018 >= 1 for every bridge.
f      = 2.0*z(9 - min_2018) + 1.0*z(age) + 0.5*z(log1p(adt))
         # age = 2018 - year_built; z standardizes over the dataset
latent = BASE_RATE_LOGIT + signal*f + noise*rng.standard_normal(n)
p_decline = logistic(latent)                    # → truth
if noise == 0:
    if signal <= 0: raise ValueError
    k = round(TARGET_RATE * n)                  # TARGET_RATE = 0.15
    worse = top-k bridges by latent             # deterministic, exactly ~15%
else:
    worse ~ bernoulli(p_decline)
Rating evolution:
  worse:     drop ∈ 1..min(3, min_2018); min_2022 = min_2018 - drop   (>= 0)
  non-worse: min_2022 >= min_2018
  regenerate 2022 rating columns consistent with the new min
Sprinkle rebuilt flags + remove rows from df_2022 (exercises exclusions).
Return (df_2018, df_2022, truth[structure_id -> latent, p_decline])
```
BASE_RATE_LOGIT calibration: simulate signal=0, noise=1 over a seed sweep;
choose the constant so the realized pool decline rate (mean of worse over the
pool from Algorithm 2) lands in [0.12, 0.22]; store as module constant.
Verified by R14.8.

### Algorithm 10: adapt_nbi(path_2018, path_2022, flat_inspect_cost, county_fee)
```
1. Read raw files (dtype=str, latin-1). Select columns BY NAME ONLY
   (no suffix matching): STRUCTURE_NUMBER_008, COUNTY_CODE_003, ADT_029,
   DECK_COND_058, SUPERSTRUCTURE_COND_059, SUBSTRUCTURE_COND_060,
   YEAR_BUILT_027, YEAR_RECONSTRUCTED_106; plus LOWEST_RATING in the
   2022 file if present (diagnostic only).
2. Cleaning ORDER (fixed): strip whitespace from selected columns →
   treat "" and "N" as NaN → convert rating/year/ADT columns to numeric.
3. County codes: strip, then zero-pad to 3 digits in BOTH years
   (no name mapping).
4. structure_id: strip whitespace in BOTH years.
5. Print BOTH unmatched counts: 2018 IDs absent from 2022, and
   2022 IDs absent from 2018.
6. If LOWEST_RATING exists in the 2022 file: print how many rows differ
   from the computed min(deck, super, sub) ignoring NaN (diagnostic of
   our min computation; no rows are changed).
7. 2018 table → R1 contract; inspect_cost = flat placeholder
   (ASSUMPTION, "placeholder units", labeled in code and report).
   Item _094 exists but is an improvement cost — NOT used.
8. 2022 table: rebuilt = (YEAR_RECONSTRUCTED_106 in 2022) >
   (YEAR_RECONSTRUCTED_106 in 2018), blank/0 treated as 0.
   Print count of rebuilt=True bridges.
9. Validate against R1; return.
Real runs use flat inspect_cost = 1, COUNTY_FEE = 5 ("placeholder units",
labeled assumptions); report notes: with flat cost, value-per-cost
ranking == raw-score ranking. Bridges replaced under a new ID appear as
missing and are excluded like any missing bridge (requirements R2.4).
```

## Module Structure

`planner/__init__.py` — package marker.

`planner/backtest.py`: load_and_validate, apply_exclusion_and_label,
marginal_cost, cost_accounting, pick, score_worst_first, score_oldest_first,
score_random, score_oracle, evaluate_metrics, make_fake, run_backtest,
bootstrap_gap_ci, plot, adapt_nbi.

`planner/properties.py` (all test functions take NO arguments):
test_budget_compliance, test_no_duplicates, test_deterioration_monotonicity,
test_oracle_upper_bound, test_exclusion_consistency,
test_sanity_noise_zero_near_oracle, test_sanity_signal_zero_near_random,
test_base_rate_calibration, test_generator_consistency.

## Module Constants

- `BAND_K = 3` — plot band + sanity property 7 (one constant).
- `TARGET_RATE = 0.15` — make_fake noise=0 top-k share.
- `BASE_RATE_LOGIT` — calibrated by simulation (realized pool rate in [0.12, 0.22]).
- `REL_TOL = 1e-9` — budget comparison tolerance (relative; no absolute epsilon).
- `BOOTSTRAP_REPS = 1000`, `BOOTSTRAP_SEED = 0`.
- `RANDOM_SEEDS = range(200)` — production random_avg.
- `RANDOM_SEEDS_TEST = range(20)` — random_avg inside property tests + tiebreak="random" baselines.
- `FAKE_SEEDS = range(50)` — oracle property.
- `SANITY_SEEDS = range(20)` — sanity Leg B + base-rate test.

## Implementation Notes

1. NaN: `pd.isna()`; ratings are integers 0–9 when present; "N" → NaN.
2. Score convention: higher = picked first, everywhere; no per-method sort direction.
3. Value-per-cost mode: finite + non-negative scores enforced by ValueError; model scores and oracle only.
4. Budget denominator: candidate pool only, county_fee once per unique pool county.
5. Tie-break: "adt" = (key DESC, adt DESC) stable; "random" = rng-shuffle then stable key sort; "random" with rng=None → ValueError.
6. Seeds: `np.random.default_rng(s)` per run; ranges reserved per Module Constants.
7. Metrics: unweighted and weighted always separate; `score` never a weight.
8. Exclusion uses 2022 info, identical for all methods — noted in any report; per-reason counts reported.
9. Bootstrap CIs: fixed selections → outcome-sampling variance only — noted in any report.
10. Test functions in properties.py take no arguments; all parameters (seeds, counts) come from Module Constants.

## Correctness Properties (executable; mirror R14)

1. Budget compliance: cost ≤ available·(1+1e-9), 100 fake datasets.
2. No duplicates in any selection.
3. Deterioration monotonicity: 11³ triples vs reference; 11×11 (min18, min22) combos incl. NaN; 20,000 random triple pairs vectorized through apply_exclusion_and_label.
4. Oracle soft aggregate: mean over 50 seeds, oracle weighted recall ≥ each method's mean − 0.01 per budget; per-seed violations reported as a table, no single-seed failure.
5. Exclusion consistency (all R2 reasons).
6. Sanity noise=0 "ceiling (uses truth)": run once, record realized model/oracle ratio per budget, report, set bar from realized ratios.
7. Sanity signal=0 (oracle excluded, baselines tiebreak="random"): Leg B only — |mean(d)| ≤ BAND_K·std(d)/√20 over 20 seeds, for BOTH unweighted and weighted recall.
8. Base-rate calibration: mean(worse) over pool ∈ [0.12, 0.22], 20 seeds, signal=0, noise=1.
9. Generator consistency: pipeline-computed worse == generator's drawn worse for every pool bridge.

---
**End of Design**
