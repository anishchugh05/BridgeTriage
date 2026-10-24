# Backtest Harness — Implementation Tasks

Canonical signatures (defined in requirements.md; referenced here, not redefined):
- `pick(scores, budget, candidates, pool_cost_info, county_fee, rank_by_value_per_cost=False, tiebreak="adt", rng=None) -> list[str]` (R5.1)
- `run_backtest(df_2018, df_2022, county_fee, budgets=[0.01, 0.02, 0.05, 0.10, 0.20], model_scores=None, tiebreaks=("adt",)) -> DataFrame` (R11.1)
- `make_fake(n, seed, signal, noise) -> (df_2018, df_2022, truth)` (R13.1)

Property tests use `RANDOM_SEEDS_TEST = range(20)` for random_avg (not 200). All test functions take no arguments.

## Tasks

- [x] 1. Set up planner package and data loading
  - Create `planner/__init__.py` (package marker) and `planner/backtest.py`: module docstring, module constants (BAND_K = 3, TARGET_RATE = 0.15, BASE_RATE_LOGIT placeholder, REL_TOL = 1e-9, BOOTSTRAP_REPS = 1000, BOOTSTRAP_SEED = 0, RANDOM_SEEDS = range(200), RANDOM_SEEDS_TEST = range(20), FAKE_SEEDS = range(50), SANITY_SEEDS = range(20)), imports (pandas, numpy, matplotlib)
  - Implement `load_and_validate` per Algorithm 1 (R1): column checks (no `score` column), "N"→NaN, integer 0–9 rating validation, structure_id str coercion, 2022 lookup index, ValueError on inspect_cost NaN/<=0 or adt < 0
  - _Requirements: R1_
  - Demo: `python -c "import pandas as pd; from planner.backtest import load_and_validate; a=pd.DataFrame({'structure_id':['x'],'county':['c'],'year_built':[1950],'adt':[100.0],'deck':['N'],'super':[5],'sub':[6],'inspect_cost':[1.0]}); b=pd.DataFrame({'structure_id':['x'],'deck':[4],'super':[5],'sub':[6],'rebuilt':[0]}); d18,d22,idx=load_and_validate(a,b); assert pd.isna(d18.loc[0,'deck']) and 'x' in idx; bad=a.assign(inspect_cost=[0.0]); import traceback
try: load_and_validate(bad,b); assert False
except ValueError: print('deck NaN ok, index ok, inspect_cost=0 rejected')"`

- [x] 2. Implement make_fake synthetic data generator
  - `make_fake` per Algorithm 9 (R13): returns `(df_2018, df_2022, truth)`; min_2018 ≥ 1 and ≥ 1 non-NaN rating per bridge; `f = 2.0*z(9-min_2018) + 1.0*z(age) + 0.5*z(log1p(adt))`; `latent = BASE_RATE_LOGIT + signal*f + noise*N(0,1)`; noise=0 requires signal>0 (else ValueError) and worse = top round(TARGET_RATE*n) by latent with TARGET_RATE = 0.15; evolution per safety-drop label: worse start≥5 → end in 0..4, worse start≤4 → fall 1..min(3, start); non-worse never safety drops (start≤4 → end≥start; start≥5 → end≥5, so a 5 never lands on 4)
  - Calibrate BASE_RATE_LOGIT by simulation (realized pool rate in [0.12, 0.22] at signal=0, noise=1); store as constant
  - _Requirements: R13_
  - Demo: `python -c "from planner.backtest import make_fake, TARGET_RATE; d18,d22,t=make_fake(2000,1,1.0,0.0); import numpy as np; k=round(TARGET_RATE*2000); assert (t.latent.nlargest(k).index.isin(d18.structure_id)).all(); mins=np.nanmin(np.vstack([d18['deck'].fillna(9),d18['super'].fillna(9),d18['sub'].fillna(9)]),axis=0); assert (mins>=1).all(); print('noise=0 top-k deterministic, min_2018>=1 ok'); import traceback
try: make_fake(100,2,0.0,0.0); assert False
except ValueError: print('signal=0,noise=0 rejected')"`

- [x] 3. Implement exclusion and labeling pass
  - `apply_exclusion_and_label` per Algorithm 2 (R1.4, R2, R3, R4.4): skip rebuilt / missing in 2022 / NaN min either side / year_built NaN or <=0 / adt NaN / duplicate structure_id (all rows of a duplicated id excluded, count reported); `worse = is_safety_drop(min_2018, min_2022) = (min_2022 < min_2018) and (min_2022 <= 4)` (defined locally in backtest.py, not imported from model); candidates DataFrame indexed by structure_id with columns county, inspect_cost, adt, year_built, min_2018; pool_cost_info with total_cost_pool (POOL only) and per-reason exclusion_counts
  - _Requirements: R2, R3, R4.4_
  - Demo: `python -c "import pandas as pd; from planner.backtest import *; a=pd.DataFrame({'structure_id':['a','b','c','d','e','f'],'county':['x','x','y','y','y','y'],'year_built':[1950,1950,1950,1950,None,1950],'adt':[10.0]*6,'deck':[5,5,5,5,5,7],'super':[5,5,5,5,5,7],'sub':[5,5,5,5,5,7],'inspect_cost':[1.0]*6}); b=pd.DataFrame({'structure_id':['a','b','c','e','f'],'deck':[4,5,5,5,5],'super':[5,5,5,5,5],'sub':[5,5,5,5,5],'rebuilt':[0,1,0,0,0]}); c,o,p=apply_exclusion_and_label(*load_and_validate(a,b),5.0); assert set(c.index)=={'a','c','f'} and o['a']['worse'] and not o['c']['worse'] and not o['f']['worse'] and p['total_cost_pool']==3+2*5 and p['exclusion_counts']['rebuilt']==1 and p['exclusion_counts']['missing_2022']==1; print('a worse (5->4 safety drop); f NOT worse (7->5 stays above 4); pool cost 13')"`

- [x] 4. Implement cost rule (marginal_cost + cost_accounting)
  - `marginal_cost(county, inspect_cost, counties_seen, county_fee)` and `cost_accounting(selected_ids, candidates, county_fee)` per Algorithm 3 (R4): fee once per county, selection order; the ONLY cost logic in the codebase
  - _Requirements: R4_
  - Demo: `python -c "import pandas as pd; from planner.backtest import cost_accounting; c=pd.DataFrame({'county':['x','x','y'],'inspect_cost':[1.0,1.0,1.0]},index=['a','b','c']); t,n=cost_accounting(['a','b','c'],c,5.0); assert t==13.0 and n==2; print('cost 13.0 (1+5 +1 +1+5), 2 counties')"`

- [x] 5. Implement pick()
  - Canonical signature (R5.1) per Algorithm 4: ValueError unless all scores finite (both modes), non-negative (value-per-cost mode), and rng given when tiebreak="random"; key = score or score/inspect_cost; tiebreak="adt" sorts (key DESC, adt DESC) stable, tiebreak="random" shuffles with rng then stable key sort; greedy via `marginal_cost`, budget check `cost <= available*(1+REL_TOL)`; O(n log n)
  - _Requirements: R4, R5_
  - Demo: `python -c "import pandas as pd, numpy as np; from planner.backtest import pick; c=pd.DataFrame({'county':['x','x','y'],'inspect_cost':[1.0,1.0,1.0],'adt':[100.0,50.0,10.0],'year_built':[1950,1960,1970],'min_2018':[3.0,5.0,7.0]},index=['a','b','c']); p={'total_cost_pool':13.0,'unique_counties':{'x','y'}}; s=pick({'a':3.0,'b':2.0,'c':1.0},0.9,c,p,5.0); assert s==['a','b']; print('adt tiebreak:',s,'cost 7.0')
try: pick({'a':1.0},0.5,c,p,5.0,tiebreak='random'); assert False
except ValueError: print('random tiebreak without rng rejected')"`

- [ ] 6. Implement the four scoring functions
  - `score_worst_first = -min_2018`, `score_oldest_first = -year_built`, `score_random(candidates, seed)` via `np.random.default_rng`, `score_oracle = worse*log1p(adt)` (value-per-cost); tie-break behavior comes from pick's `tiebreak` argument
  - _Requirements: R6, R7, R8, R9_
  - Demo: `python -c "import pandas as pd; from planner.backtest import score_worst_first; c=pd.DataFrame({'county':['x','x'],'inspect_cost':[1.0,1.0],'adt':[1.0,1.0],'year_built':[1950,1960],'min_2018':[3.0,7.0]},index=['a','b']); s=score_worst_first(c); assert s['a']==-3.0 and s['a']>s['b']; print('worst_first: a(-3) picked before b(-7)')"`

- [ ] 7. Implement evaluation metrics
  - `evaluate_metrics(selected_ids, outcomes, adt_dict)` per Algorithm 6 (R10): TP/FP, weighted TP/FP via log1p(adt), headline recall and weighted_recall
  - _Requirements: R10_
  - Demo: `python -c "from planner.backtest import evaluate_metrics, is_safety_drop; o={'a':{'worse':True,'min_2018':5,'min_2022':4},'b':{'worse':False,'min_2018':5,'min_2022':5},'c':{'worse':True,'min_2018':6,'min_2022':4},'d':{'worse':False,'min_2018':7,'min_2022':5}}; assert is_safety_drop(5,4) and not is_safety_drop(7,5); m=evaluate_metrics(['a','b'],o,{'a':10.0,'b':10.0,'c':10.0,'d':10.0}); assert m.tp==1 and m.fp==1 and abs(m.recall-0.5)<1e-12 and abs(m.weighted_recall-0.5)<1e-12; print('a worse (5->4), d not (7->5); recall 0.5, weighted recall 0.5')"`

- [ ] 8. Implement run_backtest orchestrator
  - `run_backtest` with the canonical signature (R11.1) per Algorithm 7: shared pool; per `tiebreaks` entry — "adt" runs deterministic baselines once, "random" runs them over RANDOM_SEEDS_TEST and reports the MEAN as ONE row with tiebreak="random"; default call yields exactly one worst_first row per budget; random_avg 200 production seeds; oracle; model via pick(..., rank_by_value_per_cost=True) when model_scores given
  - _Requirements: R11_
  - Demo: `python -c "from planner.backtest import *; d18,d22,t=make_fake(400,2,0.5,0.5); r=run_backtest(d18,d22,5.0); w=r[(r.method=='worst_first')&(r.budget==0.05)&(~r.is_aggregate)]; assert len(w)==1 and 0.0<=w.recall.iloc[0]<=1.0; print(r[~r.is_aggregate].groupby('method').recall.mean().round(3))"`

- [ ] 9. Implement plot (headline figures)
  - `plot(results_df, output_path=None)` per Algorithm 8 (R12): budget-vs-recall and budget-vs-weighted-recall; oracle dashed; random band mean ± BAND_K·std (BAND_K = 3); output_path is a PREFIX — write {prefix}_recall.png and {prefix}_weighted.png and return the two figures; legend, grid, labels
  - _Requirements: R12_
  - Demo: `python -c "import os; from planner.backtest import *; d18,d22,t=make_fake(400,2,0.5,0.5); figs=plot(run_backtest(d18,d22,5.0),'/tmp/fig'); assert len(figs)==2 and os.path.exists('/tmp/fig_recall.png') and os.path.exists('/tmp/fig_weighted.png'); print('2 figures returned and written')"`

- [ ] 10. Implement bootstrap_gap_ci and wire into plot
  - `bootstrap_gap_ci(selections, outcomes, adt, metric) -> {method: {budget: (lo, hi)}}` per Algorithm 8 / R12.2: selections fixed; per replicate `w = rng.multinomial(n_pool, uniform)` over pool row indices; numerator and denominator as weighted sums; gap vs worst_first; metric="recall" for Fig 1, "weighted_recall" for Fig 2; BOOTSTRAP_REPS = 1000, rng seed BOOTSTRAP_SEED = 0; report notes CIs capture outcome-sampling variance only
  - _Requirements: R12_
  - Demo: `python -c "from planner.backtest import *; d18,d22,t=make_fake(400,2,0.5,0.5); r=run_backtest(d18,d22,5.0); sel=...; ci=bootstrap_gap_ci(sel,o,a,'recall'); assert ci['worst_first'][0.05]==(0,0) and all(lo<=hi for m in ci.values() for lo,hi in m.values()); print('worst_first CI [0,0]; lo<=hi everywhere')"`

- [ ] 11. Property: budget compliance
  - Create `planner/properties.py`; `test_budget_compliance()` (no arguments): total_cost ≤ available·(1+1e-9), 100 make_fake datasets, all methods, all budgets (random_avg on 20 test seeds)
  - _Requirements: R14.1_
  - Demo: `python -m pytest planner/properties.py::test_budget_compliance -q`

- [ ] 12. Property: no duplicate selections
  - `test_no_duplicates()`: len(selected) == len(set(selected)), every method and budget
  - _Requirements: R14.2_
  - Demo: `python -m pytest planner/properties.py::test_no_duplicates -q`

- [ ] 13. Property: deterioration monotonicity
  - `test_deterioration_monotonicity()` (no arguments) per R14.3: (a) exhaustive min step over all 11³ (deck, super, sub) triples (0–9 + NaN) vs a reference implementation; (b) safety-drop/exclusion comparison over all 11×11 (min_2018, min_2022) combos incl. NaN against `is_safety_drop = (end < start) and (end <= 4)`; (c) 20,000 random triple pairs vectorized through `apply_exclusion_and_label` vs the reference
  - _Requirements: R14.3_
  - Demo: `python -m pytest planner/properties.py::test_deterioration_monotonicity -q`

- [ ] 14. Property: oracle upper bound (soft aggregate)
  - `test_oracle_upper_bound()`: over 50 make_fake seeds, mean oracle weighted recall ≥ each method's mean − 0.01 per budget; per-seed violations reported as a table, no single-seed failure; real-data leg when adapter output exists; unweighted NOT tested
  - _Requirements: R14.4_
  - Demo: `python -m pytest planner/properties.py::test_oracle_upper_bound -q`

- [ ] 15. Property: exclusion consistency
  - `test_exclusion_consistency()`: no excluded id selected; pool = input − (all R2 reasons incl. duplicates and missing year_built/adt); excluded ids absent from budget denominator
  - _Requirements: R14.5_
  - Demo: `python -m pytest planner/properties.py::test_exclusion_consistency -q`

- [ ] 16. Property: sanity — noise=0 near oracle ("ceiling (uses truth)")
  - `test_sanity_noise_zero_near_oracle()`: make_fake(signal=1, noise=0), model score = truth.p_decline·log1p(adt), labeled "ceiling (uses truth)"; run once, record realized model/oracle weighted-recall ratio per budget, REPORT the ratios, set the assertion bar from the realized ratios (not a fixed constant)
  - _Requirements: R14.6_
  - Demo: `python -m pytest planner/properties.py::test_sanity_noise_zero_near_oracle -q`

- [ ] 17. Property: sanity — signal=0 near random
  - `test_sanity_signal_zero_near_random()` (oracle excluded; baselines run with tiebreak="random"): Leg B is the only gate — per seed d = method recall − dataset random_avg mean (20 test seeds), assert |mean(d)| ≤ BAND_K·std(d)/√20 over 20 make_fake seeds, computed for BOTH unweighted recall AND weighted recall
  - _Requirements: R14.7_
  - Demo: `python -m pytest planner/properties.py::test_sanity_signal_zero_near_random -q`

- [ ] 18. Property: base-rate calibration
  - `test_base_rate_calibration()`: over 20 make_fake seeds (signal=0, noise=1), mean(worse) (safety-drop label) over the pool returned by apply_exclusion_and_label ∈ [0.12, 0.22]
  - _Requirements: R14.8_
  - Demo: `python -m pytest planner/properties.py::test_base_rate_calibration -q`

- [ ] 19. Property: generator consistency
  - `test_generator_consistency()`: for every pool bridge, pipeline-computed worse (apply_exclusion_and_label via is_safety_drop) equals the generator's drawn worse, across seeds and (signal, noise) settings
  - _Requirements: R14.9_
  - Demo: `python -m pytest planner/properties.py::test_generator_consistency -q`

- [ ] 20. Property: safety-drop parity with model
  - `test_is_safety_drop_parity()`: compare local `is_safety_drop` with `model.is_safety_drop` over all 11×11 inputs (0–9 plus NaN for start and end); SKIPPED if `model` is not importable (pytest.importorskip or skipif)
  - _Requirements: R14.10_
  - Demo: `python -m pytest planner/properties.py::test_is_safety_drop_parity -q` (reports SKIPPED until model exists)

- [ ] 21. Refactor fetch_years.py — required
  - Download to a temp filename and rename on success (no partial files in data/)
  - Wrap logic in functions with an `if __name__ == "__main__":` guard
  - Resolve data/ relative to the script's own folder (works from any cwd)
  - Write the 2018-vs-2022 overlap summary to `data/overlap.txt` (in addition to stdout)
  - _Requirements: supports Algorithm 10 inputs_
  - Demo: `python fetch_years.py && test -s data/overlap.txt && head -3 data/overlap.txt`

- [ ] 22. Build NBI adapter (adapt_nbi) — required
  - Run refactored `fetch_years.py` to obtain MA18.txt / MA22.txt (or confirm present in data/); implement `adapt_nbi(path_2018, path_2022, flat_inspect_cost, county_fee)` per Algorithm 10
  - Select columns BY NAME ONLY; cleaning order: strip whitespace → ""/"N"→NaN → numeric; county codes stripped and zero-padded to 3 digits in both years; structure_id stripped both years
  - Print BOTH unmatched counts (2018-absent-from-2022 and 2022-absent-from-2018); if LOWEST_RATING exists in 2022, print rows differing from computed min
  - rebuilt = (YEAR_RECONSTRUCTED_106 in 2022) > (YEAR_RECONSTRUCTED_106 in 2018), blank/0 as 0; inspect_cost flat placeholder in "placeholder units", labeled as assumption (item _094 exists but is improvement cost, not used); print count of rebuilt=True bridges
  - _Requirements: R1, R2, Assumptions 2, 8_
  - Demo: `python -c "from planner.backtest import adapt_nbi; d18,d22=adapt_nbi('data/MA18.txt','data/MA22.txt',1.0,5.0); print(len(d18),'bridges adapted; rebuilt/unmatched counts printed above')"`

- [ ] 23. Run baselines on real data — required
  - Run worst_first, oldest_first, random_avg, oracle on adapted data, ALL budgets, `tiebreaks=("adt", "random")` so the ADT tie-break effect is visible; report both (the "random" leg is one mean row per baseline)
  - Print: pool size, count of worse bridges, recall and weighted recall per method per budget, filling blank tiebreak with "adt" before printing; note that flat inspect_cost = 1 makes value-per-cost ranking equal raw-score ranking (placeholder units); save plots
  - _Requirements: R10, R11, R12_
  - Demo: `python -c "from planner.backtest import *; d18,d22=adapt_nbi('data/MA18.txt','data/MA22.txt',1.0,5.0); r=run_backtest(d18,d22,5.0,tiebreaks=('adt','random')); r['tiebreak']=r['tiebreak'].fillna('adt'); print(r[['method','budget','tiebreak','recall','weighted_recall']].dropna())"`

## Task Dependency Graph

```
1 (package + load/validate + constants)
 ├─→ 2 (make_fake; needs only R1 formats)
 ├─→ 3 (exclusion/label)
 │    └─→ 4 (cost rule)
 │         └─→ 5 (pick)
 │              ├─→ 6 (scoring) ─→ 7 (metrics) ─→ 8 (run_backtest) ─→ 9 (plot) ─→ 10 (bootstrap_gap_ci)
 │              └────────────────── 2,3,4 feed 5 ─┘
 │
11–20 (properties) depend on 1–8 (14 uses 22 output for its real-data leg; must pass on fake alone; 20 skips until model exists)
21 (fetch_years refactor) is independent; produces raw inputs for 22
22 (adapter) depends on 1 and 21
23 (real-data run) depends on 22, 8, 9, 10; independent of 11–20
```
