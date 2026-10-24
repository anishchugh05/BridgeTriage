# Model card — BridgeTriage decline model (Part B)

This card documents the model that scores Massachusetts bridges for the
BridgeTriage drone-inspection planner. All numbers below come from the
cross-validation harness (`model/validate.py`) on the real FHWA NBI panel; all
costs and probabilities are **planning estimates, not official measurements**.

## What it predicts (target)

A **safety drop** between two inspection years, using the worst of the deck,
superstructure, and substructure condition ratings (NBI items 058/059/060,
0–9 scale):

- the bridge crosses into poor condition — end grade ≤ 4 while start grade ≥ 5, **or**
- it was already poor (start ≤ 4) and fell at least one more grade.

High-grade slips that stay above 4 (e.g. 9 → 7) are **not** drops: the bridge is
still in fair-or-better shape. Bridges with a null `worst_rating` (culverts,
dam spillways, etc.) are not applicable and are excluded.

## Training data

- Source: FHWA National Bridge Inventory, Massachusetts, via Part A's cleaned
  `bridges/panel.parquet`. Panel years: **2012, 2016, 2018, 2022**.
- Training interval: **2012 → 2016** (a four-year horizon). The two-year
  2016 → 2018 gap is never fit; it would mismatch the backtest horizon.
- Each row is a bridge paired across the interval by NBI structure number.
  Rebuilt pairs (Part A's `rebuilt` flag, or a reconstruction-year change) and
  condition improvements are excluded as non-decline events.
- Rare event: 191 safety drops among 4,183 eligible paired bridges in CV.

## Features

The selected model is a logistic regression on:

- **condition** (worst_rating), one-hot over grades 0–9
- **material** (NBI item 043A), one-hot
- **age** (years since built)
- **log ADT** (average daily traffic)
- **insp_freq_mo** (required inspection frequency)
- **fracture_critical** flag
- **underwater** flag

`months_since_insp` (months from last inspection to the snapshot date, capped at
60) is available but is **not** in the selected model — see "What didn't help."

## Cross-validation results

Five folds, grouped by bridge id (a bridge never appears in both train and test
of a fold). Pooled across folds: 4,183 test pairs, 191 positives. Mean ± std
over the five folds:

| model | AUC (mean ± std) | log loss (mean ± std) |
|---|---|---|
| per-condition baseline | 0.8020 ± 0.0118 | 0.1539 ± 0.0207 |
| grouped (material × age-band × condition) | 0.8105 ± 0.0134 | 0.1566 ± 0.0234 |
| **logistic without months_since_insp** | **0.8188 ± 0.0232** | 0.1540 ± 0.0229 |
| logistic with months_since_insp | 0.8133 ± 0.0211 | 0.1544 ± 0.0228 |

**Selection rule:** pick by mean CV AUC, but prefer the simpler model unless a
more complex one beats it by more than one standard deviation. The logistic
without `months_since_insp` has the best mean AUC and is selected; adding the
inspection-recency feature does not improve it beyond noise, so the simpler
logistic wins.

## Calibration (selected model, out-of-fold)

Predicted probabilities binned into five equal-count quantiles; predicted vs.
actual drop rate on held-out folds:

| bin | n | mean predicted | actual drop rate |
|---|---|---|---|
| 1 (lowest) | 837 | 0.0033 | 0.0036 |
| 2 | 836 | 0.0066 | 0.0012 |
| 3 | 837 | 0.0139 | 0.0179 |
| 4 | 836 | 0.0648 | 0.0730 |
| 5 (highest) | 837 | 0.1403 | 0.1326 |

Predicted and actual track closely across the range — the model is well
calibrated, not just well ranked. The highest-risk quintile drops ~13% of the
time versus ~0.4% in the lowest.

## Top 10 coefficients (log-odds), explained

| feature | coef | one-line reading |
|---|---|---|
| cond=5 | +2.03 | grade 5 is the biggest drop risk — right on the edge of poor, most room to slip across |
| cond=4 | +1.42 | grade 4 is already poor and frequently worsens further |
| cond=7 | −1.29 | grade 7 bridges are solidly fair and rarely drop to poor in four years |
| cond=8 | −1.01 | grade 8 (good) bridges almost never cross into poor so fast |
| material=Concrete | −0.79 | plain concrete spans drop less often than the baseline mix |
| material=Prestressed continuous | −0.66 | prestressed-continuous spans are relatively robust over four years |
| cond=9 | −0.54 | grade 9 (excellent) bridges are the least likely to reach poor |
| cond=3 | −0.54 | grade 3 is already deep in poor, with little grade left to lose (floor effect) |
| material=Steel | +0.44 | steel spans drop somewhat more often than average |
| material=Concrete continuous | −0.36 | concrete-continuous spans drop slightly less than average |

The condition grade dominates, which matches intuition: a bridge sitting at 5 or
4 is where a four-year slip into poor is most likely, while already-floor (3) and
already-good (7–9) bridges are both unlikely to *newly* cross the poor line.

## Limitations

- **Rare, noisy target.** Only 191 drops in 4,183 CV pairs; AUC differences
  between models are small relative to their fold-to-fold std, so model choice
  is a modest edge, not a decisive one.
- **Four-year horizon only.** Trained on 2012 → 2016 and validated against
  2018 → 2022-style gaps; it is not a per-year forecast.
- **Condition is self-reported NBI data.** We model recorded rating changes,
  which can reflect inspection timing and inspector judgment, not only physical
  deterioration. Rebuilds are excluded via Part A's flag, which we trust.
- **No structure-length column yet**, so drone-cost length is a material-median
  estimate (every reason says "length estimated").
- **Costs and flight times are planning estimates**, not vendor quotes.
- **Massachusetts only**; coefficients should not be transferred to other states
  without refitting.
- Geographic, environmental, and materials-chemistry factors (salt exposure,
  flood history, traffic loading beyond ADT) are not modeled.

## What we tried that didn't help

- **months_since_insp as a feature.** Added and A/B-tested in CV; the logistic
  *with* it (AUC 0.8133) did not beat the logistic *without* it (0.8188), so the
  selection rule correctly keeps the simpler model. Inspection recency carried
  little signal for this four-year drop target.
- **The grouped (material × age-band × condition) model.** It beats the baseline
  (0.8105 vs 0.8020) but loses to logistic and is harder to calibrate, so it is
  not selected.
- **Treating grade monotonically.** An earlier version forced drop probability
  to move monotonically with grade; the real data is non-monotone (grade 5 is
  riskier than both 3 and 7), so that constraint was removed.
- **improvement_cost as inspection cost.** It is a repair-cost field, not an
  inspection price, so it is never used as `insp_cost`.
