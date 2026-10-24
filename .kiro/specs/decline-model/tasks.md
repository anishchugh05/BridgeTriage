# Tasks — Decline Model (Part B)

Documents work already completed in `model/`. All boxes are checked because the
spec was written to capture the implemented, tested behavior.

- [x] 1. Scaffold the model package and public API
  - Export `fit_decline`, `score`, `DeclineRates` from `model/__init__.py`.
  - _Requirements: 1, 5_

- [x] 2. Condition extraction honoring NBI "N"
  - Condition = min(deck, super, sub), falling back to culvert; all-NaN → NaN.
  - Row-wise and vectorized variants; NaN never treated as 0.
  - _Requirements: 3_

- [x] 3. Implement `fit_decline` with no-lookahead pairing
  - Filter to `start`/`end` years, pair by `id`, label grade drops.
  - Raise `ValueError` when `end <= start`.
  - _Requirements: 1, 2_

- [x] 4. Exclude rebuilt/repaired bridges from training
  - Drop pairs where `year_recon` changed or condition improved.
  - _Requirements: 2_

- [x] 5. Grouped drop-probability estimates with shrinkage
  - material × age band × condition, shrunk toward material × condition, then
    condition alone, then global.
  - _Requirements: 4_

- [x] 6. Shared safety-drop target and horizon→hazard conversion
  - `model/target.py::is_safety_drop` (+ vectorized) as the single definition;
    retarget the grouped model to it; `h = -ln(1-p)/horizon_months`.
  - _Requirements: 2b, 4_

- [x] 6b. Second model: logistic regression
  - `fit_logistic` on condition/material one-hot + age, log ADT, insp_freq_mo,
    fracture_critical, underwater; same interface as the grouped model.
  - _Requirements: 4_

- [x] 6c. Model selection by cross-validated AUC
  - `validate.select_model` (simpler-preferred unless +1 std); `score` accepts
    either model.
  - _Requirements: 4_

- [x] 6d. Material code→name lookup
  - `model/materials.py` (043A codes) rendered in `score` reasons.
  - _Requirements: 5_

- [x] 6e. months_since_insp feature
  - Months from `insp_date` to the row's NBI snapshot (April 1), capped at 60,
    fallback to `insp_freq_mo` when missing; used in the logistic model and the
    `score` reason ("last inspected N mo ago"). `load_panel` parses `insp_date`.
  - _Requirements: 4_

- [x] 6f. 5-fold CV grouped by bridge id
  - `validate.cross_validate` reporting mean ± std AUC/log loss for grouped,
    logistic(±msi), and baseline; plus selected logistic coefficients and the
    safety-drop rate by months_since_insp bucket.
  - _Requirements: 4_

- [x] 7. Implement `score` — model-specific horizon scaling and normalized output
  - Logistic: direct 12-month probability scaling, no recency hazard replay.
  - Grouped: retain recency hazard path.
  - Eight-column NaN-free output; `expected_drops_per_10k == score`.
  - Use the labeled drone-flight cost/minutes estimate when no real inspection
    price exists.
  - _Requirements: 5_

- [x] 8. Fix inspection-date fallback (bug)
  - Missing `insp_date` → `insp_freq_mo` else 24 mo; reason notes "inspection
    date unknown"; does not inflate rank.
  - _Requirements: 5.2_

- [x] 9. Fix crossing detection with word boundaries (bug)
  - Word-boundary regex for RR/RAILROAD/RAILWAY/MBTA and water tokens; "CHERRY
    ST" not rail, "RR MBTA"/"BMRR" are.
  - _Requirements: 5.4_

- [x] 10. Synthetic panel generator (`model/fake_data.py`)
  - Contract columns and types for 2012–2022 with a learnable decline process,
    rebuilds, culvert N/A, and missing costs.
  - _Requirements: 6.1_

- [x] 11. Validation harness (`model/validate.py`)
  - Five folds grouped by eligible bridge id; mean ± std AUC/log loss for all
    four candidates; five-bin selected-model OOF calibration; inspection-age
    drop rates and selected logistic coefficients.
  - _Requirements: 4, 6.2_

- [x] 12. Test suite (`tests/test_model.py`)
  - Target/rebuild/no-lookahead, direct logistic and grouped hazard paths,
    normalized score formula, calibration, missing-date/crossing handling,
    named coefficients, null-rating omission, and batch files. 20 tests.
  - _Requirements: 2b, 5, 6.4_

- [x] 13. Batch scorer (`model/run.py`)
  - Select/refit on 2012→2016; score 2018/2022 on April 1; write both parquet
    outputs under `data/processed/`.
  - _Requirements: 6.3_

- [x] 14. Drone-flight cost model (`model/flight_cost.py`)
  - Estimate cost and minutes from length, water/rail access, and nearest depot;
    use material medians / 60 km location fallback with explicit labels.
  - Integrate into score output and reasons; test length, crossing, distance,
    and missing coordinates.
  - _Requirements: 5.5, 5.6, 5.9_