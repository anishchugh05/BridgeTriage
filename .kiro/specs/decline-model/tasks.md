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

- [x] 7. Implement `score` — p_worse, consequence, cost, score, reason
  - Exact six-column, NaN-free output with a one-line reason per row.
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
  - 80/20 bridge split (fixed seed), fit on 2012→2016 train, evaluate on
    held-out 20%; report positives/log loss/AUC for grouped, logistic, and the
    per-condition baseline; print drop rate by condition. Real-panel or
    synthetic fallback.
  - _Requirements: 4, 6.2_

- [x] 12. Test suite (`tests/test_model.py`)
  - Safety-drop definition, rebuild exclusion, no-lookahead, clean output,
    missing-date fallback, word-boundary crossings, material-name reasons, and
    a logistic-model smoke test. 12 tests passing.
  - _Requirements: 2b, 6.3_
