# Implementation Plan: Production Bridge Inspection Planner

## Overview

Implement the approved Python/Streamlit production application incrementally: establish strict runtime and schema boundaries, compose the existing real model with a deterministic planner and leakage-free backtest, refactor the UI around one validated statewide selection, then add reproducible deployment manifests, automated application checks, and CI validation. Every implementation step names its target files and builds on earlier validated contracts.

## Tasks

- [x] 1. Establish Python package, runtime-mode, provenance, and error foundations
  - [x] 1.1 Create explicit application and planner packages
    - Create `app/__init__.py` and replace `planner/.gitkeep` with `planner/__init__.py` so repository-root imports have explicit package boundaries.
    - Create the `tests/unit/`, `tests/property/`, `tests/integration/`, `tests/app/`, and `tests/repository/` package directories needed by later automated tests.
    - _Requirements: 4.1, 5.1, 9.1, 9.3_

  - [x] 1.2 Implement exact runtime-mode resolution and provenance contracts in `app/config.py`
    - Define `RuntimeMode`, `Provenance`, and provider mode-tag contracts; resolve only absent, exact `production`, and exact `demo` settings without trimming or case folding.
    - Reject every other present `BRIDGE_APP_MODE` value before provider construction or data access.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

  - [x] 1.3 Implement safe, dependency-scoped error primitives in `app/errors.py`
    - Define typed internal failures, the fixed error-code registry, `UserSafeError`, `Outcome[T]`, one-time structured logging, and rendering of exactly component, summary, and code.
    - Keep exception details, paths, environment values, secrets, stack traces, and raw rows out of public diagnostics.
    - _Requirements: 8.1, 8.8, 8.9, 8.10, 8.11, 8.12, 8.13_

  - [ ]* 1.4 Write the property test for exact runtime-mode resolution in `tests/property/test_property_01_runtime_mode.py`
    - **Property 1: Exact runtime-mode resolution**
    - Generate absent, accepted, case-varied, whitespace, empty, and arbitrary setting values; assert exclusive activation or the stable invalid-mode failure before data access.
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.5, 1.7, 1.8**

  - [ ]* 1.5 Write the property test for stable sanitized diagnostics in `tests/property/test_property_24_public_diagnostics.py`
    - **Property 24: Public diagnostics are stable and sanitized**
    - Generate arbitrary exception payloads and assert exact public fields, stable category mapping, matching structured-log identity, and sensitive-value exclusion.
    - **Validates: Requirements 8.9, 8.10, 8.12, 8.13, 11.13**

  - [ ]* 1.6 Write runtime and error example tests in `tests/unit/test_config_examples.py` and `tests/unit/test_errors.py`
    - Cover the exact accepted literals, opposite-mode deactivation, fixed registry entries, one log event per handled exception, and invalid-mode short-circuiting.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 8.9, 8.10, 8.11, 8.12, 8.13, 11.12, 11.13_

- [x] 2. Make `model.data` the sole strict source and score integration boundary
  - [x] 2.1 Implement source reading, validation, and lossless adaptation in `model/data.py`
    - Define the exact ordered 21-column `SOURCE_COLUMNS` contract, typed schema failures, repository-relative `read_source_panel`, non-mutating `validate_source_panel`, and `adapt_source_panel`.
    - Validate required years, unique keys, nonempty strings, integral and finite fields, ratings and rowwise minimum semantics, exact booleans, reconstruction years, repair costs, coordinate bounds, and month-start inspection dates.
    - Rename only `structure_id` to `id` and `insp_freq_months` to `insp_freq_mo`, add `year_built = year - age`, preserve zero reconstruction years and `improvement_cost`, and retain a compatibility `load_panel` wrapper over this boundary.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19, 2.20, 2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.29_

  - [x] 2.2 Add scored-output validation and identifier reconciliation to `model/data.py`
    - Validate exact score columns/order, one-to-one ranking identifiers, complete non-null output, numeric bounds, score arithmetic, nonblank reasons, and modeled-cost disclosure.
    - Reconcile model `id` to 2022 source `structure_id` with one-to-one merge validation and fail atomically on missing, duplicate, or unexpected identifiers.
    - _Requirements: 2.27, 2.28, 2.30, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18, 3.19, 3.27_

  - [x] 2.3 Export the authoritative data-boundary API from `model/__init__.py`
    - Export panel bundle, schema adapter, scored-table validator, and reconciliation entry points without introducing a second mapping in `app` or `planner`.
    - Preserve the existing public fitting, scoring, and Safety_Drop exports used by services and backtests.
    - _Requirements: 2.18, 2.19, 2.20, 2.21, 2.27, 4.2, 5.16_

  - [ ]* 2.4 Write the source-schema property test in `tests/property/test_property_04_source_schema.py`
    - **Property 4: Source validation is sound and complete for the published schema**
    - Generate valid source frames and focused invalid mutations for every schema predicate; assert invalid input never produces an adapted panel.
    - **Validates: Requirements 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.29, 11.1, 11.2**

  - [ ]* 2.5 Write the lossless-adaptation property test in `tests/property/test_property_05_adaptation.py`
    - **Property 5: Adaptation is lossless except for the declared mapping**
    - Check source-frame immutability, row and value preservation, exact renames, derivation, identifier-year bijection, and the absence of an `improvement_cost` to `insp_cost` mapping.
    - **Validates: Requirements 2.18, 2.19, 2.20, 2.21, 2.22, 2.24, 2.25, 2.26, 11.3**

  - [ ]* 2.6 Write the reconciliation property test in `tests/property/test_property_06_reconciliation.py`
    - **Property 6: Score reconciliation is an identifier bijection**
    - Generate equal and mismatched source/score identifier sets and assert successful one-row-per-bridge reconciliation only for equal unique sets.
    - **Validates: Requirements 2.27, 2.28, 11.3**

  - [ ]* 2.7 Write the concrete source acceptance/rejection matrix in `tests/unit/test_source_schema_matrix.py`
    - Cover all 21 columns in order, four required years, nullable ratings, coordinates, booleans, dates, uniqueness, and each required missing/extra/reordered/out-of-range mutation.
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 11.1, 11.2_

  - [ ]* 2.8 Write adapter and real-asset integration tests in `tests/integration/test_schema_adapter.py`
    - Validate exact renames/derivation, source-memory and source-byte immutability, row-count preservation, identifier reconciliation failures, and loading only `bridges/panel.parquet` through file-relative paths.
    - _Requirements: 2.1, 2.18, 2.19, 2.20, 2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.27, 2.28, 2.29, 2.30, 9.4, 9.5, 11.3_

- [x] 3. Adjust real-model cost provenance and implement validated current-model caching
  - [x] 3.1 Make the narrow modeled-cost provenance change in `model/score.py`
    - Replace fallback-cost wording with the exact phrase `modeled inspection-cost estimate` while preserving constants, fitting behavior, scoring arithmetic, and output schema.
    - Update model comments/docstrings so `improvement_cost` remains explicitly distinct from inspection cost.
    - _Requirements: 3.15, 3.18, 3.19, 7.21, 7.22_

  - [ ]* 3.2 Update and extend real-model tests in `tests/test_model.py` and `tests/unit/test_model_cost_provenance.py`
    - Assert the exact modeled-cost phrase, positive fallback costs, no dependency on `improvement_cost`, unchanged six-column output, and exact score arithmetic.
    - _Requirements: 3.9, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18, 3.19, 11.5_

  - [x] 3.3 Create validated cache identities and transactional cache wrappers in `app/services.py`
    - Add immutable identities for source, model state, scored table, and backtest values using runtime mode, deterministic content digests, intervals, dates, configuration fingerprints, and budgets.
    - Wrap `st.cache_data`/`st.cache_resource` behind a testable `ModelCache` that validates before exposure, supports exact-key eviction, never caches failures, and removes stale successes before forced recomputation.
    - _Requirements: 1.16, 1.17, 3.20, 3.21, 3.22, 3.23, 3.24, 3.25, 3.26, 5.23, 5.24, 5.25, 5.28, 8.14, 8.15, 8.17, 8.18_

  - [x] 3.4 Implement the bounded `CurrentModelService` in `app/services.py`
    - Fit `model.fit_decline` from an explicitly bounded 2018/2022 frame, score the complete 2022 adapted frame with `model.score(..., 2022-04-01)`, validate model state and score output, and reconcile every ranking identifier before return.
    - Keep coordinate filtering outside fitting, scoring, reconciliation, and cache keys derived from the complete ranking population.
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.20, 3.21, 3.27_

  - [ ]* 3.5 Write the bounded-current-model property test in `tests/property/test_property_07_current_model.py`
    - **Property 7: Current-model inputs are time-bounded and population-complete**
    - Use injected spies and generated panels to assert exclusive fit years, eligible pair rules, exact as-of date, and complete scoring despite invalid or null coordinates.
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 11.4**

  - [ ]* 3.6 Write the scored-contract property test in `tests/property/test_property_08_scored_contract.py`
    - **Property 8: Valid scored output obeys one complete algebraic contract**
    - Mutate schema, identifiers, bounds, finite values, arithmetic, reasons, cost disclosure, and repair cost independently; accept only complete valid outputs.
    - **Validates: Requirements 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18, 3.19, 3.27, 11.5**

  - [ ]* 3.7 Write the transactional-cache property test in `tests/property/test_property_09_transactional_cache.py`
    - **Property 9: Validated caches are transactional**
    - Generate successful, changed, failed, forced, and recovered computation sequences and assert reuse, invalidation, and validate-before-store behavior.
    - **Validates: Requirements 3.22, 3.23, 3.24, 3.25, 3.26, 5.24, 5.25, 5.28, 8.14, 8.15, 8.17, 8.18, 11.6**

  - [ ]* 3.8 Write current-model and cache integration tests in `tests/integration/test_current_model_service.py`
    - Spy on real-service boundaries for exact 2018/2022 fit input, 2022 complete scoring input, `2022-04-01`, rebuilt/null exclusions, no-coordinate retention, reuse counters, changed-key recomputation, and invalid-output rejection.
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.20, 3.21, 3.22, 3.23, 3.24, 3.25, 3.26, 3.27, 11.4, 11.6_

- [x] 4. Implement the deterministic real planner and exact selection metrics
  - [x] 4.1 Create planner contracts in `planner/contracts.py`
    - Validate exact scored input, duplicate IDs, finite numeric fields, exclusive selection mode, budget types/range, count types/range, and immutable Selection/SelectionMetrics postconditions.
    - Reject booleans as numeric budget/count values and never return partial selections on validation failure.
    - _Requirements: 4.2, 4.4, 4.5, 4.6, 4.12, 4.16, 4.17, 4.18_

  - [x] 4.2 Implement selection and metrics in `planner/selection.py` and export them from `planner/__init__.py`
    - Sort by descending `score` then ascending string `id`; implement budget selection with exact decimal affordability and continuation past oversized candidates.
    - Implement deterministic fixed-count prefixes and derive selected count, exact selected cost, and expected Safety_Drop coverage from one validated selection.
    - _Requirements: 4.1, 4.3, 4.7, 4.8, 4.9, 4.10, 4.11, 4.13, 4.14, 4.15, 4.16, 4.17, 4.18, 4.19, 4.20, 4.21_

  - [ ]* 4.3 Write the budget-selection property test in `tests/property/test_property_10_budget_selection.py`
    - **Property 10: Budget selection is deterministic, unique, and affordable**
    - Generate valid scored tables and budgets; compare output to the specified greedy traversal and assert order, continuation, uniqueness, membership, and exact affordability.
    - **Validates: Requirements 4.2, 4.3, 4.4, 4.6, 4.7, 4.8, 4.9, 4.10, 4.11, 4.16, 4.17, 4.18, 11.7, 11.8**

  - [ ]* 4.4 Write the fixed-count property test in `tests/property/test_property_11_fixed_count.py`
    - **Property 11: Fixed-count selection is the deterministic prefix**
    - Generate counts and invalid mode combinations; assert exact prefixes, zero/oversized behavior, uniqueness, membership, and no Selection for invalid requests.
    - **Validates: Requirements 4.4, 4.5, 4.12, 4.13, 4.14, 4.15, 4.16, 4.17, 4.18, 11.7, 11.8**

  - [ ]* 4.5 Write the selection-metrics property test in `tests/property/test_property_12_selection_metrics.py`
    - **Property 12: Selection metrics are exact projections of one selection**
    - Assert metrics exactly equal projections of selected IDs and remain identical across every summary consumer.
    - **Validates: Requirements 4.19, 4.20, 4.21, 6.9, 6.10, 7.1, 11.8**

  - [ ]* 4.6 Write the concrete planner matrix in `tests/unit/test_selection.py`
    - Cover zero and exact-fit budgets, unaffordable leading and later affordable candidates, score ties, duplicate inputs, invalid numeric values, zero/oversized counts, exclusive modes, ordering, membership, totals, and coverage.
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 4.11, 4.12, 4.13, 4.14, 4.15, 4.16, 4.17, 4.18, 4.19, 4.20, 4.21, 11.7, 11.8_

- [x] 5. Implement a two-phase leakage-free real backtest
  - [x] 5.1 Implement historical planning in `planner/backtest.py`
    - Define immutable `BacktestPlan` audit data; pass only 2012/2016 rows to `fit_decline`, score the complete unique 2018 candidates at `2018-04-01`, and freeze candidate IDs, 2018 modeled costs, budgets, and all method selections before outcomes are available.
    - Implement `ours`, null-last `worst_first`, and descending-age `oldest_first` ordering with ascending ID tie breaks and one shared budget-affordability iterator.
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 5.12, 5.13, 5.26_

  - [x] 5.2 Implement outcome evaluation, result validation, and public orchestration in `planner/backtest.py` and `planner/__init__.py`
    - Match at most one 2018 and 2022 record per selected ID, exclude missing/null/rebuilt outcomes, call the shared Safety_Drop target, and retain measured counts without superiority adjustments.
    - Return exact `budget`, `method`, `caught` columns with one canonical row per distinct budget/method and validate method/budget sets, order, integral counts, and eligible-count bounds before return.
    - _Requirements: 5.1, 5.14, 5.15, 5.16, 5.17, 5.18, 5.19, 5.20, 5.21, 5.22, 5.27, 5.28_

  - [ ]* 5.3 Write the no-future-observation property test in `tests/property/test_property_13_backtest_time_bounds.py`
    - **Property 13: Historical planning cannot observe future outcomes**
    - Inject access-recording fit, score, and selection functions; mutate 2022 fields and assert historical planning inputs and frozen plans do not change.
    - **Validates: Requirements 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.26, 11.9**

  - [ ]* 5.4 Write the shared-constraints property test in `tests/property/test_property_14_backtest_methods.py`
    - **Property 14: Backtest methods share constraints and deterministic orderings**
    - Generate candidate tables and assert all methods share IDs, costs, budgets, affordability while obeying each exact deterministic ordering.
    - **Validates: Requirements 5.9, 5.10, 5.11, 5.12, 5.13, 11.9, 11.11**

  - [ ]* 5.5 Write the measured-outcomes property test in `tests/property/test_property_15_backtest_outcomes.py`
    - **Property 15: Backtest outcomes are measured exactly**
    - Generate endpoint availability, ratings, rebuilt flags, selections, and baseline wins/ties; assert exact eligible Safety_Drop counts and canonical result shape/order.
    - **Validates: Requirements 5.14, 5.15, 5.16, 5.17, 5.18, 5.19, 5.20, 5.21, 5.22, 5.27, 11.10, 11.11**

  - [ ]* 5.6 Write the concrete backtest and baseline matrix in `tests/unit/test_backtest.py`
    - Cover both Safety_Drop branches, non-drops, missing records, null ratings, rebuilt endpoints, duplicate records, baseline null/tie ordering, affordability, exact counts/schema/order, and baseline victories without count rewriting.
    - _Requirements: 5.11, 5.12, 5.13, 5.14, 5.15, 5.16, 5.17, 5.18, 5.19, 5.20, 5.21, 5.22, 5.27, 11.10, 11.11_

  - [ ]* 5.7 Write the real backtest boundary integration test in `tests/integration/test_backtest_engine.py`
    - Assert exact fit years/date, common candidates/costs/budgets, no 2022 field access before evaluation, public exports, invalid-result rejection, and deterministic results from the real model/planner composition.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.23, 5.24, 5.25, 5.26, 5.27, 5.28, 11.9, 11.11_

- [x] 6. Complete cached production services and isolate explicit demo providers
  - [x] 6.1 Complete production provider composition and backtest caching in `app/services.py`
    - Add repository-relative source hashing/loading, real model/planner/backtest provider construction, exact production provenance, and cached validated backtest orchestration keyed by source content, distinct budgets, timeline, model configuration, planner configuration, and mode.
    - Keep source, current ranking, selection, and backtest outcomes independent so a branch failure preserves unrelated validated results.
    - _Requirements: 1.10, 1.11, 1.13, 1.14, 3.20, 3.21, 5.23, 5.24, 5.25, 5.28, 8.2, 8.3, 8.4, 8.5, 8.8, 8.14, 8.15, 8.17, 8.18, 9.4, 9.5, 9.6, 9.7_

  - [x] 6.2 Create explicit demo-only providers in `app/demo.py`
    - Move any retained fabricated data/calculation behavior out of fallback imports, tag every demo object with demo provenance, and expose it only through the exact demo provider branch.
    - Remove production use of `app/data.py` path fallback and `app/stubs.py` automatic fallback; production failures must remain production failures.
    - _Requirements: 1.5, 1.6, 1.7, 1.10, 1.12, 1.13, 1.14, 1.15, 8.16, 9.5, 9.6, 9.7_

  - [ ]* 6.3 Write the production-provenance isolation property test in `tests/property/test_property_02_provenance_isolation.py`
    - **Property 2: Production provenance cannot degrade to demo provenance**
    - Generate provider successes, failures, and wrong-mode bundles; assert production accepts only real tags and never activates or substitutes demo output.
    - **Validates: Requirements 1.10, 1.12, 1.13, 1.14, 8.16**

  - [ ]* 6.4 Write the cache-identity isolation property test in `tests/property/test_property_03_cache_identity.py`
    - **Property 3: Cache identities isolate modes and all semantic inputs**
    - Vary one semantic identity input at a time across every namespace and assert equality only when all required inputs, including runtime mode, are equal.
    - **Validates: Requirements 1.16, 1.17, 3.20, 3.21, 3.24, 5.23, 5.25, 11.6**

  - [ ]* 6.5 Write production/demo service integration tests in `tests/integration/test_provider_services.py`
    - Cover exact real-provider provenance, lazy disjoint demo construction, no fallback on each production failure, source/model/score/backtest cache reuse and key changes, invalid-value eviction, and validated recovery.
    - _Requirements: 1.10, 1.11, 1.12, 1.13, 1.14, 1.16, 1.17, 3.22, 3.23, 3.24, 3.25, 3.26, 5.23, 5.24, 5.25, 5.28, 8.14, 8.15, 8.16, 8.17, 8.18, 11.6, 11.12, 11.13_

- [x] 7. Build shared state and pure presentation helpers around one statewide selection
  - [x] 7.1 Implement immutable application state in `app/state.py`
    - Define active-budget validation with initial/last-valid `500000`, exact allowed range/step, `SelectionContext`, `ViewContext`, and deterministic selection identity.
    - Require renderers to consume the single context rather than calling model, planner, cache, or demo code.
    - _Requirements: 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12_

  - [ ]* 7.2 Write the shared-budget-state property test in `tests/property/test_property_16_active_budget.py`
    - **Property 16: Active-budget state yields one shared statewide selection**
    - Generate requested-budget and county-filter sequences; assert valid-state retention, one selection per accepted rerun, shared identity, and county-filter independence.
    - **Validates: Requirements 6.4, 6.5, 6.6, 6.7, 6.8, 6.11, 6.12**

  - [x] 7.3 Create shared notice, headline, map, and Budget renderers in `app/views.py`
    - Render the current-ranking summary and required `Map` before secondary sections; prepare markers only from exact coordinate/county predicates while retaining the complete ranking elsewhere.
    - Encode selected markers red, unselected scores with a nondecreasing sequential transform, ADT with a nondecreasing radius transform, and exact tooltip source/model fields.
    - Render Budget metrics and rows from the shared selection, using hazard-first/score/ID ordering and modeled-cost labels; call the exact demo indicator guard before each data-derived view.
    - _Requirements: 1.15, 6.1, 6.2, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12, 6.13, 6.14, 6.15, 6.16, 6.17, 6.18, 6.19, 6.20, 7.1, 7.2, 7.3, 7.4, 7.21_

  - [x] 7.4 Add measured Backtest and sparse Bridge detail renderers to `app/views.py`
    - Plot unmodified method counts and format same-budget comparisons without overstatement or zero-baseline ratios.
    - Reindex detail history to 2012/2016/2018/2022 with `Unavailable`, show reconciled reason/descriptors, and build an identifier-associated FHWA InfoBridge URL without imputation.
    - Apply the exact demo indicator guard to both sections.
    - _Requirements: 1.15, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 7.11, 7.12, 7.13_

  - [x] 7.5 Add Hazards, Export, and Methods renderers to `app/views.py`
    - Intersect exact `underwater OR fracture_critical` hazards with the shared selection and apply canonical ordering.
    - Build exact selected-only CSV columns/order/values, adjacent real provenance and modeled-cost disclosure, and an immediately adjacent demo indicator before a demo download.
    - Render all approved model/backtest timelines, Safety_Drop/rebuilt/cost assumptions, schema mapping/reconciliation, NBI-to-SNBI transition, and four production source labels.
    - _Requirements: 1.11, 1.15, 7.14, 7.15, 7.16, 7.17, 7.18, 7.19, 7.20, 7.21, 7.22, 7.23, 7.24, 7.25, 7.26, 7.27, 7.28_

  - [ ]* 7.6 Write the map-projection property test in `tests/property/test_property_17_map_projection.py`
    - **Property 17: Map rendering is a projection, not a population filter**
    - Vary coordinates/counties and assert exact marker membership while scores, selection, metrics, detail, hazards, tables, and export populations remain unchanged.
    - **Validates: Requirements 3.8, 6.11, 6.12, 6.13, 6.14, 6.15, 6.16, 6.20**

  - [ ]* 7.7 Write the map-encoding property test in `tests/property/test_property_18_map_encodings.py`
    - **Property 18: Map encodings and tooltips preserve ordering and source values**
    - Generate score/ADT pairs and source strings; assert monotonic/equal encodings and exact tooltip values.
    - **Validates: Requirements 6.17, 6.18, 6.19**

  - [ ]* 7.8 Write the selected-evidence property test in `tests/property/test_property_19_selected_evidence.py`
    - **Property 19: Selected evidence and hazards use one canonical ordering**
    - Assert exact hazard classification/intersection, source/model value retention, and hazard/score/ID ordering for selected views.
    - **Validates: Requirements 7.2, 7.3, 7.4, 7.14, 7.15**

  - [ ]* 7.9 Write the backtest-presentation property test in `tests/property/test_property_20_backtest_presentation.py`
    - **Property 20: Backtest presentation never overstates measured results**
    - Generate method counts including wins, ties, losses, and zero baselines; assert exact plotted values and permitted comparison copy/formulas.
    - **Validates: Requirements 7.5, 7.6, 7.7, 7.8, 7.9**

  - [ ]* 7.10 Write the bridge-detail property test in `tests/property/test_property_21_bridge_detail.py`
    - **Property 21: Bridge detail preserves sparse history and identifier association**
    - Generate sparse histories and identifiers; assert ordered year slots, exact available values, `Unavailable` gaps, reconciled text, and encoded-ID URLs.
    - **Validates: Requirements 7.10, 7.11, 7.12, 7.13**

  - [ ]* 7.11 Write the export-projection property test in `tests/property/test_property_22_export.py`
    - **Property 22: Export is an exact deterministic projection of the selection**
    - Generate reconciled populations/selections and assert exact row multiset, columns, values, hazard semantics, modeled cost, and county/hazard/score/ID order.
    - **Validates: Requirements 7.16, 7.17, 7.18, 7.19, 7.21**

  - [ ]* 7.12 Write concrete view-helper tests in `tests/unit/test_views.py`
    - Cover initial budget, exact section labels/order metadata, coordinate boundaries, equal score/ADT encodings, zero-baseline text, missing detail cells, FHWA link association, exact CSV bytes/columns, demo indicator placement hooks, and required Methods/provenance copy.
    - _Requirements: 1.15, 6.1, 6.2, 6.3, 6.4, 6.13, 6.14, 6.15, 6.16, 6.17, 6.18, 6.19, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 7.11, 7.12, 7.13, 7.16, 7.17, 7.18, 7.19, 7.20, 7.21, 7.22, 7.23, 7.24, 7.25, 7.26, 7.27, 7.28_

- [x] 8. Replace the stale Streamlit composition root and enforce failure dependency closure
  - [x] 8.1 Rewrite `app/app.py` as the production composition root with repository-root bootstrap
    - Derive and insert the repository root before absolute `app`, `model`, and `planner` imports; remove bare local imports, automatic stubs, data-path fallback, and repeated planner calls.
    - Resolve mode before data access, construct exactly one provider branch, validate budget before selecting once statewide, collect independent outcomes, build one immutable view context, and render headline/Map then Budget, Backtest, Bridge detail, Hazards, Export, and Methods in order.
    - Render each safe failure before its dependent region while preserving independent validated output, and keep production mode active through production failures.
    - _Requirements: 1.4, 1.6, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 6.1, 6.2, 6.6, 6.7, 6.8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.16, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 8.2 Write the dependency-closure property test in `tests/property/test_property_23_failure_visibility.py`
    - **Property 23: Failure visibility follows dependency closure**
    - Generate failures at each declared component boundary and assert errors precede and suppress exactly transitive dependents while unrelated validated branches remain visible.
    - **Validates: Requirements 2.30, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 11.13**

  - [ ]* 8.3 Write the caller-directory independence property test in `tests/property/test_property_25_repository_paths.py`
    - **Property 25: Repository paths are independent of caller working directory**
    - Execute import/path helpers from generated foreign working directories and assert identical repository `model`, `planner`, and `bridges/panel.parquet` identities without substitutes.
    - **Validates: Requirements 9.3, 9.4, 9.5**

  - [ ]* 8.4 Write composition, failure-scope, and path integration tests in `tests/integration/test_app_composition.py`
    - Inject each critical failure and assert exact public fields/logging, stale-cache eviction, same-mode recovery, one statewide selection, independent branch preservation, production-only real outputs, and imports/assets from repository-relative paths.
    - _Requirements: 1.9, 1.10, 1.12, 1.13, 1.14, 2.30, 6.6, 6.7, 6.8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12, 8.13, 8.14, 8.15, 8.16, 8.17, 8.18, 9.2, 9.3, 9.4, 9.5, 11.13_

- [x] 9. Add exact Python dependency, runtime, and deployment artifacts
  - [x] 9.1 Create the Python 3.11 declaration and exact direct manifests
    - Add `.python-version` with the single supported major/minor and create `requirements/runtime-direct.txt` plus `requirements/test-direct.txt` with exact `==` pins and strict runtime/test separation.
    - Include every directly imported production dependency in runtime, include pytest/Hypothesis/AppTest test tooling only in test, and keep `pytest` out of runtime.
    - _Requirements: 9.1, 10.1, 10.2, 10.3, 10.7, 10.8, 10.9, 10.10, 10.11, 10.12, 10.13_

  - [x] 9.2 Replace root dependency files with checked-in resolved locks
    - Generate Python-3.11/Linux `requirements.txt` as the exact hashed runtime lock and `requirements-test.txt` as the exact hashed runtime-plus-test lock, including every transitive distribution without open ranges.
    - Ensure Streamlit Community Cloud can install the root runtime lock and CI can install the test overlay without resolving undeclared ranges.
    - _Requirements: 9.1, 9.2, 10.4, 10.5, 10.6, 10.10, 10.11, 10.14, 10.15, 10.17_

  - [x] 9.3 Create machine-checkable dependency validation in `scripts/validate_dependencies.py`
    - Reject non-exact direct/resolved entries, missing hashes, role drift, unrepresented installed versions, and incompatible installed graphs before tests or smoke checks.
    - Validate required imports for Streamlit, pandas, PyArrow, PyDeck, NumPy, scikit-learn, `model`, `planner`, and Backtest_Engine exports.
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10, 10.11, 10.12, 10.13, 10.14, 10.15, 10.16, 10.17_

  - [x] 9.4 Create version-controlled local and Community Cloud deployment guidance in `README.md`
    - Record separate exact runtime/test installation commands, repository-root launch and test/smoke commands, `app/app.py`, Python 3.11, deployment repository/branch placeholders, no-secret setup, required local asset, and no generation/download behavior.
    - Document exact mode values/default/demo indicator/no-fallback failure behavior and concrete Community Cloud entry-point/Python settings.
    - _Requirements: 9.2, 9.5, 9.6, 9.7, 9.10, 9.11, 9.12, 9.13, 9.14, 9.15, 9.16, 9.17, 9.18, 9.19, 9.20, 10.13_

  - [ ]* 9.5 Write repository artifact and dependency contract tests in `tests/repository/test_deployment_artifacts.py`
    - Parse runtime declarations, direct manifests, resolved locks, deployment commands/settings, package/assets, and guide values; assert exact pins, role separation, consistent Python version, and all required version-controlled paths.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.10, 9.11, 9.12, 9.13, 9.14, 9.15, 9.16, 9.17, 9.18, 9.19, 9.20, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.12, 10.13, 10.17_

- [x] 10. Add bounded AppTest coverage and GitHub Actions validation
  - [x] 10.1 Create the one-shot production smoke runner in `scripts/smoke_app.py`
    - Run `AppTest.from_file("app/app.py")` once in exact production mode with a 120-second limit and blocked unexpected network access.
    - Assert the real dataset/model/planner/backtest initialize, production source label/headline/map render, no uncaught exception occurs, and demo indicator/provenance are absent.
    - _Requirements: 9.8, 9.9, 11.14, 11.15, 11.16, 11.17, 11.18, 11.23, 11.24, 11.25_

  - [ ]* 10.2 Write bounded smoke-runner tests in `tests/app/test_smoke_contract.py`
    - Verify one production AppTest invocation, local asset usage, required initial-render assertions, demo absence, no external network calls, failure propagation, and timeout enforcement.
    - _Requirements: 9.5, 9.6, 9.7, 9.8, 9.9, 11.14, 11.15, 11.16, 11.17, 11.18, 11.24, 11.25_

  - [ ]* 10.3 Write Streamlit interaction and failure AppTests in `tests/app/test_app_modes_and_state.py` and `tests/app/test_app_failures.py`
    - Cover absent/production/demo/invalid/case-mismatched modes, opposite-mode deactivation, indicator placement at every data view/download, production provenance, real-result rendering, budget reruns, county-filter independence, section order, and no demo fallback.
    - Inject each critical failure to verify exact safe fields, dependency-scoped suppression, independent-output preservation, one log event, stale invalidation, and validated recovery.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.11, 6.12, 7.20, 7.28, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12, 8.13, 8.14, 8.15, 8.16, 8.17, 8.18, 11.12, 11.13_

  - [x] 10.4 Create `.github/workflows/validate.yml`
    - Trigger on every pull request and push to the configured shared branch; use Ubuntu and the checked-in Python 3.11 declaration.
    - Validate manifest syntax, install the exact runtime lock before the test lock, validate versions/graph/imports, invoke pytest exactly once in non-watch mode, then invoke the smoke runner exactly once with production mode and bounded job/runtime timeouts.
    - _Requirements: 9.1, 9.20, 10.13, 10.14, 10.15, 10.16, 10.17, 11.19, 11.20, 11.21, 11.22, 11.23, 11.24, 11.25_

  - [ ]* 10.5 Write the CI workflow contract test in `tests/repository/test_validation_workflow.py`
    - Parse the workflow and assert triggers, Python-version source, install order, dependency gate ordering, exactly one non-watch pytest invocation, exactly one production smoke invocation, and fail-fast status propagation.
    - _Requirements: 10.13, 10.14, 10.15, 10.16, 10.17, 11.19, 11.20, 11.21, 11.22, 11.23, 11.24, 11.25_

- [ ] 11. Validate the complete package in a clean-checkout-equivalent environment
  - [ ]* 11.1 Run the exact CI-equivalent validation sequence in a fresh Python 3.11 environment
    - Install the checked-in runtime lock, then the test lock; run `scripts/validate_dependencies.py`, invoke `python -m pytest` exactly once in non-watch mode, and invoke `python scripts/smoke_app.py` exactly once with `BRIDGE_APP_MODE=production`.
    - Use only version-controlled local assets, assert no generation or external network access is required, and verify the bounded initial production render from repository-root commands.
    - _Requirements: 9.2, 9.5, 9.6, 9.7, 9.8, 9.9, 9.20, 10.10, 10.11, 10.14, 10.15, 10.16, 10.17, 11.14, 11.15, 11.16, 11.17, 11.18, 11.19, 11.21, 11.22, 11.23, 11.24, 11.25_

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and may be skipped for a faster implementation pass; implementation tasks are never optional.
- Every correctness property in the approved design has its own property-test sub-task near the implementation it validates.
- The plan uses Python 3.11 and the existing Python model package selected during clarification; no language selection is needed.
- All production data, model, planner, and backtest outputs remain real and validated. Fabricated behavior is reachable only through exact explicit demo mode and is labeled at every data-derived view and download.
- Checkpoints and validation tasks use automated, non-watch commands; the plan contains no manual review, deployment clicking, commit, or push activity.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "9.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "2.1", "3.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "3.3", "4.2", "9.2"] },
    { "id": 3, "tasks": ["2.3", "7.1", "9.4"] },
    { "id": 4, "tasks": ["3.4", "5.1", "7.3"] },
    { "id": 5, "tasks": ["5.2", "7.4"] },
    { "id": 6, "tasks": ["6.1", "6.2", "7.5", "9.3"] },
    { "id": 7, "tasks": ["8.1"] },
    { "id": 8, "tasks": ["10.1"] },
    { "id": 9, "tasks": ["10.4"] },
    { "id": 10, "tasks": ["12"] }
  ]
}
```
