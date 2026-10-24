# Requirements Document

## Introduction

The Massachusetts Bridge Inspection Planner must be deployable from a clean Git checkout to Streamlit Community Cloud for final submission. The deployed application must use the repository's real Massachusetts bridge data, decline model, planner, and backtest. Demo behavior must require explicit activation and must remain visibly separate from production behavior. The release must preserve the existing map, budget planning, backtest, bridge detail, hazard, export, and methods capabilities while adding reproducible dependencies, automated validation, and user-safe failure states.

## Glossary

- **Bridge_Inspection_Planner**: The complete Streamlit decision-support application for prioritizing additional Massachusetts bridge inspections.
- **NBI**: The Federal Highway Administration National Bridge Inventory data source used by the application.
- **SNBI**: The Specification for the National Bridge Inventory that succeeds the NBI schema.
- **Runtime_Mode_Setting**: The `BRIDGE_APP_MODE` configuration value. Accepted values are `production` and `demo`.
- **Production_Mode**: The default runtime mode that uses only the Real_Dataset, Real_Model_Service, Real_Planner, and Backtest_Engine.
- **Demo_Mode**: An explicitly selected nonproduction runtime mode that may use Demo_Data and demo calculations for demonstrations or isolated tests.
- **Runtime_Mode_Controller**: The application component that validates the Runtime_Mode_Setting and selects Production_Mode or Demo_Mode.
- **Real_Dataset**: The version-controlled `bridges/panel.parquet` file containing actual Massachusetts NBI records for 2012, 2016, 2018, and 2022.
- **Demo_Data**: Fabricated or generated bridge records, including `bridges/fake.parquet`, that are permitted only in Demo_Mode.
- **Source_Schema**: The published 21-column bridge data contract: `structure_id`, `year`, `county`, `material`, `age`, `adt`, `deck`, `super`, `sub`, `worst_rating`, `insp_freq_months`, `fracture_critical`, `underwater`, `year_recon`, `rebuilt`, `lat`, `lon`, `carries`, `crosses`, `improvement_cost`, and `insp_date`.
- **Adapted_Schema**: The model-facing schema in which `structure_id` becomes `id`, `insp_freq_months` becomes `insp_freq_mo`, and `year_built` is derived as `year - age`; all other model inputs retain their source meanings.
- **Schema_Adapter**: The sole integration boundary that validates Source_Schema data, creates Adapted_Schema data for the model and planner, and maps model identifiers back to Source_Schema identifiers for presentation.
- **Adapted_Panel**: An in-memory representation of Real_Dataset produced by Schema_Adapter without modifying Real_Dataset.
- **Ranking_Year**: The 2022 NBI snapshot used for the current inspection ranking.
- **Ranking_As_Of**: April 1, 2022, represented as `2022-04-01`, and supplied to the Real_Model_Service when scoring Ranking_Year records.
- **Ranking_Population**: The set containing one record for every unique bridge identifier in Ranking_Year, independent of map-coordinate availability.
- **Real_Model_Service**: The repository `model` package implementation that initializes decline rates and implements the three-argument scoring contract `score(table, rates, asof)`.
- **Model_State**: Decline rates fitted by Real_Model_Service from a specified four-year historical interval.
- **Scored_Table**: The real model output containing exactly `id`, `p_worse`, `consequence`, `insp_cost`, `score`, and `reason`, with one row per Ranking_Population identifier.
- **Model_Cache**: The cache that reuses Real_Dataset loading, Model_State initialization, and Scored_Table computation when the relevant inputs are unchanged.
- **Real_Planner**: The version-controlled `planner` package implementation that creates budget-constrained or fixed-count selections from Scored_Table and contains no generated outcomes.
- **Active_Budget**: The dollar budget currently selected in the user interface, initially 500,000 dollars.
- **Selection**: The ordered set of unique Scored_Table identifiers returned by Real_Planner for a budget or fixed count.
- **Safety_Drop**: A change in worst condition rating where the ending rating is 4 or lower and the starting rating is 5 or higher, or where the starting rating is 4 or lower and the ending rating decreases by at least one; both ratings must be present and the ending record must not be rebuilt.
- **Baseline_Methods**: The `worst_first` method ordered by ascending `worst_rating` and the `oldest_first` method ordered by descending `age`.
- **Backtest_Engine**: The real planner component that trains on 2012 and 2016, selects using information available through 2018, and evaluates observed Safety_Drop outcomes in 2022.
- **Backtest_Result**: A table containing exactly `budget`, `method`, and `caught`, with methods `ours`, `worst_first`, and `oldest_first` for every requested budget.
- **Hazard_Bridge**: A bridge for which `underwater` or `fracture_critical` is true.
- **User_Interface**: The Streamlit presentation layer containing the headline, map, controls, charts, tables, links, and downloads.
- **Demo_Indicator**: The visible text `DEMO MODE - FABRICATED DATA` shown when Demo_Mode is active.
- **Critical_Runtime_Failure**: A missing or unreadable Real_Dataset, Source_Schema violation, production import failure, model initialization or scoring failure, planner initialization or selection failure, or invalid production output contract.
- **User_Safe_Error**: A rendered error containing the failed component, a fixed plain-language summary, and an error code that also appears in server logs.
- **Error_Handler**: The component that converts runtime failures into User_Safe_Error output and structured server logs without substituting demo results.
- **Deployment_Package**: The version-controlled repository files required to install and run the application in Streamlit Community Cloud.
- **Deployment_Guide**: Version-controlled documentation for local startup and Streamlit Community Cloud deployment.
- **Dependency_Manifest**: The version-controlled runtime and test dependency files with exact package versions.
- **Clean_Checkout**: A new clone of the submitted Git revision with no untracked project files and no preinstalled project dependencies.
- **Automated_Test_Suite**: The pytest-based unit, integration, and application smoke tests for production behavior.
- **Smoke_Check**: A single noninteractive application execution that verifies production startup and initial rendering.
- **Validation_Pipeline**: Version-controlled automation that installs dependencies and runs the Automated_Test_Suite and Smoke_Check for repository changes.

## Requirements

### Requirement 1: Exact Production and Demo Mode Isolation

**User Story:** As a submission reviewer, I want production and demo execution to be mutually exclusive, so that every unlabeled recommendation has real-data provenance.

#### Acceptance Criteria

1. WHEN Runtime_Mode_Setting is absent, THE Runtime_Mode_Controller SHALL activate Production_Mode.
2. WHEN Runtime_Mode_Setting is absent, THE Runtime_Mode_Controller SHALL keep Demo_Mode deactivated.
3. WHEN Runtime_Mode_Setting is exactly `production`, THE Runtime_Mode_Controller SHALL activate Production_Mode.
4. WHEN Runtime_Mode_Setting is exactly `production`, THE Runtime_Mode_Controller SHALL deactivate Demo_Mode before production-derived output is computed.
5. WHEN Runtime_Mode_Setting is exactly `demo`, THE Runtime_Mode_Controller SHALL activate Demo_Mode.
6. WHEN Runtime_Mode_Setting is exactly `demo`, THE Runtime_Mode_Controller SHALL deactivate Production_Mode before demo-derived output is computed.
7. THE Runtime_Mode_Controller SHALL require the explicit, case-sensitive Runtime_Mode_Setting value `demo` for every Demo_Mode activation.
8. IF Runtime_Mode_Setting is present with a value other than the exact literals `production` or `demo`, THEN THE Runtime_Mode_Controller SHALL leave Production_Mode and Demo_Mode deactivated.
9. IF Runtime_Mode_Setting is present with a value other than the exact literals `production` or `demo`, THEN THE Error_Handler SHALL render a User_Safe_Error before any data-derived output.
10. WHILE Production_Mode is active, THE Bridge_Inspection_Planner SHALL obtain data-derived values exclusively from Real_Dataset, Real_Model_Service, Real_Planner, and Backtest_Engine.
11. WHILE Production_Mode is active, THE User_Interface SHALL identify Real_Dataset, Real_Model_Service, Real_Planner, and Backtest_Engine as the active provenance.
12. WHILE Production_Mode is active, THE Runtime_Mode_Controller SHALL reject Demo_Data and demo calculation objects before data-derived output is rendered.
13. IF a production component fails to initialize or execute, THEN THE Runtime_Mode_Controller SHALL keep Demo_Mode deactivated for the failed rerun.
14. IF a production component fails to initialize or execute, THEN THE Error_Handler SHALL render the corresponding User_Safe_Error without substituting demo-derived output.
15. WHILE Demo_Mode is active, THE User_Interface SHALL display Demo_Indicator above each data-derived view and each download control.
16. THE Model_Cache SHALL include the active runtime mode in every data, model, scoring, and backtest cache identity.
17. WHEN the active runtime mode changes, THE Model_Cache SHALL serve cached values only from entries created for the newly active runtime mode.

### Requirement 2: Observable Source Validation and Lossless Adaptation

**User Story:** As an integrator, I want a strict and observable schema boundary, so that malformed source data cannot produce partial or ambiguously mapped recommendations.

#### Acceptance Criteria

1. WHEN Production_Mode loads bridge data, THE Schema_Adapter SHALL read Real_Dataset from the repository-relative path `bridges/panel.parquet`.
2. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that the column names and column order equal the 21 fields in Source_Schema.
3. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that every `year` value is an integer in the set `2012`, `2016`, `2018`, and `2022`.
4. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that each of the years `2012`, `2016`, `2018`, and `2022` has at least one record.
5. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that each `structure_id` and `year` pair occurs exactly once.
6. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that every `structure_id` is a nonempty string after surrounding whitespace is removed.
7. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that `county`, `material`, `carries`, and `crosses` contain nonempty strings.
8. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that `age` and `adt` contain finite nonnegative integers.
9. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that `insp_freq_months` contains positive integers.
10. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that each non-null `deck`, `super`, `sub`, and `worst_rating` value is an integer from `0` through `9` inclusive.
11. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that `worst_rating` equals the minimum available value among `deck`, `super`, and `sub`, or is null exactly when all three component ratings are null.
12. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that `fracture_critical`, `underwater`, and `rebuilt` contain Boolean values.
13. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that each `year_recon` value is `0` or an integer from `1700` through the record's `year` inclusive.
14. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that each non-null `improvement_cost` value is finite and nonnegative.
15. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that each non-null `lat` value is finite and from `41.0` through `43.0` degrees inclusive.
16. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that each non-null `lon` value is finite and from `-74.0` through `-69.5` degrees inclusive.
17. WHEN Real_Dataset is read, THE Schema_Adapter SHALL verify that every `insp_date` value is a valid month-start datetime no later than April 1 of the record's `year`.
18. WHEN Source_Schema validation succeeds, THE Schema_Adapter SHALL rename `structure_id` to `id` at the model-facing boundary.
19. WHEN Source_Schema validation succeeds, THE Schema_Adapter SHALL rename `insp_freq_months` to `insp_freq_mo` at the model-facing boundary.
20. WHEN Source_Schema validation succeeds, THE Schema_Adapter SHALL derive each `year_built` value as the corresponding `year` minus `age`.
21. THE Schema_Adapter SHALL preserve `improvement_cost` as a repair-cost value without mapping `improvement_cost` to `insp_cost`.
22. THE Schema_Adapter SHALL preserve all source values other than the defined Adapted_Schema renames and `year_built` derivation.
23. THE Schema_Adapter SHALL preserve Real_Dataset byte content on disk during validation and adaptation.
24. THE Schema_Adapter SHALL preserve the source-facing in-memory Real_Dataset values and column names during validation and adaptation.
25. WHEN Adapted_Panel is created, THE Schema_Adapter SHALL preserve the Real_Dataset row count.
26. WHEN Adapted_Panel is created, THE Schema_Adapter SHALL preserve a one-to-one correspondence between every source `structure_id` and adapted `id` within each year.
27. WHEN Scored_Table is reconciled for presentation, THE Schema_Adapter SHALL match each `Scored_Table.id` to exactly one Ranking_Year `structure_id`.
28. IF identifier reconciliation finds a missing, duplicate, or unexpected identifier, THEN THE Error_Handler SHALL handle the reconciliation as a Critical_Runtime_Failure.
29. IF any Source_Schema validation fails, THEN THE Schema_Adapter SHALL return no Adapted_Panel for the failed input.
30. IF any Source_Schema validation or identifier reconciliation fails, THEN THE Error_Handler SHALL suppress every score, Selection, metric, chart, table, and export that depends on the failed input.

### Requirement 3: Bounded Real-Model Fitting, Scoring, and Caching

**User Story:** As a bridge planner, I want a completely bounded real-model run with validated outputs, so that the statewide ranking is reproducible and covers every 2022 bridge.

#### Acceptance Criteria

1. WHEN Production_Mode initializes the current ranking, THE Real_Model_Service SHALL fit Model_State with `2018` as the interval start and `2022` as the interval end.
2. WHILE Model_State is fitted for the current ranking, THE Real_Model_Service SHALL use records from years `2018` and `2022` exclusively.
3. WHILE Model_State is fitted for the current ranking, THE Real_Model_Service SHALL form training pairs only from identifiers present in both `2018` and `2022`.
4. WHILE Model_State is fitted for the current ranking, THE Real_Model_Service SHALL exclude a training pair whose 2022 ending record has `rebuilt` equal to true.
5. WHILE Model_State is fitted for the current ranking, THE Real_Model_Service SHALL exclude a training pair whose starting or ending `worst_rating` is null.
6. WHEN Model_State is available, THE Real_Model_Service SHALL call `score(table, rates, asof)` with the complete adapted Ranking_Year table, Model_State, and Ranking_As_Of respectively.
7. THE Real_Model_Service SHALL represent Ranking_As_Of as the date `2022-04-01`.
8. THE Real_Model_Service SHALL include every unique Ranking_Year identifier in Ranking_Population regardless of null or out-of-range map coordinates.
9. WHEN scoring succeeds, THE Real_Model_Service SHALL return Scored_Table columns in the exact order `id`, `p_worse`, `consequence`, `insp_cost`, `score`, and `reason`.
10. WHEN scoring succeeds, THE Real_Model_Service SHALL return exactly one Scored_Table row for every Ranking_Population identifier.
11. WHEN scoring succeeds, THE Real_Model_Service SHALL return no Scored_Table identifier outside Ranking_Population.
12. THE Real_Model_Service SHALL constrain each `p_worse` value to a finite number from `0` through `1` inclusive.
13. THE Real_Model_Service SHALL produce a positive finite `consequence` value for each scored identifier.
14. THE Real_Model_Service SHALL produce a positive finite `insp_cost` value for each scored identifier.
15. THE Real_Model_Service SHALL calculate each `score` as `p_worse * consequence / insp_cost`.
16. THE Real_Model_Service SHALL produce a finite nonnegative `score` value for each scored identifier.
17. THE Real_Model_Service SHALL produce a nonempty string `reason` for each scored identifier.
18. WHEN the source record has no valid per-bridge inspection cost, THE Real_Model_Service SHALL calculate `insp_cost` as a modeled inspection-cost estimate without using `improvement_cost`.
19. WHEN `insp_cost` is a modeled inspection-cost estimate, THE Real_Model_Service SHALL include the text `modeled inspection-cost estimate` in `reason`.
20. THE Model_Cache SHALL key current Model_State entries by Real_Dataset content, the `2018` start year, the `2022` end year, model configuration, and Production_Mode.
21. THE Model_Cache SHALL key current Scored_Table entries by Ranking_Population content, Model_State identity, Ranking_As_Of, scoring configuration, and Production_Mode.
22. WHEN all inputs in a Model_State cache identity are unchanged, THE Model_Cache SHALL reuse the validated Model_State without another fit operation.
23. WHEN all inputs in a Scored_Table cache identity are unchanged, THE Model_Cache SHALL reuse the validated Scored_Table without another scoring operation.
24. WHEN any input in a model or scoring cache identity changes, THE Model_Cache SHALL compute and validate a new value before rendering dependent output.
25. IF Model_State initialization or Scored_Table computation fails validation, THEN THE Model_Cache SHALL keep the failed value absent from the cache.
26. IF a newly computed value fails validation for an existing cache identity, THEN THE Model_Cache SHALL invalidate the prior value for that identity before dependent output is rendered.
27. IF Scored_Table has an incorrect column, missing identifier, duplicate identifier, unexpected identifier, null output field, invalid numeric value, or empty `reason`, THEN THE Error_Handler SHALL handle the result as a Critical_Runtime_Failure.

### Requirement 4: Deterministic Budget and Fixed-Count Planning

**User Story:** As a transportation decision-maker, I want deterministic and validated selection modes, so that every recommendation is reproducible, unique, and within its requested constraint.

#### Acceptance Criteria

1. THE Real_Planner SHALL expose the production selection implementation from the version-controlled `planner` package.
2. WHEN Real_Planner receives Scored_Table, THE Real_Planner SHALL validate the exact Scored_Table contract before ordering candidates.
3. THE Real_Planner SHALL order candidates by descending `score` and then ascending string `id`.
4. WHEN exactly one selection mode is requested, THE Real_Planner SHALL apply the requested budget-constrained mode or fixed-count mode.
5. IF budget-constrained mode and fixed-count mode are both requested or both omitted, THEN THE Real_Planner SHALL return a validation error without a Selection.
6. IF a budget is nonnumeric, negative, infinite, or not a number, THEN THE Real_Planner SHALL return a validation error before ordering candidates.
7. WHEN budget-constrained mode receives a finite nonnegative budget, THE Real_Planner SHALL evaluate candidates in deterministic planner order.
8. WHEN a candidate's positive `insp_cost` is less than or equal to the remaining budget, THE Real_Planner SHALL append the candidate identifier to Selection.
9. WHEN a candidate's `insp_cost` exceeds the remaining budget, THE Real_Planner SHALL continue evaluating later candidates that can fit within the remaining budget.
10. WHEN budget-constrained mode receives a budget of zero, THE Real_Planner SHALL return an empty Selection.
11. THE Real_Planner SHALL keep the sum of selected `insp_cost` values less than or equal to the requested budget.
12. IF a fixed count is nonnumeric, nonintegral, negative, infinite, or not a number, THEN THE Real_Planner SHALL return a validation error before ordering candidates.
13. WHERE fixed-count mode receives a nonnegative integer `N`, THE Real_Planner SHALL return the first `min(N, Scored_Table row count)` candidates in deterministic planner order.
14. WHERE fixed-count mode receives `N` equal to zero, THE Real_Planner SHALL return an empty Selection.
15. WHERE fixed-count mode receives `N` greater than the Scored_Table row count, THE Real_Planner SHALL return every Scored_Table identifier exactly once.
16. THE Real_Planner SHALL preserve deterministic planner order in Selection.
17. THE Real_Planner SHALL include each selected identifier exactly once.
18. THE Real_Planner SHALL include only identifiers present in Scored_Table.
19. WHEN Selection is created, THE Bridge_Inspection_Planner SHALL calculate selected total cost as the exact sum of selected `insp_cost` values.
20. WHEN Selection is created, THE Bridge_Inspection_Planner SHALL calculate expected Safety_Drop coverage as the exact sum of selected `p_worse` values.
21. WHEN Selection is created, THE Bridge_Inspection_Planner SHALL calculate selected bridge count as the number of identifiers in Selection.

### Requirement 5: Leakage-Free Backtest and Measured Baselines

**User Story:** As a submission reviewer, I want a time-bounded comparison against deterministic baselines, so that displayed performance represents observed outcomes without future-data leakage or forced superiority.

#### Acceptance Criteria

1. THE Backtest_Engine SHALL expose the production backtest implementation from the version-controlled `planner` package.
2. WHEN a backtest starts, THE Backtest_Engine SHALL fit backtest Model_State with `2012` as the interval start and `2016` as the interval end.
3. WHILE backtest Model_State is fitted, THE Backtest_Engine SHALL use records from years `2012` and `2016` exclusively.
4. WHILE backtest Model_State is fitted, THE Backtest_Engine SHALL exclude a training pair whose 2016 ending record has `rebuilt` equal to true.
5. WHILE backtest Model_State is fitted, THE Backtest_Engine SHALL exclude a training pair whose starting or ending `worst_rating` is null.
6. WHEN backtest Model_State is available, THE Backtest_Engine SHALL call `score(table, rates, asof)` with the complete adapted 2018 table, backtest Model_State, and `2018-04-01` respectively.
7. WHILE 2018 candidates are scored and selected, THE Backtest_Engine SHALL use fields dated 2018 or earlier exclusively.
8. THE Backtest_Engine SHALL construct the common candidate set from unique identifiers in the 2018 table before reading 2022 outcome fields.
9. THE Backtest_Engine SHALL use the same common candidate identifiers, 2018 modeled inspection costs, and requested budgets for `ours`, `worst_first`, and `oldest_first`.
10. WHEN the `ours` method selects candidates, THE Backtest_Engine SHALL use the deterministic budget-constrained Real_Planner order.
11. WHEN the `worst_first` method orders candidates, THE Backtest_Engine SHALL sort non-null 2018 `worst_rating` values in ascending order, place null ratings after numeric ratings, and break ties by ascending string `id`.
12. WHEN the `oldest_first` method orders candidates, THE Backtest_Engine SHALL sort 2018 `age` values in descending order and break ties by ascending string `id`.
13. WHEN a baseline method selects candidates, THE Backtest_Engine SHALL apply the same affordability rule used by budget-constrained Real_Planner.
14. WHEN a 2018 Selection is evaluated, THE Backtest_Engine SHALL match each selected identifier to at most one 2018 record and at most one 2022 record.
15. WHEN a selected identifier has a missing 2018 record, missing 2022 record, null 2018 `worst_rating`, null 2022 `worst_rating`, or true 2022 `rebuilt` value, THE Backtest_Engine SHALL exclude the identifier from outcome evaluation.
16. WHEN a selected identifier is eligible for outcome evaluation, THE Backtest_Engine SHALL classify Safety_Drop using the 2018 starting rating and 2022 ending rating exactly as defined in the Glossary.
17. THE Backtest_Engine SHALL calculate `caught` as the integer count of selected eligible identifiers classified as Safety_Drop.
18. THE Backtest_Engine SHALL preserve measured `caught` values without modifying values to make `ours` equal or superior to a Baseline_Methods result.
19. WHEN a backtest completes, THE Backtest_Engine SHALL return Backtest_Result columns in the exact order `budget`, `method`, and `caught`.
20. WHEN a backtest completes, THE Backtest_Engine SHALL return exactly one Backtest_Result row for each distinct requested budget and each method in `ours`, `worst_first`, and `oldest_first`.
21. WHEN a backtest completes, THE Backtest_Engine SHALL order Backtest_Result by ascending `budget` and then by the method order `ours`, `worst_first`, and `oldest_first`.
22. THE Backtest_Engine SHALL constrain every `caught` value to a nonnegative integer no greater than the corresponding selected eligible bridge count.
23. THE Model_Cache SHALL key Backtest_Result entries by Real_Dataset content, requested budgets, backtest interval, as-of date, model configuration, planner configuration, and Production_Mode.
24. WHEN every Backtest_Result cache input is unchanged, THE Model_Cache SHALL reuse the validated Backtest_Result without another backtest execution.
25. WHEN any Backtest_Result cache input changes, THE Model_Cache SHALL compute and validate a new Backtest_Result before rendering backtest output.
26. IF fit, scoring, candidate construction, cost assignment, ordering, or selection uses a 2022 outcome field, THEN THE Error_Handler SHALL reject the Backtest_Result before rendering.
27. IF Backtest_Result violates its schema, row cardinality, method set, budget set, ordering, or `caught` bounds, THEN THE Error_Handler SHALL reject the Backtest_Result before rendering.
28. IF Backtest_Result fails validation, THEN THE Model_Cache SHALL keep the invalid Backtest_Result absent from the cache.

### Requirement 6: Shared Statewide Selection, Summary, and Map

**User Story:** As a transportation decision-maker, I want every view synchronized to one statewide budget selection, so that filters and map eligibility do not change the recommended plan.

#### Acceptance Criteria

1. THE User_Interface SHALL provide sections labeled `Map`, `Budget`, `Backtest`, `Bridge detail`, `Hazards`, `Export`, and `Methods`.
2. WHEN production initialization succeeds, THE User_Interface SHALL render the current-ranking headline and statewide map before secondary analysis sections.
3. THE User_Interface SHALL initialize Active_Budget to `500000` dollars.
4. THE User_Interface SHALL permit Active_Budget values from `100000` through `3000000` dollars inclusive in increments of `50000` dollars.
5. IF a requested Active_Budget is outside the permitted values, THEN THE User_Interface SHALL retain the most recent permitted Active_Budget.
6. WHEN Active_Budget changes to a permitted value, THE Bridge_Inspection_Planner SHALL create one statewide Selection from the complete Scored_Table through Real_Planner.
7. WHILE an Active_Budget is unchanged, THE User_Interface SHALL use the same statewide Selection for the headline, map, Budget, Bridge detail, Hazards, Export, and selection-dependent Methods content.
8. WHEN Selection changes, THE User_Interface SHALL update every selection-dependent view during the same Streamlit rerun.
9. THE User_Interface SHALL display Active_Budget, selected bridge count, Ranking_Population count, selected total cost, and expected Safety_Drop coverage in the current-ranking summary.
10. THE User_Interface SHALL derive each current-ranking summary value from the active statewide Selection and complete Ranking_Population.
11. WHEN a county filter is selected, THE User_Interface SHALL limit displayed map records to the selected county.
12. WHEN a county filter is selected, THE Bridge_Inspection_Planner SHALL preserve the statewide Selection without recomputing Selection from the filtered map records.
13. THE User_Interface SHALL classify a Ranking_Year record as map-eligible when `lat` is finite from `41.0` through `43.0` inclusive and `lon` is finite from `-74.0` through `-69.5` inclusive.
14. WHEN the statewide map renders without a county filter, THE User_Interface SHALL plot exactly one marker for every map-eligible Ranking_Year record.
15. WHEN the statewide map renders with a county filter, THE User_Interface SHALL plot exactly one marker for every map-eligible Ranking_Year record in the selected county.
16. WHEN a plotted record belongs to Selection, THE User_Interface SHALL render the marker in red.
17. WHEN a plotted record does not belong to Selection, THE User_Interface SHALL map `score` to a continuous sequential color scale whose normalized color-scale value is nondecreasing as `score` increases.
18. THE User_Interface SHALL map nonnegative `adt` to marker size through a nondecreasing function that gives equal `adt` values equal marker sizes.
19. WHEN a map marker is inspected, THE User_Interface SHALL display `structure_id`, `carries`, `crosses`, `county`, and `score` from the marker's reconciled record.
20. WHEN a Ranking_Population record is not map-eligible, THE Bridge_Inspection_Planner SHALL retain the record in scoring, Selection, summary metrics, tables, detail choices, hazards, and exports.

### Requirement 7: Exact Evidence, Detail, Hazard, Export, and Provenance Views

**User Story:** As a transportation decision-maker, I want exact supporting evidence and provenance for the active plan, so that I can review, interpret, and export measured results.

#### Acceptance Criteria

1. WHEN the Budget section renders, THE User_Interface SHALL display selected bridge count, selected total cost, expected Safety_Drop coverage, and Active_Budget from the active statewide Selection.
2. WHEN the Budget section renders a selected-bridge table, THE User_Interface SHALL use source values from Ranking_Year and model values from Scored_Table without substituting generated values.
3. WHEN the Budget or Hazards section orders selected bridges, THE User_Interface SHALL place Hazard_Bridge records before non-hazard records.
4. WHEN the Budget or Hazards section orders records within the same hazard group, THE User_Interface SHALL sort by descending `score` and then ascending `structure_id`.
5. WHEN the Backtest section renders, THE User_Interface SHALL plot measured `caught` against `budget` for `ours`, `worst_first`, and `oldest_first` directly from Backtest_Result.
6. WHEN the Backtest section compares methods, THE User_Interface SHALL compare `caught` values from the same budget.
7. WHEN `ours` exceeds a nonzero baseline at a compared budget, THE User_Interface SHALL calculate any displayed difference as `ours caught - baseline caught` and any displayed ratio as `ours caught / baseline caught`.
8. WHEN `ours` exceeds a zero baseline at a compared budget, THE User_Interface SHALL display the exact measured counts and absolute difference without displaying a ratio.
9. WHEN `ours` is equal to or below a baseline at a compared budget, THE User_Interface SHALL display the exact measured counts without the phrase `more declines caught`.
10. WHEN a bridge is chosen in Bridge detail, THE User_Interface SHALL display available `deck`, `super`, and `sub` ratings for `2012`, `2016`, `2018`, and `2022` in ascending year order.
11. IF a requested bridge-year or rating value is absent, THEN THE User_Interface SHALL display `Unavailable` for the corresponding rating without generating a replacement value.
12. WHEN a bridge is chosen in Bridge detail, THE User_Interface SHALL display the Scored_Table `reason`, `carries`, `crosses`, and `county` associated with the selected `structure_id`.
13. WHEN a bridge is chosen in Bridge detail, THE User_Interface SHALL provide an FHWA InfoBridge link associated with the selected `structure_id`.
14. THE User_Interface SHALL classify a Ranking_Year bridge as a Hazard_Bridge exactly when `underwater` is true or `fracture_critical` is true.
15. WHEN the Hazards section renders, THE User_Interface SHALL include exactly the Hazard_Bridge identifiers present in the active statewide Selection.
16. WHEN Export renders, THE User_Interface SHALL generate one CSV data row for every identifier in the active statewide Selection and no row for an unselected identifier.
17. WHEN Export generates CSV content, THE User_Interface SHALL include columns in the exact order `county`, `structure_id`, `worst_rating`, `score`, `insp_cost`, `hazard`, `carries`, `crosses`, and `reason`.
18. WHEN Export generates CSV content, THE User_Interface SHALL derive `hazard` from the exact Hazard_Bridge classification.
19. WHEN Export orders CSV rows, THE User_Interface SHALL sort by ascending `county`, descending `hazard`, descending `score`, and ascending `structure_id`.
20. WHEN Export offers a download, THE User_Interface SHALL display adjacent provenance text identifying Real_Dataset, Real_Model_Service, Real_Planner, Active_Budget, and the current Selection.
21. WHEN an `insp_cost` value is displayed in the User_Interface, THE User_Interface SHALL label the value as a modeled inspection-cost estimate rather than a quote or repair cost.
22. WHEN Export offers CSV content, THE User_Interface SHALL state adjacent to the download that `insp_cost` contains modeled inspection-cost estimates and does not contain `improvement_cost`.
23. THE User_Interface SHALL describe the NBI source, the `2018` to `2022` current-model interval, and Ranking_As_Of in Methods.
24. THE User_Interface SHALL describe the `2012` to `2016` fit, `2018` selection, and `2022` outcome timeline in Methods.
25. THE User_Interface SHALL describe Safety_Drop, rebuilt-record exclusion, and modeled inspection-cost assumptions in Methods.
26. THE User_Interface SHALL describe the Source_Schema-to-Adapted_Schema mapping and identifier reconciliation in Methods.
27. THE User_Interface SHALL describe the NBI-to-SNBI transition in Methods.
28. WHILE Production_Mode is active, THE User_Interface SHALL identify Real_Dataset as the data source, Real_Model_Service as the scoring source, Real_Planner as the selection source, and Backtest_Engine as the backtest source.

### Requirement 8: Dependency-Scoped and User-Safe Failure Handling

**User Story:** As a submission reviewer, I want failures bounded to dependent output with safe diagnostics, so that valid independent evidence remains available and stale results cannot appear current.

#### Acceptance Criteria

1. IF a Critical_Runtime_Failure occurs, THEN THE Error_Handler SHALL render a User_Safe_Error before rendering output that depends on the failed component.
2. IF Real_Dataset loading or Source_Schema validation fails, THEN THE Error_Handler SHALL suppress all data-derived headlines, maps, metrics, charts, tables, detail views, and downloads for the failed rerun.
3. IF current Model_State initialization or Scored_Table validation fails, THEN THE Error_Handler SHALL suppress current-ranking output and every output that depends on current Scored_Table.
4. IF Real_Planner initialization or current Selection fails, THEN THE Error_Handler SHALL suppress the headline, Selection highlights, selection metrics, selected tables, selected hazards, and export for the failed Selection.
5. IF Backtest_Engine or Backtest_Result fails after current ranking succeeds, THEN THE Error_Handler SHALL suppress Backtest output while preserving validated current-ranking output.
6. IF Bridge detail fails after current ranking succeeds, THEN THE Error_Handler SHALL suppress the failed detail output while preserving validated current-ranking output.
7. IF Export fails after current Selection succeeds, THEN THE Error_Handler SHALL suppress the failed download while preserving validated on-screen Selection output.
8. WHEN a component fails, THE Error_Handler SHALL preserve validated output whose dependency inputs exclude the failed component.
9. WHEN Error_Handler renders User_Safe_Error, THE Error_Handler SHALL display exactly the failed component name, the component's fixed plain-language summary, and the error code as diagnostic fields.
10. THE Error_Handler SHALL assign the same stable error code to repetitions of the same failure category.
11. WHEN Error_Handler handles one exception, THE Error_Handler SHALL emit one structured server-log event containing the User_Safe_Error code, failed component, exception class, and exception message.
12. WHEN Error_Handler emits a structured server-log event, THE Error_Handler SHALL use the same error code and failed component shown in User_Safe_Error.
13. THE Error_Handler SHALL keep stack traces, local filesystem paths, environment values, credentials, secret values, and raw data rows absent from User_Safe_Error.
14. IF a failed computation has a prior cached success for the same cache identity, THEN THE Model_Cache SHALL invalidate the prior cached value before the failure state is rendered.
15. IF a computed value fails its output contract, THEN THE Model_Cache SHALL keep the invalid value absent from the cache.
16. IF any production component fails, THEN THE Runtime_Mode_Controller SHALL preserve Production_Mode and keep Demo_Mode deactivated for that rerun.
17. WHEN a failed component runs again with valid inputs, THE Error_Handler SHALL replace the component's failure state only after the new output passes the applicable validation contract.
18. WHEN a failed component recovers, THE Model_Cache SHALL store only the newly validated output under the current cache identity.

### Requirement 9: Version-Controlled Git and Streamlit Deployment

**User Story:** As a submission maintainer, I want every deployment input encoded or documented in the repository, so that a clean checkout can render the production application without repair, generation, or hidden assets.

#### Acceptance Criteria

1. THE Deployment_Package SHALL version-control the Streamlit application, Real_Model_Service, Real_Planner, Backtest_Engine, Real_Dataset, Dependency_Manifest, Python runtime declaration, Deployment_Guide, and Validation_Pipeline.
2. WHEN runtime dependencies are installed in a Clean_Checkout, THE Deployment_Package SHALL start from the repository root with `streamlit run app/app.py`.
3. WHEN Streamlit Community Cloud launches `app/app.py`, THE Deployment_Package SHALL make the repository-root `model` and `planner` packages importable through version-controlled application or package configuration.
4. THE Deployment_Package SHALL resolve application modules and runtime data assets from repository-relative paths rather than the process working directory outside the repository.
5. WHEN Production_Mode starts, THE Deployment_Package SHALL load Real_Dataset from the version-controlled `bridges/panel.parquet` asset.
6. WHILE Production_Mode starts, THE Deployment_Package SHALL require no bridge-data generation step.
7. WHILE Production_Mode starts, THE Deployment_Package SHALL require no network download of bridge data.
8. WHEN dependencies are already installed and `streamlit run app/app.py` starts in a Clean_Checkout, THE Deployment_Package SHALL make the initial production render available within `120` seconds.
9. WHEN the initial production render succeeds, THE User_Interface SHALL include the production data-source label, current-ranking headline, and map.
10. THE Deployment_Guide SHALL identify `app/app.py` as the Streamlit Community Cloud entry point.
11. THE Deployment_Guide SHALL state the single declared Python major and minor version.
12. THE Deployment_Guide SHALL document the runtime dependency installation command.
13. THE Deployment_Guide SHALL document the test dependency installation command separately from the runtime installation command.
14. THE Deployment_Guide SHALL document `streamlit run app/app.py` as a repository-root local launch command.
15. THE Deployment_Guide SHALL document the accepted Runtime_Mode_Setting values, the Production_Mode default, explicit Demo_Mode activation, Demo_Indicator, and production failure behavior.
16. THE Deployment_Guide SHALL list every version-controlled runtime data asset required by Production_Mode.
17. THE Deployment_Guide SHALL document Streamlit Community Cloud repository and branch values as deployment-supplied placeholders rather than fixed repository-specific literals.
18. THE Deployment_Guide SHALL document the Streamlit Community Cloud entry-point and Python-version settings with concrete expected values.
19. THE Deployment_Guide SHALL document a deployment path that requires no application secrets.
20. THE Deployment_Guide SHALL document the commands used to run the Automated_Test_Suite and Smoke_Check before deployment.

### Requirement 10: Separately Installable and Exactly Pinned Dependencies

**User Story:** As a submission maintainer, I want separate, exact runtime and test dependency sets, so that production deployment remains minimal while clean-checkout validation remains reproducible.

#### Acceptance Criteria

1. THE Dependency_Manifest SHALL separate runtime dependencies and test-only dependencies into version-controlled, independently addressable files.
2. THE Dependency_Manifest SHALL pin each direct runtime dependency entry with an exact `==` version.
3. THE Dependency_Manifest SHALL pin each direct test dependency entry with an exact `==` version.
4. THE Dependency_Manifest SHALL provide a runtime lock or resolved manifest that records every transitive runtime package with an exact version.
5. THE Dependency_Manifest SHALL provide a test lock or resolved manifest that records every transitive test package with an exact version.
6. WHERE a top-level Streamlit requirements file lists direct runtime dependencies only, THE Dependency_Manifest SHALL capture transitive runtime pins in the accompanying runtime lock or resolved manifest.
7. THE Dependency_Manifest SHALL include every third-party package imported directly by production code as a direct runtime dependency.
8. THE Dependency_Manifest SHALL include `pytest` in the test-only dependency set.
9. THE Dependency_Manifest SHALL keep `pytest` absent from the direct runtime dependency set.
10. WHEN the runtime dependency set is installed by itself, THE Deployment_Package SHALL import and start the production application without test-only packages.
11. WHEN the test-only dependency set is installed over the runtime dependency set, THE Automated_Test_Suite SHALL have every package required for one non-watch test execution.
12. THE Deployment_Package SHALL declare one supported Python major and minor version in one version-controlled runtime declaration.
13. THE Deployment_Guide and Validation_Pipeline SHALL reference the same declared Python major and minor version.
14. WHEN dependency validation runs, THE Validation_Pipeline SHALL verify that every installed package represented in the applicable lock or resolved manifest matches the recorded version.
15. WHEN dependency validation runs, THE Validation_Pipeline SHALL verify that the installed dependency graph has no missing or incompatible requirement.
16. WHEN runtime import validation runs, THE Validation_Pipeline SHALL import Streamlit, pandas, PyArrow, PyDeck, NumPy, scikit-learn, Real_Model_Service, Real_Planner, and Backtest_Engine successfully.
17. IF a direct dependency is unpinned, a resolved dependency lacks an exact version, an installed version differs from its recorded version, or the dependency graph is incompatible, THEN THE Validation_Pipeline SHALL stop before pytest and Smoke_Check.

### Requirement 11: Concrete Automated Tests and Clean-Checkout Validation

**User Story:** As a submission maintainer, I want concrete automated matrices and a bounded production smoke test, so that integration, provenance, and deployment regressions fail before submission.

#### Acceptance Criteria

1. THE Automated_Test_Suite SHALL test Source_Schema acceptance with all 21 columns in exact order, all four accepted years, valid nullable ratings, valid coordinate bounds, valid booleans, valid dates, and unique identifiers.
2. THE Automated_Test_Suite SHALL test Source_Schema rejection for a missing column, extra column, reordered column, unsupported year, missing required year, duplicate key, invalid rating, invalid Boolean, invalid date, latitude outside `41.0` through `43.0`, and longitude outside `-74.0` through `-69.5`.
3. THE Automated_Test_Suite SHALL test exact field renames, `year_built` derivation, `improvement_cost` preservation, source immutability, row-count preservation, and identifier reconciliation in Schema_Adapter.
4. THE Automated_Test_Suite SHALL test the current model matrix for exclusive `2018` and `2022` fit inputs, rebuilt and null-rating exclusions, the complete 2022 scoring table, the `2022-04-01` as-of value, and retention of identifiers without map coordinates.
5. THE Automated_Test_Suite SHALL test Scored_Table acceptance and rejection for exact columns, identifier cardinality, probability bounds, finite numeric values, positive modeled costs, score arithmetic, nonempty reasons, and modeled-cost disclosure.
6. THE Automated_Test_Suite SHALL test Model_Cache reuse, input-key changes, mode isolation, invalidation, and rejection of invalid values for Model_State and Scored_Table.
7. THE Automated_Test_Suite SHALL test Real_Planner with zero budget, exact-fit budget, unaffordable leading candidates, later affordable candidates, equal-score ties, duplicate-input rejection, invalid budgets, fixed count zero, fixed count above population size, and mutually exclusive selection-mode validation.
8. THE Automated_Test_Suite SHALL test Selection affordability, deterministic order, unique identifiers, membership in Scored_Table, selected total cost, selected bridge count, and expected Safety_Drop coverage.
9. THE Automated_Test_Suite SHALL test Backtest_Engine for exclusive `2012` and `2016` fit inputs, the `2018-04-01` scoring call, 2018-only selection inputs, common candidates, common costs, common budgets, and exclusion of 2022 fields before evaluation.
10. THE Automated_Test_Suite SHALL test backtest outcome matching, missing-record exclusion, null-rating exclusion, rebuilt exclusion, both Safety_Drop branches, non-drop outcomes, exact `caught` counts, exact Backtest_Result schema, deterministic row order, and measured results without forced superiority.
11. THE Automated_Test_Suite SHALL test `worst_first` numeric ordering, null-last behavior, `oldest_first` ordering, ascending identifier tie breaks, baseline affordability, Backtest_Result cache reuse, cache invalidation, and invalid-result rejection.
12. THE Automated_Test_Suite SHALL test absent Runtime_Mode_Setting, exact `production`, exact `demo`, invalid values, case-sensitive values, mode activation, opposite-mode deactivation, Demo_Indicator placement, production provenance, and rejection of automatic demo fallback.
13. THE Automated_Test_Suite SHALL test each Critical_Runtime_Failure category for dependency-scoped suppression, exact User_Safe_Error fields, matching error codes, one structured log event, sensitive-diagnostic exclusion, stale-cache invalidation, independent-output preservation, and validated rerun recovery.
14. WHEN Smoke_Check runs, THE Smoke_Check SHALL launch the Streamlit application once in Production_Mode from the Clean_Checkout.
15. WHILE Smoke_Check runs, THE Smoke_Check SHALL load version-controlled Real_Dataset and initialize Real_Model_Service, Real_Planner, and Backtest_Engine.
16. WHEN Smoke_Check reaches the initial render, THE Smoke_Check SHALL verify the production data-source label, current-ranking headline, map, and absence of an uncaught exception.
17. WHEN Smoke_Check reaches the initial render, THE Smoke_Check SHALL verify that Demo_Indicator and Demo_Data provenance are absent.
18. WHEN Smoke_Check starts after dependencies are installed, THE Smoke_Check SHALL complete initial-render assertions within `120` seconds.
19. THE Validation_Pipeline SHALL begin validation from a Clean_Checkout of the submitted revision.
20. THE Validation_Pipeline SHALL run for every pull request and every push to each configured shared branch.
21. WHEN Validation_Pipeline runs, THE Validation_Pipeline SHALL install the exactly pinned runtime dependency set before the test-only dependency set.
22. WHEN dependency installation and validation succeed, THE Validation_Pipeline SHALL invoke pytest exactly once in non-watch mode.
23. WHEN pytest succeeds, THE Validation_Pipeline SHALL invoke Smoke_Check exactly once in noninteractive mode.
24. IF dependency installation, dependency validation, pytest, production import, Smoke_Check, or the `120`-second bound fails, THEN THE Validation_Pipeline SHALL report a failing status.
25. WHILE pytest and Smoke_Check execute after dependency installation, THE Automated_Test_Suite SHALL use local fixtures and version-controlled assets with zero external network requests.
