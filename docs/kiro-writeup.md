How We Used Kiro: Bridge Triage (Hackachusetts R6)
1. Summary
Bridge Triage ranks Massachusetts bridges by chance of a safety drop using FHWA National Bridge Inventory data for 2012, 2016, 2018, and 2022, picks which bridges get extra inspection under a dollar budget, and backtests the picks against worst-first and oldest-first. Three teammates built it in parallel, each owning one layer: data (bridges/, and later the Streamlit app in app/), the model (model/), and the planner and backtest (planner/). Kiro held the team together through one shared contract in steering, four specs under .kiro/specs/, and spec task lists that put the work that unblocked others first. All four specs, all five steering docs, and the data checks described below are in this repo.

2. Which Kiro tools we used
Kiro IDE: used for everything. The specs (requirements.md, design.md, tasks.md), the spec metadata (.config.kiro, tasks.meta.json), and the steering docs are all IDE artifacts. The bugfix spec's .config.kiro records "workflowType": "fast-task", "specType": "bugfix", which is the IDE's bugfix spec flow.
Kiro CLI, Web, and Crew: not used. The repo has no artifacts from them. Since each teammate owned one folder and worked on one branch, a single IDE workflow per layer covered what we needed.
Each layer was built on its own branch (person-a-data, person-a-app, part-b-model, person-c-backtest). The work came together on integration, merged, and final.

3. Spec driven development
We have four specs, and each one is shaped around the blockers of its layer:

bridges-data: requirements, design, and 16 tasks, all done. Task 3 and task 4 come first on purpose: "produce bridges/fake.parquet" and "push fake.parquet first to unblock B and C", meaning the model and planner layers. Real data comes after that (tasks 5 to 10), then checks and the summary. The design's "Deliverable order" section is fake.parquet, then bridges_2022.parquet, then panel.parquet.
decline-model: requirements, design, and 20 tasks, all done. The spec says outright that it "documents work already completed in model/". It came after the code, not before.
app: requirements written as numbered SHALL criteria, a design with 25 named correctness properties, and a task plan with a JSON dependency graph of 11 waves. Every required implementation task (1 to 10, plus checkpoint 12) is done. Task 11, the clean environment validation run, is still open, and the 25 optional property test tasks were not built.
flight-cost-cache-fingerprint (bugfix spec): written when merging part-b-model at commit 1f0cc7c into the app broke startup with MODEL-FIT-001. The cause was app/services.py importing COST_BASE from model.score after the model layer moved costs into model/flight_cost.py. Only task 1.1 (preserve the model layer in the merge) is done. The regression test and the fingerprint repair are still open.
The shared contract is the 21 column panel schema in bridges/README.md, from structure_id through insp_date. It began as 20 columns. bridges-data task 16 added insp_date as column 21 and states "Keep existing column positions unchanged", so the model and planner code did not have to change. The app spec repeats the same 21 columns word for word as Source_Schema. The contract fixes two definitions. Rebuilt means that, compared with the bridge's previous appearance, a deck, super, or sub rating went up or year_recon changed, and "a first appearance is never rebuilt". Drop has two layers: the data summary counts any fall of 1 or more in worst_rating, while model/target.py (decline-model task 6) defines a safety drop as crossing into poor (4 or below) or worsening while already poor. Both the model and the backtest use that one definition.

Spec work and vibe coding compared, from our own history:

The spec first bridges-data ordering worked. The model and planner layers started against fake.parquet, which has the same 21 columns in the same order as the real panel, before the real data was clean.
Spec work also made fixes easy to trace. bridges-data task 11 is recorded as its own fix: a first appearance is never rebuilt, and "2012 count is now 0".
The model was vibe coded first and the spec written afterwards. That produced a correct spec, but two bugs only showed up along the way: task 8 (a missing insp_date fallback that could inflate rank) and task 9 (crossing detection needed word boundaries, so "CHERRY ST" is not read as rail).
The cost of skipping a spec at a boundary shows in the bugfix spec. The model layer renamed its cost constants, the app still imported the old ones, and production startup failed.
4. Agent hooks
Not used. The repo has no .kiro/hooks/ folder. The automation we did set up is CI rather than a Kiro hook: .github/workflows/validate.yml (from app task 10.4) runs on every pull request and every push to the shared branch. It installs the hashed lock files, runs scripts/validate_dependencies.py, runs pytest once, and then runs the production smoke test scripts/smoke_app.py. With one owner per folder and CI covering every push, we did not add save time hooks.

5. Steering docs
Five docs are in .kiro/steering/:

product.md (always included): the problem, the score formula, the backtest plan, and the non-negotiable rules. Those rules are real data only, no lookahead, NBI "N" means not applicable rather than zero, every score gets a one line reason, and costs are labeled as estimates.
contract.md (always included): the handoff contract. It says which layer owns which folder, gives the data to model schema, the model to planner score output columns and formula, and the backtest rule (fit 2012 to 2016, pick with 2018, check 2022). It also says that "Changing either column list is a contract change, announce it". It was the first draft of the contract, and the schema later moved to bridges/README.md, which now owns it.
structure.md (always included): a repo map and ownership table, plus rules such as "Don't reach across parts into internal helpers".
tech.md (always included): the stack, the FHWA URL pattern, reading raw NBI as latin-1 strings so "N" and blanks survive, and matching NBI columns by suffix.
frontend-design.md (manual inclusion): concrete design rules you can check. Examples are one easing curve cubic-bezier(0.23, 1, 0.32, 1), all motion under 300ms, animating only transform and opacity, tabular-nums on every number, red as the only accent for risk, all CSS in one block in app/style.py, and no em or en dashes.
Two approaches helped most. Handoff steering with folder ownership (contract.md plus structure.md) let three teammates run Kiro at the same time without touching each other's folders. The front end rules were written as measurable values rather than adjectives, so Kiro's output can be checked against them line by line.

6. MCP
Not used. There is no .kiro/settings/mcp.json in the repo. We pulled the data straight from the FHWA delimited files (fetch_years.py, bridges/loader.py), and the app links each bridge to its FHWA InfoBridge page (app task 7.4), so we never needed a fetch or GitHub server during the build.

7. Property based testing
Partly used. The app design defines 25 correctness properties. Examples are Property 10 (budget selection is deterministic, unique, and affordable), Property 13 (historical planning cannot observe future outcomes), and Property 17 (the map is a projection, not a population filter). Hypothesis is pinned in the test lock (hypothesis 6.170.9, listed in tests/unit/test_validate_dependencies.py), and tests/property/ was created for these tests. The property test tasks were marked optional, though, and on this branch tests/property/ holds only __init__.py. The same rules are covered by example based tests in tests/unit/ (for example test_planner_contracts.py, test_historical_planning.py, test_backtest_outcomes.py) and by tests/test_model.py. The project documentation lists planner invariants in planner/properties.py on the final branch. Because no Hypothesis run is recorded, we make no claim about bugs it found.

8. Kiro powers and third party tools
Powers: none installed or used.

The third party libraries we relied on are pandas, NumPy, PyArrow, scikit-learn, Streamlit, PyDeck, and matplotlib, plus pytest and Hypothesis for tests. All of them are pinned with hashes in requirements.txt and requirements-test.txt.

9. Data verification Kiro ran
Sources: four FHWA delimited Massachusetts files (data/MA12.txt, MA16.txt, MA18.txt, MA22.txt), each reported OK per URL (bridges-data task 5). The 2022 header and item positions were checked before parsing (task 6).
Coordinates: bridges/loader.py decodes NBI's packed degrees, minutes, and seconds, where latitude is 8 digits and longitude is 9 digits (DDD MM SS ss). Massachusetts longitudes come out negative, as contract.md requires.
Panel: bridges/panel.parquet has 20,757 rows covering 5,506 bridges. Per year, from bridges/SUMMARY.md, that is 5,120 (2012), 5,171 (2016), 5,215 (2018), and 5,251 (2022). The raw 2022 snapshot bridges/data.csv has 5,252 rows. The panel join drops rare colliding (structure_id, year) duplicates so the key stays unique.
Year to year matching: build_panel printed consecutive year match rates of 97.6% (2012 to 2016), 99.1% (2016 to 2018), and 98.3% (2018 to 2022).
run_checks (task 12) passed: no duplicate (structure_id, year), lat and lon inside Massachusetts, ratings from 0 to 9 or null, no negative ages, and row counts reported per year.
Missingness kept honest: null worst_rating counts are 299, 308, 324, and 324 per year (culverts and similar). Rebuilt counts are 0, 518, 325, and 394, and the 0 for 2012 confirms the first appearance rule.
Cross checks across layers: the scored files have 4,891 rows (2018) and 4,927 rows (2022). Those match exactly the number of rated bridges in those years in SUMMARY.md, so only null rating rows were skipped. The backtest pool of 4,420 plus its exclusions (393 rebuilt, 310 null in 2018, 91 missing in 2022, 1 null in 2022) adds up to the 5,215 bridges in 2018.
10. What we would do next
SNBI schema mapping: FHWA is moving from NBI to the Specification for the National Bridge Inventory, so 2026 data needs a mapping onto our 21 column contract before the model can be refit. The app's Methods tab already describes the NBI to SNBI transition (app task 7.5).
Wire app/app.py to the real pipeline. On final it still falls back to app/stubs.py because it imports planner.pick and calls score with one argument. The production pieces it needs (app/config.py, app/errors.py, app/services.py) already exist.
Finish the open flight-cost-cache-fingerprint tasks (regression test, fingerprint repair, full validation) and app task 11 (clean environment run).
Clear the dependency gate by declaring matplotlib for planner/backtest.py or moving the plotting import out of it.
Build the 25 property tests from the app design with Hypothesis, and add a save hook that runs pytest on model/, planner/, and bridges/.
The backtest is honest but modest: the model beats both baselines at budgets of 1% to 5% and beats oldest-first at every budget, but worst-first edges it at 10% to 20%, with only 137 decliners in the pool. More years of data would tighten the confidence intervals.