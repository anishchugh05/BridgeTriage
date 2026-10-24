"""Correctness properties for the backtest harness (R14).

Executable mirrors of requirements.md R14. Conventions (R14 preamble,
design Implementation Note 10):
- ALL test functions take NO arguments; every parameter (dataset counts,
  seeds, budgets) comes from backtest's Module Constants or the module
  constants below.
- random_avg runs on RANDOM_SEEDS_TEST (20 seeds) inside property tests,
  not the 200 production seeds (RANDOM_SEEDS). run_backtest reads the
  module-level RANDOM_SEEDS at call time, so _run_backtest_test_seeds
  swaps it for the duration of a call (and always restores it).

Run: python -m pytest planner/properties.py -q
"""

import numpy as np

from planner import backtest
from planner.backtest import (
    RANDOM_SEEDS_TEST,
    REL_TOL,
    apply_exclusion_and_label,
    load_and_validate,
    make_fake,
    pick,
    run_backtest,
    score_oldest_first,
    score_oracle,
    score_random,
    score_worst_first,
)

# ---------------------------------------------------------------------------
# Property-test constants (not part of backtest's Module Constants)
# ---------------------------------------------------------------------------

COUNTY_FEE = 5.0        # same flat fee used throughout the backtest demos
FAKE_N = 400            # bridges per make_fake dataset
N_DATASETS = 100        # datasets per property (R14.1: 100 make_fake datasets)
# (signal, noise) regimes cycled across datasets so every generation mode
# is exercised: deterministic top-k (noise=0), pure noise (signal=0), mixed.
REGIMES = ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (1.0, 1.0))
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.20)   # run_backtest's default budgets
ALL_METHODS = ("worst_first", "oldest_first", "random_avg", "oracle", "model")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run_backtest_test_seeds(df_2018, df_2022, county_fee, **kwargs):
    """run_backtest with the random_avg leg on RANDOM_SEEDS_TEST (R14).

    Property tests use the 20 test seeds for random_avg, not the 200
    production seeds; run_backtest's tiebreaks parameter does not control
    the random_avg seed range. run_backtest looks up the module-level
    RANDOM_SEEDS when called, so swap in RANDOM_SEEDS_TEST for the call
    and restore the original afterwards no matter what.
    """
    original = backtest.RANDOM_SEEDS
    backtest.RANDOM_SEEDS = RANDOM_SEEDS_TEST
    try:
        return run_backtest(df_2018, df_2022, county_fee, **kwargs)
    finally:
        backtest.RANDOM_SEEDS = original


def _shared_pool(df_2018, df_2022, county_fee):
    """Rebuild the ONE shared candidate pool run_backtest builds (R11.2).

    Same deterministic pipeline (load_and_validate then
    apply_exclusion_and_label), so candidates/outcomes/pool_cost_info are
    identical to the ones used inside run_backtest.
    """
    a18, a22, index = load_and_validate(df_2018, df_2022)
    return apply_exclusion_and_label(a18, a22, index, county_fee)


def _model_scores(truth, outcomes):
    """Ceiling-style model scores: p_decline * log1p(adt) over the pool.

    Finite and non-negative (p_decline in (0, 1); pool adt is non-NaN per
    R2), satisfying pick's value-per-cost validation (R5.3).
    """
    return {
        sid: float(truth.loc[sid, "p_decline"] * np.log1p(o["adt"]))
        for sid, o in outcomes.items()
    }


# ---------------------------------------------------------------------------
# R14.1: Budget compliance
# ---------------------------------------------------------------------------

def test_budget_compliance():
    """Every selection respects its budget.

    **Validates: Requirements R14.1**

    Over 100 make_fake datasets, for ALL methods (worst_first,
    oldest_first, random_avg on the 20 test seeds, oracle, model) and ALL
    budgets, the reported cost_total of every non-aggregate row satisfies
    cost_total <= available * (1 + REL_TOL), with
    available = budget * total_cost_pool (R4.4, R4.5; REL_TOL = 1e-9, a
    relative tolerance with no absolute epsilon). Aggregate rows carry no
    cost (R8.3) and are skipped.
    """
    violations = []
    rows_checked = 0
    covered = set()
    expected_rows_per_dataset = len(BUDGETS) * (
        2 + len(RANDOM_SEEDS_TEST) + 2   # baselines + random_avg seeds + oracle + model
    )

    for i in range(N_DATASETS):
        signal, noise = REGIMES[i % len(REGIMES)]
        df_2018, df_2022, truth = make_fake(FAKE_N, seed=i, signal=signal,
                                            noise=noise)
        _, outcomes, pool_cost_info = _shared_pool(df_2018, df_2022,
                                                   COUNTY_FEE)
        total_cost_pool = pool_cost_info["total_cost_pool"]

        results = _run_backtest_test_seeds(
            df_2018, df_2022, COUNTY_FEE,
            model_scores=_model_scores(truth, outcomes),
        )

        actual = results[~results["is_aggregate"]]
        assert len(actual) == expected_rows_per_dataset, (
            f"dataset {i}: expected {expected_rows_per_dataset} "
            f"non-aggregate rows (all methods x budgets), got {len(actual)}"
        )

        for row in actual.itertuples():
            rows_checked += 1
            covered.add((row.method, row.budget))
            available = row.budget * total_cost_pool
            if not row.cost_total <= available * (1 + REL_TOL):
                violations.append(
                    f"dataset={i} method={row.method} budget={row.budget} "
                    f"seed={row.seed} cost_total={row.cost_total!r} "
                    f"available={available!r}"
                )

    assert covered == {(m, b) for m in ALL_METHODS for b in BUDGETS}, (
        f"missing (method, budget) coverage: "
        f"{sorted({(m, b) for m in ALL_METHODS for b in BUDGETS} - covered)}"
    )
    assert not violations, (
        f"{len(violations)} of {rows_checked} selections exceeded "
        f"available*(1+{REL_TOL}):\n" + "\n".join(violations[:10])
    )

# ---------------------------------------------------------------------------
# R14.2: No duplicate selections
# ---------------------------------------------------------------------------

def test_no_duplicates():
    """No selection contains the same bridge twice.

    **Validates: Requirements R14.2**

    Over 100 make_fake datasets, for ALL methods (worst_first,
    oldest_first, random_avg on the 20 test seeds, oracle, model) and ALL
    budgets, every selection satisfies
    len(selected) == len(set(selected)): no structure_id appears twice in
    any selection. run_backtest reports metrics rows only (R11), so the
    selections are reconstructed through the same deterministic pipeline
    run_backtest uses: the ONE shared pool (R11.2, _shared_pool) plus pick
    with the same score dicts (score_worst_first / score_oldest_first with
    tiebreak="adt", score_random per test seed, score_oracle and the model
    scores with rank_by_value_per_cost=True). Each reconstructed
    selection's unique count is cross-checked against the count_selected
    run_backtest reports for the same (method, budget [, seed]) row, tying
    the reconstruction to the harness's actual selections.
    """
    violations = []
    selections_checked = 0
    covered = set()
    expected_rows_per_dataset = len(BUDGETS) * (
        2 + len(RANDOM_SEEDS_TEST) + 2   # baselines + random_avg seeds + oracle + model
    )

    for i in range(N_DATASETS):
        signal, noise = REGIMES[i % len(REGIMES)]
        df_2018, df_2022, truth = make_fake(FAKE_N, seed=i, signal=signal,
                                            noise=noise)
        candidates, outcomes, pool_cost_info = _shared_pool(df_2018, df_2022,
                                                            COUNTY_FEE)
        model_scores = _model_scores(truth, outcomes)

        results = _run_backtest_test_seeds(
            df_2018, df_2022, COUNTY_FEE, model_scores=model_scores,
        )
        actual = results[~results["is_aggregate"]]
        assert len(actual) == expected_rows_per_dataset, (
            f"dataset {i}: expected {expected_rows_per_dataset} "
            f"non-aggregate rows (all methods x budgets), got {len(actual)}"
        )

        # The same budget-independent score dicts run_backtest builds on the
        # shared pool (R11.2); random_avg on the 20 test seeds.
        baseline_scores = {
            "worst_first": score_worst_first(candidates),
            "oldest_first": score_oldest_first(candidates),
        }
        oracle_scores = score_oracle(outcomes)
        random_scores = {
            s: score_random(candidates, s) for s in RANDOM_SEEDS_TEST
        }

        # Reconstruct every selection run_backtest makes, per (method,
        # budget [, seed]).
        selections = {}
        for budget in BUDGETS:
            for method, scores in baseline_scores.items():
                selections[(method, budget, None)] = pick(
                    scores, budget, candidates, pool_cost_info, COUNTY_FEE,
                    tiebreak="adt",
                )
            for s in RANDOM_SEEDS_TEST:
                selections[("random_avg", budget, s)] = pick(
                    random_scores[s], budget, candidates, pool_cost_info,
                    COUNTY_FEE,
                )
            selections[("oracle", budget, None)] = pick(
                oracle_scores, budget, candidates, pool_cost_info,
                COUNTY_FEE, rank_by_value_per_cost=True, tiebreak="adt",
            )
            selections[("model", budget, None)] = pick(
                model_scores, budget, candidates, pool_cost_info,
                COUNTY_FEE, rank_by_value_per_cost=True, tiebreak="adt",
            )

        # Cross-check: each run_backtest row's count_selected equals the
        # reconstructed selection's unique-id count for the same
        # (method, budget [, seed]).
        for row in actual.itertuples():
            seed = None if np.isnan(row.seed) else int(row.seed)
            selected = selections[(row.method, row.budget, seed)]
            assert row.count_selected == len(set(selected)), (
                f"dataset {i}: run_backtest row (method={row.method} "
                f"budget={row.budget} seed={seed}) reports "
                f"count_selected={row.count_selected} but the reconstructed "
                f"selection has {len(set(selected))} unique ids"
            )

        for (method, budget, seed), selected in selections.items():
            selections_checked += 1
            covered.add((method, budget))
            if len(selected) != len(set(selected)):
                seen_once, dups = set(), set()
                for sid in selected:
                    if sid in seen_once:
                        dups.add(sid)
                    seen_once.add(sid)
                violations.append(
                    f"dataset={i} method={method} budget={budget} "
                    f"seed={seed} len(selected)={len(selected)} "
                    f"unique={len(set(selected))} "
                    f"duplicated ids: {sorted(dups)[:5]}"
                )

    assert covered == {(m, b) for m in ALL_METHODS for b in BUDGETS}, (
        f"missing (method, budget) coverage: "
        f"{sorted({(m, b) for m in ALL_METHODS for b in BUDGETS} - covered)}"
    )
    assert not violations, (
        f"{len(violations)} of {selections_checked} selections contained a "
        "duplicated structure_id:\n" + "\n".join(violations[:10])
    )
