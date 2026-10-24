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

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from planner import backtest
from planner.backtest import (
    EXCLUSION_REASONS,
    FAKE_SEEDS,
    RANDOM_SEEDS_TEST,
    REL_TOL,
    apply_exclusion_and_label,
    is_safety_drop,
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
RATING_VALUES = tuple(range(10)) + (float("nan"),)  # 0-9 plus NaN (R14.3)
N_TRIPLE_PAIRS = 20_000    # random (2018, 2022) triple pairs in R14.3(c)
TRIPLE_PAIR_SEED = 0       # fixed rng seed for the R14.3(c) draws
TRIPLE_PAIR_NAN_P = 0.15   # per-cell NaN probability, matches make_fake (R13.8)
ORACLE_SLACK = 0.01        # R14.4 slack on the MEANS over seeds (per budget)
REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_PATHS = (REPO_ROOT / "data" / "MA18.txt", REPO_ROOT / "data" / "MA22.txt")


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

# ---------------------------------------------------------------------------
# R14.3: Deterioration monotonicity
# ---------------------------------------------------------------------------

# Independent reference implementations (R14.3): written directly from the
# R3 formulas, they must NOT call the functions under test
# (apply_exclusion_and_label / is_safety_drop).

def _ref_min_ignoring_nan(deck, super_, sub):
    """Reference min step: explicit Python loop, min ignoring NaN (R3.1).

    Returns NaN when all three components are NaN (the pipeline must then
    exclude the bridge, R2.1 / R3.3).
    """
    best = None
    for v in (deck, super_, sub):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if best is None or v < best:
            best = v
    return float("nan") if best is None else float(best)


def _ref_is_safety_drop(start, end):
    """Reference label: the literal formula (end < start) and (end <= 4) (R3.2).

    Python semantics make every NaN comparison False, so NaN on either side
    yields False — such bridges are excluded from the pool, never labeled
    (R3.3).
    """
    return bool((end < start) and (end <= 4))


def _frames_from_triples(ids, triples18, triples22):
    """Build a minimal valid (df_2018, df_2022) pair from rating triples.

    All non-rating fields are valid and constant (one county, year_built,
    adt, inspect_cost; rebuilt=0; every id unique and present in both
    years), so the ONLY exclusions the pipeline can make are NaN-min ones —
    exactly what R14.3 checks.
    """
    df_2018 = pd.DataFrame({
        "structure_id": list(ids),
        "county": "C001",
        "year_built": 1950,
        "adt": 100.0,
        "deck": [t[0] for t in triples18],
        "super": [t[1] for t in triples18],
        "sub": [t[2] for t in triples18],
        "inspect_cost": 1.0,
    })
    df_2022 = pd.DataFrame({
        "structure_id": list(ids),
        "deck": [t[0] for t in triples22],
        "super": [t[1] for t in triples22],
        "sub": [t[2] for t in triples22],
        "rebuilt": 0,
    })
    return df_2018, df_2022


def test_deterioration_monotonicity():
    """worse iff (min_2022 < min_2018) and (min_2022 <= 4); NaN min -> excluded.

    **Validates: Requirements R14.3**

    Three legs, each checked against the independent references above
    (written from the R3 formulas, never calling the functions under test):

      (a) Exhaustive min step: all 11^3 = 1331 (deck, super, sub) triples
          over 0-9 plus NaN are run through the pipeline as BOTH the 2018
          and 2022 ratings. Every pool bridge's min_2018/min_2022 must
          equal the reference min-ignoring-NaN loop, and the single
          all-NaN triple must be excluded (nan_min_2018) — an all-NaN
          triple yields a NaN min, and a NaN min means absent from pool.

      (b) Safety-drop / exclusion: all 11x11 = 121 (min_2018, min_2022)
          combos including NaN. The pipeline's is_safety_drop must equal
          the literal formula (end < start) and (end <= 4) on every combo;
          through the pipeline (mins pinned by all-equal triples), any NaN
          min on either side means the bridge is absent from the pool,
          while every kept bridge's worse label equals the formula.

      (c) 20,000 random (2018 triple, 2022 triple) pairs (integer 0-9
          cells, 15% NaN per cell) run vectorized through
          apply_exclusion_and_label; per id, the exclusion decision, pool
          mins, and worse label must match the reference, and the
          aggregate exclusion counts and pool size must match the
          reference tallies.
    """
    violations = []

    # ---------------- Leg (a): exhaustive 11^3 min step ----------------
    triples = list(product(RATING_VALUES, repeat=3))
    ids_a = [f"A{i:04d}" for i in range(len(triples))]
    df_2018, df_2022 = _frames_from_triples(ids_a, triples, triples)
    _, outcomes_a, pool_a = apply_exclusion_and_label(
        *load_and_validate(df_2018, df_2022), COUNTY_FEE
    )
    n_all_nan_a = 0
    for sid, t in zip(ids_a, triples):
        ref = _ref_min_ignoring_nan(*t)
        if np.isnan(ref):
            n_all_nan_a += 1
            if sid in outcomes_a:
                violations.append(
                    f"(a) all-NaN triple {t}: {sid} present in pool"
                )
            continue
        o = outcomes_a.get(sid)
        if o is None:
            violations.append(
                f"(a) triple {t}: {sid} absent from pool (ref min {ref})"
            )
        elif o["min_2018"] != ref or o["min_2022"] != ref or o["worse"]:
            violations.append(
                f"(a) triple {t}: outcomes={o}; ref min {ref} for both "
                "years (identical triples can never be worse)"
            )
    assert n_all_nan_a == 1, (
        f"(a) expected exactly 1 all-NaN triple among the 11^3, got {n_all_nan_a}"
    )
    if pool_a["exclusion_counts"]["nan_min_2018"] != 1:
        violations.append(
            f"(a) nan_min_2018 exclusions "
            f"= {pool_a['exclusion_counts']['nan_min_2018']}; expected 1 "
            "(the all-NaN triple)"
        )
    other_exclusions_a = sum(
        v for k, v in pool_a["exclusion_counts"].items() if k != "nan_min_2018"
    )
    if other_exclusions_a:
        violations.append(
            f"(a) unexpected exclusions beyond the all-NaN triple: "
            f"{pool_a['exclusion_counts']}"
        )
    if len(outcomes_a) != len(triples) - 1:
        violations.append(
            f"(a) pool size {len(outcomes_a)}; expected {len(triples) - 1} "
            "(every triple except the all-NaN one)"
        )

    # ---------- Leg (b): 11x11 safety-drop / exclusion combos ----------
    # Direct comparison of the pipeline's is_safety_drop against the
    # literal formula on every (start, end) combo.
    for start in RATING_VALUES:
        for end in RATING_VALUES:
            got = bool(is_safety_drop(start, end))
            want = _ref_is_safety_drop(start, end)
            if got != want:
                violations.append(
                    f"(b) is_safety_drop(start={start}, end={end}) = {got}; "
                    f"literal formula = {want}"
                )

    # Through the pipeline: min_2018 pinned by an all-equal 2018 triple
    # (all-NaN for a NaN min), same for min_2022.
    combos = [(s, e) for s in RATING_VALUES for e in RATING_VALUES]
    ids_b = [f"B{i:03d}" for i in range(len(combos))]
    t18_b = [(np.nan,) * 3 if np.isnan(s) else (s, s, s) for s, _ in combos]
    t22_b = [(np.nan,) * 3 if np.isnan(e) else (e, e, e) for _, e in combos]
    df_2018, df_2022 = _frames_from_triples(ids_b, t18_b, t22_b)
    _, outcomes_b, pool_b = apply_exclusion_and_label(
        *load_and_validate(df_2018, df_2022), COUNTY_FEE
    )
    for sid, (start, end) in zip(ids_b, combos):
        if np.isnan(start) or np.isnan(end):
            if sid in outcomes_b:
                violations.append(
                    f"(b) (start={start}, end={end}): NaN min but {sid} "
                    "present in pool"
                )
            continue
        o = outcomes_b.get(sid)
        want = _ref_is_safety_drop(start, end)
        if o is None:
            violations.append(
                f"(b) (start={start}, end={end}): {sid} absent from pool"
            )
        elif (o["min_2018"] != start or o["min_2022"] != end
                or o["worse"] != want):
            violations.append(
                f"(b) (start={start}, end={end}): outcomes={o}; expected "
                f"mins ({start}, {end}) and worse={want}"
            )
    # NaN min_2018 is checked before min_2022 is even looked up, so the 11
    # NaN-start combos count under nan_min_2018 and the 10 valid-start /
    # NaN-end combos under nan_min_2022; the other 100 combos stay.
    counts_b = pool_b["exclusion_counts"]
    if counts_b["nan_min_2018"] != 11 or counts_b["nan_min_2022"] != 10:
        violations.append(
            f"(b) exclusion counts {counts_b}; expected nan_min_2018=11, "
            "nan_min_2022=10"
        )
    if len(outcomes_b) != 100:
        violations.append(
            f"(b) pool size {len(outcomes_b)}; expected 100 kept combos"
        )

    # ------ Leg (c): 20,000 random triple pairs through the pipeline ------
    rng = np.random.default_rng(TRIPLE_PAIR_SEED)
    cells = rng.integers(0, 10, size=(N_TRIPLE_PAIRS, 6)).astype(float)
    cells[rng.random((N_TRIPLE_PAIRS, 6)) < TRIPLE_PAIR_NAN_P] = np.nan
    triples18, triples22 = cells[:, :3], cells[:, 3:]
    ids_c = [f"C{i:05d}" for i in range(N_TRIPLE_PAIRS)]
    df_2018, df_2022 = _frames_from_triples(ids_c, triples18, triples22)
    _, outcomes_c, pool_c = apply_exclusion_and_label(
        *load_and_validate(df_2018, df_2022), COUNTY_FEE
    )

    ref_m18 = np.array([_ref_min_ignoring_nan(*t) for t in triples18])
    ref_m22 = np.array([_ref_min_ignoring_nan(*t) for t in triples22])
    excl18 = np.isnan(ref_m18)                    # NaN min 2018: checked first
    excl22 = ~excl18 & np.isnan(ref_m22)          # NaN min 2022, valid 2018
    kept = ~(excl18 | excl22)
    ref_worse_kept = np.array([
        _ref_is_safety_drop(a, b)
        for a, b in zip(ref_m18[kept], ref_m22[kept])
    ])
    # Coverage guards: the fixed-seed draws must actually exercise every
    # case this leg claims to check.
    assert excl18.any() and excl22.any(), (
        f"(c) random draws produced too few NaN-min exclusions "
        f"(nan_min_2018={int(excl18.sum())}, nan_min_2022={int(excl22.sum())})"
    )
    assert ref_worse_kept.any() and (~ref_worse_kept).any(), (
        "(c) random draws failed to produce both worse and not-worse kept "
        "bridges"
    )

    for sid, m18, m22 in zip(ids_c, ref_m18, ref_m22):
        if np.isnan(m18) or np.isnan(m22):
            if sid in outcomes_c:
                violations.append(
                    f"(c) {sid}: ref mins ({m18}, {m22}) but present in pool"
                )
            continue
        o = outcomes_c.get(sid)
        want = _ref_is_safety_drop(m18, m22)
        if o is None:
            violations.append(
                f"(c) {sid}: ref mins ({m18}, {m22}) but absent from pool"
            )
        elif (o["min_2018"] != m18 or o["min_2022"] != m22
                or o["worse"] != want):
            violations.append(
                f"(c) {sid}: outcomes={o}; ref mins ({m18}, {m22}), "
                f"worse={want}"
            )
    counts_c = pool_c["exclusion_counts"]
    if (counts_c["nan_min_2018"] != int(excl18.sum())
            or counts_c["nan_min_2022"] != int(excl22.sum())):
        violations.append(
            f"(c) exclusion counts {counts_c}; expected nan_min_2018="
            f"{int(excl18.sum())}, nan_min_2022={int(excl22.sum())}"
        )
    if len(outcomes_c) != int(kept.sum()):
        violations.append(
            f"(c) pool size {len(outcomes_c)}; expected {int(kept.sum())}"
        )

    assert not violations, (
        f"{len(violations)} deterioration-monotonicity violations "
        f"(R14.3):\n" + "\n".join(violations[:10])
    )

# ---------------------------------------------------------------------------
# R14.4: Oracle upper bound (soft aggregate, weighted only)
# ---------------------------------------------------------------------------

def _weighted_recall_by_method(results):
    """Per (method, budget) weighted recall from ONE run_backtest result.

    random_avg is represented by its stat="mean" aggregate row over the
    test seeds (R8.3); every other method by its single non-aggregate row
    per budget. Returns {method: {budget: weighted_recall}}.
    """
    out = {}
    for method in ALL_METHODS:
        sub = results[results["method"] == method]
        if sub.empty:
            continue
        if method == "random_avg":
            sub = sub[sub["is_aggregate"] & (sub["stat"] == "mean")]
        else:
            sub = sub[~sub["is_aggregate"]]
        out[method] = dict(zip(sub["budget"], sub["weighted_recall"]))
    return out


def _oracle_bound_check(dataset_recs, label, per_seed_rows):
    """Aggregate R14.4 check over one leg's datasets.

    dataset_recs: list of (dataset_label, {method: {budget:
    weighted_recall}}). Every (dataset, budget, method) where some other
    method's weighted recall beat the oracle's is appended to
    per_seed_rows — reported as a table, never a failure by itself
    (R14.4). Returns aggregate-violation strings: at every budget the MEAN
    over datasets of oracle weighted recall must be >= each other method's
    mean - ORACLE_SLACK.
    """
    violations = []
    others = [m for m in ALL_METHODS if m != "oracle"]
    oracle_series = {b: [] for b in BUDGETS}
    method_series = {}                       # (method, budget) -> list
    for ds_label, rec in dataset_recs:
        for budget in BUDGETS:
            o_val = rec["oracle"][budget]
            oracle_series[budget].append(o_val)
            for method in others:
                if method not in rec:
                    continue
                m_val = rec[method][budget]
                method_series.setdefault((method, budget), []).append(m_val)
                if m_val > o_val:
                    per_seed_rows.append({
                        "leg": label, "dataset": ds_label,
                        "budget": budget, "method": method,
                        "oracle": round(o_val, 4),
                        "method_wrec": round(m_val, 4),
                        "deficit": round(m_val - o_val, 4),
                    })
    for budget in BUDGETS:
        o_mean = float(np.mean(oracle_series[budget]))
        for method in others:
            key = (method, budget)
            if key not in method_series:
                continue
            m_mean = float(np.mean(method_series[key]))
            if not o_mean >= m_mean - ORACLE_SLACK:
                violations.append(
                    f"[{label}] budget={budget} method={method}: "
                    f"mean oracle weighted recall {o_mean:.4f} < "
                    f"mean {method} {m_mean:.4f} - {ORACLE_SLACK}"
                )
    return violations


def test_oracle_upper_bound():
    """Mean oracle weighted recall upper-bounds every method (soft aggregate).

    **Validates: Requirements R14.4**

    Over FAKE_SEEDS (50) make_fake datasets (regimes cycled so every
    generation mode is exercised), at EACH budget:

        mean(oracle weighted recall) >= mean(method weighted recall) - 0.01

    for every other method (worst_first, oldest_first, random_avg, model;
    random_avg represented per dataset by its mean row over the 20 test
    seeds, R8.3). This is a SOFT aggregate property: the assertion is on
    the MEANS over the 50 seeds. Per-seed deficits (a dataset/budget where
    some method's weighted recall beat the oracle's) are collected and
    REPORTED as a table; a single-seed violation never fails the test
    (R14.4). Weighted recall only — unweighted dominance is not claimed or
    tested (R14.4).

    Real-data leg: the same check runs on adapt_nbi output
    (data/MA18.txt, data/MA22.txt; flat inspect_cost=1, COUNTY_FEE=5
    placeholder units) when the adapter and both raw files are available;
    until then the leg is skipped and the fake leg alone gates.
    """
    violations = []
    per_seed_rows = []

    # ---------------- Fake leg: FAKE_SEEDS make_fake datasets ----------------
    fake_recs = []
    for i in FAKE_SEEDS:
        signal, noise = REGIMES[i % len(REGIMES)]
        df_2018, df_2022, truth = make_fake(FAKE_N, seed=i, signal=signal,
                                            noise=noise)
        _, outcomes, _ = _shared_pool(df_2018, df_2022, COUNTY_FEE)
        results = _run_backtest_test_seeds(
            df_2018, df_2022, COUNTY_FEE,
            model_scores=_model_scores(truth, outcomes),
        )
        rec = _weighted_recall_by_method(results)
        assert set(rec) == set(ALL_METHODS), (
            f"dataset {i}: expected all methods {ALL_METHODS}, "
            f"got {sorted(rec)}"
        )
        for method, per_budget in rec.items():
            assert set(per_budget) == set(BUDGETS), (
                f"dataset {i} method {method}: expected budgets "
                f"{sorted(BUDGETS)}, got {sorted(per_budget)}"
            )
        fake_recs.append((f"fake-{i}", rec))
    violations += _oracle_bound_check(fake_recs, "fake", per_seed_rows)

    # ---------------- Real-data leg: only when available ----------------
    adapter = getattr(backtest, "adapt_nbi", None)
    if adapter is not None and all(p.exists() for p in REAL_PATHS):
        d18, d22 = adapter(str(REAL_PATHS[0]), str(REAL_PATHS[1]),
                           1.0, COUNTY_FEE)
        real_results = _run_backtest_test_seeds(d18, d22, COUNTY_FEE)
        real_rec = _weighted_recall_by_method(real_results)
        violations += _oracle_bound_check(
            [("real-data", real_rec)], "real", per_seed_rows,
        )
    else:
        print("R14.4 real-data leg SKIPPED: adapt_nbi and/or "
              "data/MA18.txt, data/MA22.txt not available yet")

    table = (
        pd.DataFrame(per_seed_rows).to_string(index=False)
        if per_seed_rows else "(none)"
    )
    print(f"R14.4 per-seed oracle deficits over {len(fake_recs)} fake "
          f"datasets (reported, NOT individually failing; the gate is the "
          f"mean over seeds):\n{table}")
    assert not violations, (
        f"{len(violations)} aggregate oracle-upper-bound violations (R14.4: "
        f"mean oracle weighted recall must be >= each method's mean - "
        f"{ORACLE_SLACK} per budget):\n" + "\n".join(violations)
        + f"\n\nPer-seed violations (reported, not individually failing):"
        f"\n{table}"
    )

# ---------------------------------------------------------------------------
# R14.5: Exclusion consistency
# ---------------------------------------------------------------------------

def _inject_exclusion_cases(df_2018, df_2022):
    """Append rows exercising the R2 reasons make_fake never produces.

    make_fake always emits unique structure_ids, at least one non-NaN 2018
    rating per bridge (so min_2018 is never NaN), an anchored non-NaN 2022
    rating (so min_2022 is never NaN), valid year_built, and positive adt
    (R13.2, R13.6, R13.8), so the duplicate_id, nan_min_2018,
    nan_min_2022, and missing_yearbuilt_adt reasons can never fire on
    generated data alone. Every injected row is excluded by construction —
    none ever enters the pool, so truth needs no entries for the new ids:

      first 2018 row copied again -> duplicate_id (BOTH rows of the id)
      first 2022 row copied again -> duplicate_id (id duplicated in 2022)
      INJ_MISSING: 2018 only                    -> missing_2022
      INJ_REBUILT: rebuilt=True in 2022         -> rebuilt
      INJ_NAN18:   all-NaN 2018 ratings         -> nan_min_2018
      INJ_NAN22:   all-NaN 2022 ratings         -> nan_min_2022
      INJ_YBNAN:   year_built NaN               -> missing_yearbuilt_adt
      INJ_YBZERO:  year_built 0 (<= 0)          -> missing_yearbuilt_adt
      INJ_ADTNAN:  adt NaN                      -> missing_yearbuilt_adt

    Injected rows sit in county "C999" (no make_fake county, R13.8 uses
    C000-C013), so a wrongly kept injected row also shifts the pool's
    county set and the R4.4 denominator.
    """
    valid18 = {"county": "C999", "year_built": 1950, "adt": 100.0,
               "deck": 5.0, "super": 5.0, "sub": 5.0, "inspect_cost": 1.0}
    inj18 = pd.DataFrame([
        {"structure_id": "INJ_MISSING", **valid18},
        {"structure_id": "INJ_REBUILT", **valid18},
        {"structure_id": "INJ_NAN18", **valid18,
         "deck": None, "super": None, "sub": None},
        {"structure_id": "INJ_NAN22", **valid18},
        {"structure_id": "INJ_YBNAN", **valid18, "year_built": None},
        {"structure_id": "INJ_YBZERO", **valid18, "year_built": 0},
        {"structure_id": "INJ_ADTNAN", **valid18, "adt": None},
    ])
    ok22 = {"deck": 5.0, "super": 5.0, "sub": 5.0, "rebuilt": False}
    inj22 = pd.DataFrame([
        {"structure_id": "INJ_REBUILT", **ok22, "rebuilt": True},
        {"structure_id": "INJ_NAN18", **ok22},
        {"structure_id": "INJ_NAN22",
         "deck": None, "super": None, "sub": None, "rebuilt": False},
        {"structure_id": "INJ_YBNAN", **ok22},
        {"structure_id": "INJ_YBZERO", **ok22},
        {"structure_id": "INJ_ADTNAN", **ok22},
    ])
    df_2018 = pd.concat([df_2018, df_2018.iloc[[0]], inj18],
                        ignore_index=True)
    df_2022 = pd.concat([df_2022, df_2022.iloc[[0]], inj22],
                        ignore_index=True)
    return df_2018, df_2022


def _ref_exclusion_and_pool(df_2018, df_2022):
    """Independent reference for R2 exclusion, written inside the test file.

    Computed directly from the R2 rules over the raw input frames, using
    the R14.3 references (_ref_min_ignoring_nan, _ref_is_safety_drop) for
    the R3 min/label steps; it never calls apply_exclusion_and_label and
    never reads its exclusion_counts, so the R14.5 pool-equality and
    budget-denominator checks do not trust the implementation's own
    bookkeeping. Per-reason tallies follow Algorithm 2's documented check
    order (duplicate_id, missing_2022, rebuilt, nan_min_2018,
    nan_min_2022, missing_yearbuilt_adt): a row hitting several reasons
    counts under the FIRST one, so the tallies are comparable to
    exclusion_counts.

    Returns (pool, excluded, counts): pool maps each kept structure_id to
    its reference fields (county, inspect_cost, adt, year_built,
    min_2018, min_2022, worse); excluded is the set of excluded
    structure_ids (ALL rows of a duplicated id land here, R2.1); counts
    maps each R2 reason to the number of 2018 ROWS excluded for it.
    """
    ids18 = df_2018["structure_id"].astype(str)
    ids22 = df_2022["structure_id"].astype(str)
    dup_ids = (
        set(ids18[ids18.duplicated(keep=False)])
        | set(ids22[ids22.duplicated(keep=False)])
    )
    present22 = set(ids22)
    # 2022 lookups keyed by id; values for duplicated ids are never used
    # (those rows are excluded under duplicate_id before any lookup).
    triple22 = {
        sid: (d, s, b)
        for sid, d, s, b in zip(ids22, df_2022["deck"], df_2022["super"],
                                df_2022["sub"])
    }
    rebuilt22 = {
        sid: (False if pd.isna(v) else bool(float(v)))
        for sid, v in zip(ids22, df_2022["rebuilt"])
    }

    year_built = pd.to_numeric(df_2018["year_built"], errors="coerce")
    adt = pd.to_numeric(df_2018["adt"], errors="coerce")
    inspect_cost = pd.to_numeric(df_2018["inspect_cost"], errors="coerce")

    counts = {reason: 0 for reason in EXCLUSION_REASONS}
    pool = {}
    excluded = set()
    rows = zip(ids18, df_2018["county"].astype(str), year_built, adt,
               inspect_cost, df_2018["deck"], df_2018["super"],
               df_2018["sub"])
    for sid, county, yb, a, cost, d, s, b in rows:
        if sid in dup_ids:
            reason = "duplicate_id"
        elif sid not in present22:
            reason = "missing_2022"
        elif rebuilt22[sid]:
            reason = "rebuilt"
        else:
            m18 = _ref_min_ignoring_nan(d, s, b)
            m22 = _ref_min_ignoring_nan(*triple22[sid])
            if np.isnan(m18):
                reason = "nan_min_2018"
            elif np.isnan(m22):
                reason = "nan_min_2022"
            elif pd.isna(yb) or yb <= 0 or pd.isna(a):
                reason = "missing_yearbuilt_adt"
            else:
                pool[sid] = {
                    "county": county,
                    "inspect_cost": float(cost),
                    "adt": float(a),
                    "year_built": int(yb),
                    "min_2018": m18,
                    "min_2022": m22,
                    "worse": _ref_is_safety_drop(m18, m22),
                }
                continue
        counts[reason] += 1
        excluded.add(sid)
    return pool, excluded, counts


def test_exclusion_consistency():
    """No excluded id selected; pool = input - R2 reasons; R4.4 denominator.

    **Validates: Requirements R14.5**

    Over N_DATASETS make_fake datasets (regimes cycled so every generation
    mode is exercised), each augmented by _inject_exclusion_cases so EVERY
    R2 reason fires on every dataset (make_fake alone can never produce
    duplicate ids, NaN min_2018, NaN min_2022, or invalid year_built/adt,
    R13.2/R13.6):

      (1) No excluded id in any selection: every selection of every
          method (worst_first / oldest_first with tiebreak="adt",
          random_avg on the 20 RANDOM_SEEDS_TEST seeds, oracle, model) at
          every budget is reconstructed the same way test_no_duplicates
          does — the ONE shared pool from _shared_pool plus pick with the
          same score dicts run_backtest builds (R11.2) — and no selected
          id may intersect the reference-excluded set or fall outside the
          2018 input ids.
      (2) Pool equality: the candidate pool (its id set AND every
          candidates/outcomes field per id) equals the 2018 input minus
          EVERY R2 reason — duplicates (all rows of a duplicated id),
          missing from 2022, rebuilt, NaN min_2018, NaN min_2022,
          year_built NaN/<=0 or adt NaN — as computed by the independent
          reference _ref_exclusion_and_pool, which is written in this
          file from the R2 rules and never reads
          apply_exclusion_and_label's own counts. The per-reason
          exclusion_counts must equal the reference tallies too.
      (3) Budget denominator: total_cost_pool equals sum(inspect_cost)
          over the REFERENCE pool plus n_unique_reference_pool_counties *
          county_fee (R4.4) — excluded ids contribute nothing. Every
          inspect_cost is strictly positive (R1.4) and injected rows sit
          in a county no pool row uses, so any leakage from an excluded
          id shifts the total or the county set; unique_counties must
          equal the reference pool's county set.

    A coverage guard on the reference tallies asserts every R2 reason
    occurred at least once in every dataset.
    """
    violations = []
    covered = set()
    rows_checked = 0
    selections_checked = 0

    for i in range(N_DATASETS):
        signal, noise = REGIMES[i % len(REGIMES)]
        df_2018, df_2022, truth = make_fake(FAKE_N, seed=i, signal=signal,
                                            noise=noise)
        df_2018, df_2022 = _inject_exclusion_cases(df_2018, df_2022)
        rows_checked += len(df_2018)

        ref_pool, ref_excluded, ref_counts = _ref_exclusion_and_pool(
            df_2018, df_2022
        )
        # Coverage guard: every R2 reason actually fired (reference
        # tallies, not the implementation's counts).
        for reason in EXCLUSION_REASONS:
            assert ref_counts[reason] >= 1, (
                f"dataset {i}: R2 reason {reason!r} never exercised "
                f"(reference tallies {ref_counts})"
            )

        candidates, outcomes, pool_cost_info = _shared_pool(
            df_2018, df_2022, COUNTY_FEE
        )

        # ---------------- (2) pool equality vs the reference ----------------
        missing_from_pool = sorted(set(ref_pool) - set(candidates.index))
        extra_in_pool = sorted(set(candidates.index) - set(ref_pool))
        if missing_from_pool or extra_in_pool:
            violations.append(
                f"dataset={i}: pool id set mismatch — excluded by "
                f"reference but in candidates {extra_in_pool[:5]}, kept "
                f"by reference but absent {missing_from_pool[:5]}"
            )
        for sid, ref in ref_pool.items():
            if sid not in candidates.index:
                continue  # already reported in the set mismatch above
            c = candidates.loc[sid]
            o = outcomes.get(sid)
            mismatches = []
            if c["county"] != ref["county"]:
                mismatches.append(
                    f"county {c['county']!r} != {ref['county']!r}"
                )
            if c["inspect_cost"] != ref["inspect_cost"]:
                mismatches.append(
                    f"inspect_cost {c['inspect_cost']!r} != "
                    f"{ref['inspect_cost']!r}"
                )
            if c["adt"] != ref["adt"]:
                mismatches.append(f"adt {c['adt']!r} != {ref['adt']!r}")
            if c["year_built"] != ref["year_built"]:
                mismatches.append(
                    f"year_built {c['year_built']!r} != "
                    f"{ref['year_built']!r}"
                )
            if c["min_2018"] != ref["min_2018"]:
                mismatches.append(
                    f"min_2018 {c['min_2018']!r} != {ref['min_2018']!r}"
                )
            if o is None:
                mismatches.append("absent from outcomes")
            else:
                if o["min_2018"] != ref["min_2018"]:
                    mismatches.append(
                        f"outcomes min_2018 {o['min_2018']!r} != "
                        f"{ref['min_2018']!r}"
                    )
                if o["min_2022"] != ref["min_2022"]:
                    mismatches.append(
                        f"min_2022 {o['min_2022']!r} != {ref['min_2022']!r}"
                    )
                if o["worse"] != ref["worse"]:
                    mismatches.append(
                        f"worse {o['worse']!r} != {ref['worse']!r}"
                    )
                if o["adt"] != ref["adt"]:
                    mismatches.append(
                        f"outcomes adt {o['adt']!r} != {ref['adt']!r}"
                    )
            if mismatches:
                violations.append(
                    f"dataset={i} id={sid}: " + "; ".join(mismatches)
                )
        if pool_cost_info["exclusion_counts"] != ref_counts:
            violations.append(
                f"dataset={i}: exclusion_counts "
                f"{pool_cost_info['exclusion_counts']} != reference "
                f"tallies {ref_counts}"
            )

        # ----- (3) excluded ids absent from the budget denominator -----
        ref_counties = {r["county"] for r in ref_pool.values()}
        want_total = (
            sum(r["inspect_cost"] for r in ref_pool.values())
            + len(ref_counties) * COUNTY_FEE
        )
        got_total = pool_cost_info["total_cost_pool"]
        if not np.isclose(got_total, want_total, rtol=REL_TOL, atol=0.0):
            violations.append(
                f"dataset={i}: total_cost_pool {got_total!r} != reference "
                f"{want_total!r} (sum inspect_cost over the reference "
                f"pool + {len(ref_counties)} counties * {COUNTY_FEE}, R4.4)"
            )
        if pool_cost_info["unique_counties"] != ref_counties:
            violations.append(
                f"dataset={i}: unique_counties != reference pool counties "
                f"(extra "
                f"{sorted(pool_cost_info['unique_counties'] - ref_counties)[:5]}"
                f", missing "
                f"{sorted(ref_counties - pool_cost_info['unique_counties'])[:5]}"
                f")"
            )

        # ------------- (1) no excluded id in any selection -------------
        # The same budget-independent score dicts run_backtest builds on
        # the shared pool (R11.2); random_avg on the 20 test seeds.
        baseline_scores = {
            "worst_first": score_worst_first(candidates),
            "oldest_first": score_oldest_first(candidates),
        }
        oracle_scores = score_oracle(outcomes)
        model_scores = _model_scores(truth, outcomes)
        random_scores = {
            s: score_random(candidates, s) for s in RANDOM_SEEDS_TEST
        }

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

        for (method, budget, seed), selected in selections.items():
            selections_checked += 1
            covered.add((method, budget))
            chosen = set(selected)
            leaked = sorted(chosen & ref_excluded)
            outside = sorted(chosen - set(ref_pool) - ref_excluded)
            if leaked or outside:
                violations.append(
                    f"dataset={i} method={method} budget={budget} "
                    f"seed={seed}: excluded ids selected {leaked[:5]}, "
                    f"ids outside the 2018 input {outside[:5]}"
                )

    assert covered == {(m, b) for m in ALL_METHODS for b in BUDGETS}, (
        f"missing (method, budget) coverage: "
        f"{sorted({(m, b) for m in ALL_METHODS for b in BUDGETS} - covered)}"
    )
    assert not violations, (
        f"{len(violations)} exclusion-consistency violations (R14.5) over "
        f"{N_DATASETS} datasets ({rows_checked} input rows, "
        f"{selections_checked} selections):\n" + "\n".join(violations[:10])
    )
