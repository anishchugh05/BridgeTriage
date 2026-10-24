"""Validation harness for the decline model (Part B).

Splits bridges 80/20 (fixed seed), fits on the training bridges over the
2012 -> 2016 interval, and evaluates on the held-out 20% over that same 4-year
interval. The prediction target is the shared safety drop (see
:mod:`model.target`). Reports positives, log loss, and AUC for three models:

    - grouped   : fit_decline (material x age band x condition, shrunk)
    - logistic  : fit_logistic (condition/material one-hot + age, log ADT, ...)
    - baseline  : per-condition drop rate only

and prints the new drop rate by starting condition.

``select_model`` returns whichever of grouped/logistic has the better test AUC,
which is what score() should use.

Run:
    .venv/bin/python -m model.validate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_panel
from .score import _build_pairs, fit_decline, fit_logistic

FIT_START, FIT_END = 2012, 2016
TEST_SEED = 1234
TEST_FRACTION = 0.20


def _log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney). Returns 0.5 if a class is absent."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _split_bridges(panel: pd.DataFrame, seed: int, test_fraction: float):
    ids = np.sort(panel["id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    n_test = int(round(len(shuffled) * test_fraction))
    return set(shuffled[n_test:]), set(shuffled[:n_test])


def _grouped_probs(rates, eval_df: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            rates.prob_for(r.material, r.age_band, r.cond)
            for r in eval_df.itertuples()
        ]
    )


def _baseline(train_eval: pd.DataFrame) -> dict:
    per = train_eval.groupby("cond")["dropped"].mean().to_dict()
    return {"per_cond": per, "overall": float(train_eval["dropped"].mean())}


def run(panel: pd.DataFrame, seed: int = TEST_SEED,
        test_fraction: float = TEST_FRACTION) -> dict:
    train_ids, test_ids = _split_bridges(panel, seed, test_fraction)
    train_panel = panel[panel["id"].isin(train_ids)]
    test_panel = panel[panel["id"].isin(test_ids)]

    train_eval = _build_pairs(train_panel, FIT_START, FIT_END)
    test_eval = _build_pairs(test_panel, FIT_START, FIT_END)
    if test_eval.empty:
        raise ValueError("empty test evaluation set")

    y = test_eval["dropped"].to_numpy()

    # --- fit the two models on the training bridges ------------------------
    grouped = fit_decline(train_panel, FIT_START, FIT_END)
    logistic = fit_logistic(train_panel, FIT_START, FIT_END)

    # --- predictions on the held-out test pairs ----------------------------
    p_grouped = _grouped_probs(grouped, test_eval)
    p_logistic = logistic.interval_prob_frame(test_eval)
    base = _baseline(train_eval)
    p_base = np.array(
        [base["per_cond"].get(int(c), base["overall"]) for c in test_eval["cond"]]
    )

    results = {
        "n_train_bridges": len(train_ids),
        "n_test_bridges": len(test_ids),
        "n_test_pairs": int(len(test_eval)),
        "test_positives": int(y.sum()),
        "models": {
            "grouped": {
                "positives": int(y.sum()),
                "log_loss": _log_loss(y, p_grouped),
                "auc": _auc(y, p_grouped),
            },
            "logistic": {
                "positives": int(y.sum()),
                "log_loss": _log_loss(y, p_logistic),
                "auc": _auc(y, p_logistic),
            },
            "baseline": {
                "positives": int(y.sum()),
                "log_loss": _log_loss(y, p_base),
                "auc": _auc(y, p_base),
            },
        },
        "drop_rate_by_cond": test_eval.groupby("cond")["dropped"]
        .agg(["mean", "count"]).to_dict("index"),
    }
    best = "logistic" if (
        results["models"]["logistic"]["auc"] >= results["models"]["grouped"]["auc"]
    ) else "grouped"
    results["best_model"] = best
    return results


def select_model(panel: pd.DataFrame, start: int, end: int,
                 seed: int = TEST_SEED, test_fraction: float = TEST_FRACTION):
    """Fit both models and return whichever has the better held-out AUC.

    Used by callers (and score) to pick the model to score with. Returns the
    fitted model refit on ALL bridges in the interval once the better family is
    chosen on the validation split.
    """
    train_ids, test_ids = _split_bridges(panel, seed, test_fraction)
    train_panel = panel[panel["id"].isin(train_ids)]
    test_eval = _build_pairs(panel[panel["id"].isin(test_ids)], start, end)

    grouped = fit_decline(train_panel, start, end)
    logistic = fit_logistic(train_panel, start, end)
    y = test_eval["dropped"].to_numpy()
    auc_g = _auc(y, _grouped_probs(grouped, test_eval))
    auc_l = _auc(y, logistic.interval_prob_frame(test_eval))

    if auc_l >= auc_g:
        return fit_logistic(panel, start, end)
    return fit_decline(panel, start, end)


def main() -> None:
    try:
        panel = load_panel()
        source = "bridges/panel.parquet (via model.load_panel)"
    except Exception:
        from .fake_data import make_panel

        panel = make_panel()
        source = "synthetic fake_data.make_panel()"

    r = run(panel)
    print(f"source: {source}")
    print(f"split: {r['n_train_bridges']} train / {r['n_test_bridges']} test "
          f"bridges (seed {TEST_SEED}, {int(TEST_FRACTION*100)}% test)")
    print(f"test pairs: {r['n_test_pairs']}  positives (safety drops): "
          f"{r['test_positives']}")
    print()
    print(f"{'model':<12}{'positives':>11}{'log loss':>12}{'AUC':>10}")
    for name in ("grouped", "logistic", "baseline"):
        m = r["models"][name]
        print(f"{name:<12}{m['positives']:>11}{m['log_loss']:>12.4f}{m['auc']:>10.4f}")
    print(f"\nbest test AUC -> score() uses: {r['best_model']}")

    print("\ndrop rate by START condition (held-out test pairs):")
    for cond in sorted(r["drop_rate_by_cond"]):
        d = r["drop_rate_by_cond"][cond]
        print(f"  cond {cond}: {d['mean']*100:5.1f}%  (n={int(d['count'])})")


if __name__ == "__main__":
    main()
