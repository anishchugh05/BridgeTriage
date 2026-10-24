"""Validation harness for the decline model (Part B).

5-fold cross-validation **grouped by bridge id** (a bridge is never split across
train and test) over the 2012 -> 2016 interval. The target is the shared safety
drop (see :mod:`model.target`). For each fold we fit on 4/5 of bridges and score
the held-out 1/5, for four models:

    - grouped          : fit_decline (material x age band x condition, shrunk)
    - logistic (no msi): fit_logistic WITHOUT months_since_insp
    - logistic (+ msi) : fit_logistic WITH months_since_insp
    - baseline         : per-condition drop rate only

and report mean +/- std of AUC and log loss across folds.

``select_model`` picks by mean CV AUC, preferring the simpler model unless a
more complex one wins by more than one standard deviation. Complexity order:
baseline < grouped < logistic(no msi) < logistic(+ msi).

Run:
    .venv/bin/python -m model.validate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_panel
from .score import _build_pairs, fit_decline, fit_logistic

FIT_START, FIT_END = 2012, 2016
CV_SEED = 1234
N_FOLDS = 5

# Model keys in increasing complexity order (used for the tie-break rule).
MODEL_ORDER = ["baseline", "grouped", "logistic_no_msi", "logistic_msi"]


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


def _grouped_probs(rates, eval_df: pd.DataFrame) -> np.ndarray:
    return np.array(
        [rates.prob_for(r.material, r.age_band, r.cond)
         for r in eval_df.itertuples()]
    )


def _baseline_probs(train_eval: pd.DataFrame, test_eval: pd.DataFrame) -> np.ndarray:
    per = train_eval.groupby("cond")["dropped"].mean().to_dict()
    overall = float(train_eval["dropped"].mean())
    return np.array([per.get(int(c), overall) for c in test_eval["cond"]])


def _fold_assignments(ids: np.ndarray, n_folds: int, seed: int) -> dict:
    """Map each bridge id to a fold index 0..n_folds-1 (grouped by id)."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.sort(ids))
    folds = {}
    for i, bid in enumerate(shuffled):
        folds[bid] = i % n_folds
    return folds


def cross_validate(panel: pd.DataFrame, start: int = FIT_START,
                   end: int = FIT_END, n_folds: int = N_FOLDS,
                   seed: int = CV_SEED) -> dict:
    """Grouped 5-fold CV. Returns per-model lists of fold AUC and log loss."""
    ids = panel["id"].unique()
    fold_of = _fold_assignments(ids, n_folds, seed)

    metrics = {k: {"auc": [], "log_loss": []} for k in MODEL_ORDER}
    total_pairs = 0
    total_pos = 0

    for f in range(n_folds):
        test_ids = {bid for bid, fi in fold_of.items() if fi == f}
        train_ids = {bid for bid, fi in fold_of.items() if fi != f}
        train_panel = panel[panel["id"].isin(train_ids)]
        test_eval = _build_pairs(panel[panel["id"].isin(test_ids)], start, end)
        train_eval = _build_pairs(train_panel, start, end)
        if test_eval.empty or test_eval["dropped"].nunique() < 2:
            continue
        y = test_eval["dropped"].to_numpy()
        total_pairs += len(test_eval)
        total_pos += int(y.sum())

        grouped = fit_decline(train_panel, start, end)
        logit_no = fit_logistic(train_panel, start, end,
                                use_months_since_insp=False)
        logit_yes = fit_logistic(train_panel, start, end,
                                 use_months_since_insp=True)

        preds = {
            "baseline": _baseline_probs(train_eval, test_eval),
            "grouped": _grouped_probs(grouped, test_eval),
            "logistic_no_msi": logit_no.interval_prob_frame(test_eval),
            "logistic_msi": logit_yes.interval_prob_frame(test_eval),
        }
        for k, p in preds.items():
            metrics[k]["auc"].append(_auc(y, p))
            metrics[k]["log_loss"].append(_log_loss(y, p))

    summary = {}
    for k in MODEL_ORDER:
        aucs = np.array(metrics[k]["auc"])
        lls = np.array(metrics[k]["log_loss"])
        summary[k] = {
            "auc_mean": float(aucs.mean()) if len(aucs) else float("nan"),
            "auc_std": float(aucs.std(ddof=0)) if len(aucs) else float("nan"),
            "ll_mean": float(lls.mean()) if len(lls) else float("nan"),
            "ll_std": float(lls.std(ddof=0)) if len(lls) else float("nan"),
        }
    return {
        "summary": summary,
        "n_folds": n_folds,
        "total_test_pairs": total_pairs,
        "total_positives": total_pos,
    }


def _pick_by_cv(summary: dict) -> str:
    """Prefer the simpler model unless a more complex one beats it by > 1 std.

    Walk complexity order; adopt a more complex model only when its mean AUC
    exceeds the current pick's mean AUC by more than the current pick's std.
    """
    best = MODEL_ORDER[0]
    for cand in MODEL_ORDER[1:]:
        cand_auc = summary[cand]["auc_mean"]
        best_auc = summary[best]["auc_mean"]
        best_std = summary[best]["auc_std"]
        if cand_auc > best_auc + best_std:
            best = cand
    return best


def select_model(panel: pd.DataFrame, start: int = FIT_START, end: int = FIT_END,
                 n_folds: int = N_FOLDS, seed: int = CV_SEED):
    """Pick the model family by CV (simpler-preferred) and refit it on all data.

    Returns ``(model, chosen_key, cv_summary)``. The grouped/logistic model is
    refit on every bridge in the interval; the per-condition baseline, if it
    wins, is returned as a grouped model (its closest equivalent) so callers
    always get a scorable object.
    """
    cv = cross_validate(panel, start, end, n_folds, seed)
    chosen = _pick_by_cv(cv["summary"])
    if chosen == "logistic_msi":
        model = fit_logistic(panel, start, end, use_months_since_insp=True)
    elif chosen == "logistic_no_msi":
        model = fit_logistic(panel, start, end, use_months_since_insp=False)
    else:
        # "grouped" and "baseline" both map to the grouped model (the baseline
        # is a degenerate grouped model on condition only).
        model = fit_decline(panel, start, end)
    return model, chosen, cv["summary"]


def _drop_rate_by_msi_bucket(pairs: pd.DataFrame) -> pd.DataFrame:
    """Safety-drop rate by months_since_insp bucket: 0-12, 12-24, 24-36, 36+."""
    edges = [0, 12, 24, 36, np.inf]
    labels = ["0-12", "12-24", "24-36", "36+"]
    buckets = pd.cut(pairs["months_since_insp"], bins=edges, labels=labels,
                     right=False)
    g = pairs.groupby(buckets, observed=False)["dropped"].agg(["mean", "count"])
    return g


def main() -> None:
    try:
        panel = load_panel()
        source = "bridges/panel.parquet (via model.load_panel)"
    except Exception:
        from .fake_data import make_panel

        panel = make_panel()
        source = "synthetic fake_data.make_panel()"

    cv = cross_validate(panel)
    s = cv["summary"]
    print(f"source: {source}")
    print(f"{cv['n_folds']}-fold CV grouped by bridge id | "
          f"pooled test pairs {cv['total_test_pairs']}, "
          f"positives {cv['total_positives']}")
    print()
    print(f"{'model':<20}{'AUC (mean±std)':>20}{'log loss (mean±std)':>24}")
    label = {
        "baseline": "baseline",
        "grouped": "grouped",
        "logistic_no_msi": "logistic (no msi)",
        "logistic_msi": "logistic (+ msi)",
    }
    for k in MODEL_ORDER:
        m = s[k]
        auc = f"{m['auc_mean']:.4f} ± {m['auc_std']:.4f}"
        ll = f"{m['ll_mean']:.4f} ± {m['ll_std']:.4f}"
        print(f"{label[k]:<20}{auc:>20}{ll:>24}")

    chosen = _pick_by_cv(s)
    print(f"\nselected by mean CV AUC (simpler preferred unless +1 std): "
          f"{label[chosen]}")

    # Fit the selected model on all bridges; if it's a logistic, print coefs.
    model, chosen_key, _ = select_model(panel)
    if hasattr(model, "coefficients"):
        print("\nselected logistic coefficients (sorted by |value|):")
        for name, val in model.coefficients().items():
            print(f"  {name:<28}{val:+.4f}")
    else:
        print("\nselected model is grouped; no logistic coefficients to show.")

    # Safety-drop rate by months_since_insp bucket, over all 4-year pairs.
    from .data import FOUR_YEAR_PAIRS
    allp = pd.concat(
        [_build_pairs(panel, a, b) for a, b in FOUR_YEAR_PAIRS],
        ignore_index=True,
    )
    print("\nsafety-drop rate by months_since_insp bucket (all 4yr pairs):")
    byb = _drop_rate_by_msi_bucket(allp)
    for bucket, row in byb.iterrows():
        print(f"  {str(bucket):<7} {row['mean']*100:5.1f}%  (n={int(row['count'])})")


if __name__ == "__main__":
    main()
