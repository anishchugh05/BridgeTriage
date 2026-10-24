"""Cross-validated model comparison, selection, and calibration.

Five folds are grouped by bridge id over the 2012 -> 2016 interval. Each
eligible pair appears in exactly one held-out fold. The harness compares:

* condition-only baseline
* grouped material x age-band x condition model
* logistic regression without months_since_insp
* logistic regression with months_since_insp

Model selection uses mean CV AUC and prefers the simpler model unless a more
complex model improves the current choice by more than one CV-fold standard
deviation. Calibration is computed from pooled out-of-fold predictions, never
from in-sample fitted values.

Run:
    .venv/bin/python -m model.validate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FOUR_YEAR_PAIRS, load_panel
from .score import (
    _build_pairs,
    fit_condition_baseline,
    fit_decline,
    fit_logistic,
)

FIT_START, FIT_END = 2012, 2016
CV_SEED = 1234
N_FOLDS = 5

# Increasing complexity, used by the one-standard-deviation selection rule.
MODEL_ORDER = ["baseline", "grouped", "logistic_no_msi", "logistic_msi"]
MODEL_LABELS = {
    "baseline": "baseline",
    "grouped": "grouped",
    "logistic_no_msi": "logistic (no msi)",
    "logistic_msi": "logistic (+ msi)",
}


def _log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney); 0.5 when either class is absent."""
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
         for r in eval_df.itertuples()],
        dtype=float,
    )


def _baseline_probs(train_eval: pd.DataFrame, test_eval: pd.DataFrame) -> np.ndarray:
    per = train_eval.groupby("cond")["dropped"].mean().to_dict()
    overall = float(train_eval["dropped"].mean())
    return np.array([per.get(int(c), overall) for c in test_eval["cond"]])


def _fold_assignments(ids: np.ndarray, n_folds: int, seed: int) -> dict:
    """Assign every eligible bridge id to one deterministic fold."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(ids) < n_folds:
        raise ValueError(
            f"need at least {n_folds} eligible bridges; got {len(ids)}"
        )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.sort(ids))
    return {bid: i % n_folds for i, bid in enumerate(shuffled)}


def cross_validate(
    panel: pd.DataFrame,
    start: int = FIT_START,
    end: int = FIT_END,
    n_folds: int = N_FOLDS,
    seed: int = CV_SEED,
) -> dict:
    """Run grouped CV and retain per-fold metrics plus OOF predictions."""
    eligible = _build_pairs(panel, start, end)
    if eligible.empty:
        raise ValueError("no eligible bridge pairs for cross-validation")
    fold_of = _fold_assignments(eligible["id"].unique(), n_folds, seed)

    metrics = {k: {"auc": [], "log_loss": []} for k in MODEL_ORDER}
    oof_parts = []

    for fold in range(n_folds):
        test_ids = {bid for bid, f in fold_of.items() if f == fold}
        train_ids = set(fold_of) - test_ids
        train_panel = panel[panel["id"].isin(train_ids)]
        train_eval = eligible[eligible["id"].isin(train_ids)]
        test_eval = eligible[eligible["id"].isin(test_ids)].copy()
        if test_eval.empty:
            raise ValueError(f"fold {fold} has no eligible test pairs")

        y = test_eval["dropped"].to_numpy()
        grouped = fit_decline(train_panel, start, end)
        logit_no = fit_logistic(
            train_panel, start, end, use_months_since_insp=False
        )
        logit_yes = fit_logistic(
            train_panel, start, end, use_months_since_insp=True
        )
        preds = {
            "baseline": _baseline_probs(train_eval, test_eval),
            "grouped": _grouped_probs(grouped, test_eval),
            "logistic_no_msi": logit_no.interval_prob_frame(test_eval),
            "logistic_msi": logit_yes.interval_prob_frame(test_eval),
        }

        fold_oof = pd.DataFrame(
            {
                "id": test_eval["id"].to_numpy(),
                "fold": fold,
                "actual": y,
            }
        )
        for key, pred in preds.items():
            pred = np.asarray(pred, dtype=float)
            metrics[key]["auc"].append(_auc(y, pred))
            metrics[key]["log_loss"].append(_log_loss(y, pred))
            fold_oof[key] = pred
        oof_parts.append(fold_oof)

    summary = {}
    for key in MODEL_ORDER:
        aucs = np.asarray(metrics[key]["auc"], dtype=float)
        losses = np.asarray(metrics[key]["log_loss"], dtype=float)
        summary[key] = {
            "auc_mean": float(aucs.mean()),
            "auc_std": float(aucs.std(ddof=0)),
            "ll_mean": float(losses.mean()),
            "ll_std": float(losses.std(ddof=0)),
        }

    oof = pd.concat(oof_parts, ignore_index=True)
    return {
        "summary": summary,
        "n_folds": n_folds,
        "evaluated_folds": len(oof_parts),
        "total_test_pairs": len(oof),
        "total_positives": int(oof["actual"].sum()),
        "oof": oof,
    }


def _pick_by_cv(summary: dict) -> str:
    """Prefer simpler models unless added complexity wins by > one std."""
    best = MODEL_ORDER[0]
    for candidate in MODEL_ORDER[1:]:
        candidate_auc = summary[candidate]["auc_mean"]
        best_auc = summary[best]["auc_mean"]
        best_std = summary[best]["auc_std"]
        if candidate_auc > best_auc + best_std:
            best = candidate
    return best


def _fit_selected(panel: pd.DataFrame, start: int, end: int, key: str):
    if key == "logistic_msi":
        return fit_logistic(panel, start, end, use_months_since_insp=True)
    if key == "logistic_no_msi":
        return fit_logistic(panel, start, end, use_months_since_insp=False)
    if key == "grouped":
        return fit_decline(panel, start, end)
    if key == "baseline":
        return fit_condition_baseline(panel, start, end)
    raise ValueError(f"unknown model key: {key}")


def select_model(
    panel: pd.DataFrame,
    start: int = FIT_START,
    end: int = FIT_END,
    n_folds: int = N_FOLDS,
    seed: int = CV_SEED,
):
    """Select by grouped CV and refit the exact selected family on all pairs."""
    cv = cross_validate(panel, start, end, n_folds, seed)
    chosen = _pick_by_cv(cv["summary"])
    model = _fit_selected(panel, start, end, chosen)
    return model, chosen, cv["summary"]


def calibration_table(oof: pd.DataFrame, model_key: str,
                      n_bins: int = 5) -> pd.DataFrame:
    """Quantile calibration table from out-of-fold horizon predictions.

    Returns exactly ``bin, n, mean_predicted, actual_drop_rate``. Ranking before
    qcut guarantees five near-equal-count bins even when predictions are tied.
    """
    if model_key not in oof.columns:
        raise KeyError(f"missing OOF predictions for {model_key}")
    if len(oof) < n_bins:
        raise ValueError(f"need at least {n_bins} rows for {n_bins} bins")
    frame = oof[[model_key, "actual"]].rename(
        columns={model_key: "predicted"}
    ).copy()
    ranks = frame["predicted"].rank(method="first")
    frame["bin"] = pd.qcut(ranks, q=n_bins, labels=False) + 1
    result = (
        frame.groupby("bin", observed=True)
        .agg(
            n=("actual", "size"),
            mean_predicted=("predicted", "mean"),
            actual_drop_rate=("actual", "mean"),
        )
        .reset_index()
    )
    result["bin"] = result["bin"].astype(int)
    return result[["bin", "n", "mean_predicted", "actual_drop_rate"]]


def _drop_rate_by_msi_bucket(pairs: pd.DataFrame) -> pd.DataFrame:
    edges = [0, 12, 24, 36, np.inf]
    labels = ["0-12", "12-24", "24-36", "36+"]
    buckets = pd.cut(
        pairs["months_since_insp"], bins=edges, labels=labels, right=False
    )
    return pairs.groupby(buckets, observed=False)["dropped"].agg(["mean", "count"])


def main() -> None:
    try:
        panel = load_panel()
        source = "bridges/panel.parquet (via model.load_panel)"
    except Exception:
        from .fake_data import make_panel

        panel = make_panel()
        source = "synthetic fake_data.make_panel()"

    cv = cross_validate(panel)
    summary = cv["summary"]
    chosen = _pick_by_cv(summary)
    selected = _fit_selected(panel, FIT_START, FIT_END, chosen)

    print(f"source: {source}")
    print(
        f"{cv['evaluated_folds']}-fold CV grouped by bridge id | "
        f"pooled test pairs {cv['total_test_pairs']}, "
        f"positives {cv['total_positives']}"
    )
    print()
    print(f"{'model':<20}{'AUC (mean±std)':>20}{'log loss (mean±std)':>24}")
    for key in MODEL_ORDER:
        metric = summary[key]
        auc = f"{metric['auc_mean']:.4f} ± {metric['auc_std']:.4f}"
        loss = f"{metric['ll_mean']:.4f} ± {metric['ll_std']:.4f}"
        print(f"{MODEL_LABELS[key]:<20}{auc:>20}{loss:>24}")
    print(
        "\nselected by mean CV AUC (simpler preferred unless +1 std): "
        f"{MODEL_LABELS[chosen]}"
    )

    print("\nselected-model calibration (OOF horizon probability):")
    calibration = calibration_table(cv["oof"], chosen)
    print(calibration.to_string(index=False, formatters={
        "mean_predicted": lambda x: f"{x:.4f}",
        "actual_drop_rate": lambda x: f"{x:.4f}",
    }))

    if hasattr(selected, "coefficients"):
        print("\nselected logistic coefficients (sorted by |value|):")
        for name, value in selected.coefficients().items():
            print(f"  {name:<32}{value:+.4f}")
    else:
        print("\nselected model is not logistic; no logistic coefficients.")

    all_pairs = pd.concat(
        [_build_pairs(panel, start, end) for start, end in FOUR_YEAR_PAIRS],
        ignore_index=True,
    )
    print("\nsafety-drop rate by months_since_insp bucket (all 4yr pairs):")
    for bucket, row in _drop_rate_by_msi_bucket(all_pairs).iterrows():
        print(
            f"  {str(bucket):<7} {row['mean'] * 100:5.1f}%  "
            f"(n={int(row['count'])})"
        )


if __name__ == "__main__":
    main()
