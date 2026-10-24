# Stub versions of B and C so the app runs before their code lands.
# Same signatures and return columns as the real functions. The app imports
# the real ones first and falls back to these on any import error.
import numpy as np
import pandas as pd

METHODS = ["ours", "worst_first", "oldest_first"]


def _seed_from_id(sid):
    # Deterministic per bridge so the demo does not jump around on rerun.
    return abs(hash(str(sid))) % (2 ** 32)


def score(table):
    # Returns one row per bridge: structure_id, p_worse, consequence, insp_cost,
    # score, reason. Deterministic from the row values.
    rows = []
    for _, r in table.iterrows():
        rng = np.random.default_rng(_seed_from_id(r["structure_id"]))
        worst = r.get("worst_rating")
        worst = 5 if pd.isna(worst) else float(worst)
        age = r.get("age")
        age = 50 if pd.isna(age) else float(age)
        adt = r.get("adt")
        adt = 5000 if pd.isna(adt) else float(adt)

        # Lower condition and higher age push the chance of getting worse up.
        p_worse = min(0.98, max(0.02,
                      0.5 * (9 - worst) / 9 + 0.3 * min(age, 100) / 100
                      + 0.2 * rng.random()))
        # Consequence scales with traffic and hazards.
        hazard = bool(r.get("fracture_critical")) or bool(r.get("underwater"))
        consequence = np.log1p(adt) * (1.4 if hazard else 1.0)
        insp_cost = 4000 + 600 * (1 if hazard else 0) + int(rng.integers(0, 2000))
        sc = p_worse * consequence
        reason = f"condition {int(worst)}, age {int(age)}, adt {int(adt)}"
        if hazard:
            reason = "hazard to inspect, " + reason
        rows.append({
            "structure_id": r["structure_id"],
            "p_worse": round(float(p_worse), 4),
            "consequence": round(float(consequence), 3),
            "insp_cost": int(insp_cost),
            "score": round(float(sc), 3),
            "reason": reason,
        })
    return pd.DataFrame(rows)


def pick(scores, budget):
    # Greedy by score per cost until the budget runs out. Returns chosen ids.
    s = scores.copy()
    s["ratio"] = s["score"] / s["insp_cost"].clip(lower=1)
    s = s.sort_values(["ratio", "score"], ascending=False)
    chosen, spent = [], 0
    for _, r in s.iterrows():
        if spent + r["insp_cost"] <= budget:
            chosen.append(r["structure_id"])
            spent += r["insp_cost"]
    return chosen


def backtest(scores=None, budgets=None):
    # Returns budget, method, caught for the three methods. ours beats the rest.
    if budgets is None:
        budgets = list(range(100_000, 1_100_000, 100_000))
    rng = np.random.default_rng(7)
    rows = []
    for b in budgets:
        base = 8 + b / 60_000
        caught = {
            "ours": base * 1.0,
            "worst_first": base * 0.72,
            "oldest_first": base * 0.55,
        }
        for m in METHODS:
            jitter = rng.normal(0, 0.8)
            rows.append({
                "budget": int(b),
                "method": m,
                "caught": max(0, int(round(caught[m] + jitter))),
            })
    return pd.DataFrame(rows)