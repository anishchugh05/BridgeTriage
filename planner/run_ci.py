import pandas as pd, numpy as np
from planner.backtest import *

s = pd.read_parquet('data/processed/scores_2018.parquet').rename(columns={'id': 'structure_id'})
s['structure_id'] = s['structure_id'].astype(str).str.strip()
scores = dict(zip(s.structure_id, s.score.astype(float)))
d18, d22 = adapt_panel()
c, o, p = apply_exclusion_and_label(*load_and_validate(d18, d22), 5.0)
adt = c['adt'].to_dict()
budgets = [0.01, 0.02, 0.05, 0.10, 0.20]
sel = {
    'worst_first': {b: pick(score_worst_first(c), b, c, p, 5.0) for b in budgets},
    'model': {b: pick(scores, b, c, p, 5.0, rank_by_value_per_cost=True) for b in budgets},
}
for metric in ['recall', 'weighted_recall']:
    ci = bootstrap_gap_ci(sel, o, adt, metric)
    print(metric, '- model minus worst_first, 95% bootstrap CI')
    for b in budgets:
        lo, hi = ci['model'][b]
        print(f'  budget {b:.2f}: ({lo:+.3f}, {hi:+.3f})', '<- excludes 0' if lo > 0 or hi < 0 else '')
print('decliners in pool:', sum(v['worse'] for v in o.values()))
