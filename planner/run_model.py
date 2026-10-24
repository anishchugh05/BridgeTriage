import pandas as pd, numpy as np
from planner.backtest import *

s = pd.read_parquet('data/processed/scores_2018.parquet').rename(columns={'id': 'structure_id'})
s['structure_id'] = s['structure_id'].astype(str).str.strip()
print(s.shape, 'unique ids:', s.structure_id.is_unique)
print(s[['p_worse', 'consequence', 'insp_cost', 'flight_minutes', 'expected_drops_per_10k', 'score']].describe().round(4))
print('score == expected_drops_per_10k:', np.allclose(s.score, s.expected_drops_per_10k))

d18, d22 = adapt_panel()
c, o, p = apply_exclusion_and_label(*load_and_validate(d18, d22), 5.0)
scores = dict(zip(s.structure_id, s.score.astype(float)))
missing = [i for i in c.index if i not in scores]
print('pool', len(c), '| covered', len(c) - len(missing), '| missing score', len(missing))
assert not missing, 'model has no score for some pool bridges; do not read results'

r = run_backtest(d18, d22, 5.0, model_scores=scores)
keep = r[(r.method != 'random_avg') | (r.is_aggregate & (r.stat == 'mean'))]
for m in ['recall', 'weighted_recall']:
    print(m)
    print(keep.pivot_table(index='method', columns='budget', values=m).round(3))
plot(r, '/tmp/model')
