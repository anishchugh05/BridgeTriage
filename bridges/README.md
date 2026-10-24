# Bridges data contract (Person A)

Clean Massachusetts NBI tables for B (decline model) and C (picker and backtest).
Do not rename or drop columns. The schema is fixed once published.

## Files

- panel.parquet: all four years (2012, 2016, 2018, 2022) joined. Use this for
  real work: training and backtest.
- bridges_2022.parquet: the single cleaned 2022 year.
- fake.parquet: 200 fake rows in the exact schema. Use this to wire up your code
  before the real data is ready.

Load with pandas:

    import pandas as pd
    panel = pd.read_parquet("bridges/panel.parquet")

## Schema (21 columns, fixed order)

structure_id, year, county, material, age, adt, deck, super, sub,
worst_rating, insp_freq_months, fracture_critical, underwater,
year_recon, rebuilt, lat, lon, carries, crosses, improvement_cost,
insp_date

- structure_id: NBI structure number, the join key across years.
- year: 2012, 2016, 2018, or 2022.
- county: Massachusetts county name (mapped from the FIPS code).
- material: structure kind (item 043A).
- age: year minus year built.
- adt: average daily traffic.
- deck, super, sub: condition ratings 0 to 9, or null.
- worst_rating: min of deck, super, sub. Null only when all three are null.
- insp_freq_months: inspection frequency.
- fracture_critical, underwater: booleans.
- year_recon: year reconstructed, 0 means never.
- rebuilt: see definition below.
- lat, lon: decimal degrees inside Massachusetts, or null.
- carries, crosses: what the bridge carries and what it crosses.
- improvement_cost: bridge improvement cost (item 094). Repair cost, not
  inspection cost.
- insp_date: date of last inspection (item 090, MMYY) as a datetime, month
  start. Use it for time since inspection in the score.

## Definitions

- worst_rating: the min of deck, super, sub. One number for a bridge condition.
- Drop: worst_rating falls by 1 or more between two appearances of the same
  bridge. This is what we predict and try to catch.
- Rebuilt: versus the same bridge previous appearance, a deck, super, or sub
  rating rose, or year_recon changed. A first appearance is never rebuilt.
  Rebuilt is not natural decline, so exclude these rows.

## Warnings

- Year gaps are uneven: 2012 to 2016 is 4 years, 2016 to 2018 is 2 years,
  2018 to 2022 is 4 years. Train and backtest should both use the 4 year gaps
  so the horizon matches.
- Null worst_rating rows (culverts, dam spillways, and similar) must be dropped
  or handled on purpose. There are about 300 per year.
- Exclude rebuilt rows from training and from the backtest.

## Handoff for B and C

### Getting the data

    git fetch origin
    git checkout person-a-data
    git pull
    pip install -r requirements.txt

Load the panel with pd.read_parquet("bridges/panel.parquet"). Wire your code
against fake.parquet first, then swap to the real panel. Both files carry the
same 21 columns in the same order.

### Person B (decline model, model/)

- Read panel.parquet. Drop rebuilt rows and drop null worst_rating rows.
- Label per the drop definition: worst_rating falls by 1 or more between two
  appearances of the same bridge.
- Use only the 4 year gaps (2012 to 2016, 2018 to 2022) so the horizon matches.
- insp_date is a datetime. Compute time since inspection as asof minus
  insp_date. This is the "time since inspection" input to the score.
- The 2012 rows all have rebuilt = False by construction (no earlier year to
  compare), so do not read the 2012 rebuilt column as a full history flag.
- Suggested signatures: fit_decline(panel, start, end) and
  score(table, rates, asof).

### Person C (picker and backtest, planner/, app/)

- Read panel.parquet and call B's score. Join on structure_id, which is unique
  per year.
- Backtest on a 4 year gap: fit on up to 2018, pick using only 2018 and earlier
  information, then check against the actual 2022 drops. Exclude rebuilt rows.
- improvement_cost is repair cost, not inspection cost. There is no per bridge
  inspection cost in NBI, so use your own insp_cost placeholder or rate.
- lat and lon are decimal degrees ready for mapping. county is a readable name
  ready for grouping.

### Schema change policy

The 21 columns are a contract. If you need a new field, ask Person A to add it
(appended at the end so positions do not shift). Do not rename columns on your
side.
## Notes

- The 2022 raw file has one colliding structure number. That duplicate is
  dropped so (structure_id, year) is unique.
- Rebuilt counts per year: 2012 = 0, 2016 = 518, 2018 = 325, 2022 = 394.
- See SUMMARY.md for distributions and drop shares.
