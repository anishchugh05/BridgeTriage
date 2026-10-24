# Requirements - Person A Data Loading

## Goal

Turn the messy per year FHWA NBI Massachusetts files into one clean, stable
table that B (decline model) and C (picker and backtest) build on. The column
set is a fixed contract. Once published it does not change.

## Output schema (contract)

These 20 columns, in this order, never renamed or dropped:

structure_id, year, county, material, age, adt, deck, super, sub,
worst_rating, insp_freq_months, fracture_critical, underwater,
year_recon, rebuilt, lat, lon, carries, crosses, improvement_cost

## Shared definitions

- worst_rating is the min of deck, super, sub.
- Drop: worst_rating falls by 1 or more between two years a bridge appears.
- Rebuilt: excluded from training (B) and backtest (C). Flag rebuilt = True
  when any rating went up vs the previous year the bridge appears, or year_recon
  changed.

## Functional requirements

1. Deliver bridges/fake.parquet first: 200 fake rows in the exact schema so B
   and C can start before the real data is clean.
2. load_year(year) cleans any of 2012, 2016, 2018, 2022 into the schema.
   - Read as strings, latin-1, comma separated, single quote text qualifier.
   - Match columns by NBI item suffix (like _008, _058), not by name, so yearly
     renames do not break the loader.
   - Print which wanted items are missing per year.
   - Ratings items 58, 59, 60: "N" or blank is missing, else int 0 to 9.
   - Lat item 16 and lon item 17: packed DMS digits to decimal degrees, lon made
     negative, anything outside Massachusetts set to missing.
   - Derive age = year minus year_built, worst_rating = min of three ratings.
3. build_panel joins 2012, 2016, 2018, 2022 on structure_id, reports consecutive
   year match rate and added/dropped counts, flags rebuilt per the definition,
   and prints rebuilt counts per year.
4. Write bridges/bridges_2022.parquet and bridges/panel.parquet.
5. Add pyarrow to requirements.txt.

## Constraints

- Raw NBI files live in data/ and are gitignored. Never commit them.
- .gitignore keeps data/ ignored and never ignores .kiro/.
- Keep code short and plain. No em dashes or en dashes in text or comments.
- Verify item positions against the real 2022 header before trusting them.
