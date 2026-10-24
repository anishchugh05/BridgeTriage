# Requirements - Person A Data Loading

## Goal

Turn the messy per year FHWA NBI Massachusetts files into one clean, stable
table that B (decline model) and C (picker and backtest) build on. The column
set is a fixed contract. Once published it does not change.

## Output schema (contract)

These 21 columns, in this order, never renamed or dropped:

structure_id, year, county, material, age, adt, deck, super, sub,
worst_rating, insp_freq_months, fracture_critical, underwater,
year_recon, rebuilt, lat, lon, carries, crosses, improvement_cost,
insp_date

insp_date is item 090 (date of last inspection, MMYY) decoded to a datetime,
appended as column 21 so existing column positions do not shift.

## Shared definitions

- worst_rating is the min of deck, super, sub.
- Drop: worst_rating falls by 1 or more between two years a bridge appears.
- Rebuilt: versus the same bridge previous appearance, a deck, super, or sub
  rating rose, or year_recon changed. A first appearance is never rebuilt.
  Excluded from training (B) and backtest (C).

## Functional requirements

1. Deliver bridges/fake.parquet first: 200 fake rows in the exact schema so B
   and C can start before the real data is clean.
2. load_year(year) cleans any of 2012, 2016, 2018, 2022 into the schema.
   - Read as strings, latin-1, comma separated, single quote text qualifier.
   - Match columns by NBI item suffix (like _008, _058), not by name.
   - Print which wanted items are missing per year.
   - Ratings items 58, 59, 60: "N" or blank is missing, else int 0 to 9.
   - Lat item 16 (8 digit DMS) and lon item 17 (9 digit DMS): decimal degrees,
     lon negative, anything outside Massachusetts set to missing.
   - county item 003 is a numeric FIPS code mapped to a county name.
   - Derive age = year minus year_built, worst_rating = min of three ratings.
3. build_panel joins the four years on structure_id, reports consecutive year
   match rate and added/dropped counts, drops rare colliding
   (structure_id, year) duplicates so the join key is unique, flags rebuilt per
   the definition, and prints rebuilt counts per year.
4. run_checks prints PASS or FAIL for: no duplicate (structure_id, year),
   lat/lon inside Massachusetts with null counts per year, ratings 0 to 9 or
   null, age not negative, and row counts per year. Failing rows printed up to
   10.
5. write_summary writes bridges/SUMMARY.md: worst_rating distribution and null
   counts per year, top 10 lowest 2022 bridges, and drop share 2012 to 2016 and
   2018 to 2022 excluding rebuilt.
6. bridges/README.md documents the schema, definitions, how to load, which file
   to use, and the uneven gap, null worst_rating, and rebuilt warnings.
7. Write bridges/bridges_2022.parquet and bridges/panel.parquet.
8. Add pyarrow to requirements.txt.

## Constraints

- Raw NBI files live in data/ and are gitignored. Never commit them.
- .gitignore keeps data/ ignored and never ignores .kiro/.
- Keep code short and plain. No em dashes or en dashes in text or comments.
- Verify item positions against the real 2022 header before trusting them.
