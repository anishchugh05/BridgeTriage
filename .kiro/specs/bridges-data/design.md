# Design - Person A Data Loading

## Data flow

Raw FHWA .txt files (ugly, per year)
  -> load_year(year)   clean each year to the contract schema
  -> build_panel()     join four years, flag rebuilt
  -> bridges/*.parquet (the contract)
       -> B score(table)  decline model
       -> C pick(scores)  picker and backtest

## Modules

- bridges/make_fake.py: writes 200 fake rows to bridges/fake.parquet in the
  exact schema. No real data needed. Runs first to unblock the team.
- bridges/loader.py: load_year(year), build_panel(years), main().

## Column matching

Follow fetch_years.py: match by NBI item suffix written as _NNN using
str.strip().endswith("_" + suffix). Read as strings, latin-1, low_memory=False,
sep="," , quotechar="'". This survives the yearly column renames.

## Item map (verified against the real 2022 header)

structure_id 008, county 003, crosses 006A, carries 007, lat 016, lon 017,
year_built 027, adt 029, material 043A, deck 058, super 059, sub 060,
insp_date 090, insp_freq_months 091, fracture_critical 092A, underwater 092B,
improvement_cost 094, year_recon 106.

county item 003 is a numeric FIPS county code in the raw file, so the loader
maps it to a readable name for C's grouping. The reduced bridges/data.csv shows
names, the raw MA22.txt shows codes.

## Cleaning rules

- Ratings (058, 059, 060): strip, upper. "" or "N" is None, else int if 0 to 9.
- DMS lat/lon: strip non digits, decode DD MM SS ss (8 digits), lon negative.
  Verify the packing width against the real file before trusting the slicing.
  Outside Massachusetts bounds is set to None.
- age = year minus year_built when year_built is present.
- worst_rating = min of deck, super, sub skipping missing.
- fracture_critical, underwater: true when the code is Y or 1.
- year_recon: int, 0 means never.

## Panel and rebuilt

Concat the four cleaned frames, sort by structure_id then year. For each bridge
in year order, flag rebuilt when worst_rating rose vs the previous appearance or
year_recon changed. Report consecutive year match rate, added and dropped
counts, and rebuilt counts per year.

## Deliverable order

1. bridges/fake.parquet (push first)
2. bridges/bridges_2022.parquet
3. bridges/panel.parquet
