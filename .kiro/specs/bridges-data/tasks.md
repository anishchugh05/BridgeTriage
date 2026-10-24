# Tasks - Person A Data Loading

- [x] 1. Add pyarrow to requirements.txt.
- [x] 2. Fix .gitignore: ignore data/ except .gitkeep, never ignore .kiro/.
- [x] 3. Write bridges/make_fake.py, run it, produce bridges/fake.parquet.
- [x] 4. Commit and push fake.parquet first to unblock B and C.
- [x] 5. Download the four raw files into data/ (2012, 2016, 2018, 2022),
      report OK or FAIL per URL.
- [x] 6. Verify the 2022 header, confirm item positions, adjust the item map if
      county, carries, or crosses differ.
- [x] 7. Write bridges/loader.py with load_year, build_panel, main.
- [x] 8. Run load_year(2022), sanity check, write bridges/bridges_2022.parquet.
- [x] 9. Build and write bridges/panel.parquet with rebuilt flagged.
- [x] 10. Commit and push the loader and both real parquet files.
- [x] 11. Fix the rebuilt rule: a first appearance is never rebuilt. Rebuilt is
      True only when, versus the same bridge previous appearance, a deck, super,
      or sub rating rose or year_recon changed. 2012 count is now 0.
- [x] 12. Add run_checks: no duplicate (structure_id, year), lat/lon inside
      Massachusetts with null counts per year, ratings 0 to 9 or null, age not
      negative, row counts per year. Print failing rows up to 10.
- [x] 13. Add write_summary to bridges/SUMMARY.md: worst_rating distribution and
      null counts per year, top 10 lowest 2022 bridges, drop share 2012 to 2016
      and 2018 to 2022 excluding rebuilt.
- [x] 14. Write bridges/README.md for teammates (schema, definitions, how to
      load, which file, uneven gaps, null and rebuilt warnings).
- [x] 15. Update the spec and push to origin/person-a-data.
