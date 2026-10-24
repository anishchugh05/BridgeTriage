# Tasks - Person A Data Loading

- [ ] 1. Add pyarrow to requirements.txt.
- [ ] 2. Fix .gitignore: ignore data/ except .gitkeep, never ignore .kiro/.
- [ ] 3. Write bridges/make_fake.py, run it, produce bridges/fake.parquet.
- [ ] 4. Commit and push fake.parquet first to unblock B and C.
- [ ] 5. Download the four raw files into data/ (2012, 2016, 2018, 2022),
      report OK or FAIL per URL.
- [ ] 6. Verify the 2022 header, confirm item positions, adjust the item map if
      county, carries, or crosses differ.
- [ ] 7. Write bridges/loader.py with load_year, build_panel, main.
- [ ] 8. Run load_year(2022), sanity check, write bridges/bridges_2022.parquet.
- [ ] 9. Build and write bridges/panel.parquet with rebuilt flagged.
- [ ] 10. Commit and push the loader and both real parquet files.
