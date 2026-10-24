# Tasks - Inspection planner front end

- [x] 1. Add streamlit and pydeck to requirements.txt if missing.
- [x] 2. app/data.py: cached load_panel, latest-year slice, history helper.
- [x] 3. app/stubs.py: score, pick, backtest matching B and C contracts.
- [x] 4. app/style.py: one inject_css block following the steering rules.
- [x] 5. app/app.py skeleton: import with stub fallback, headline and map first.
- [x] 6. Tab 1 Map: pydeck, color by score, size by adt, county filter, chosen
      in red.
- [x] 7. Tab 2 Budget picker: slider reruns pick, show chosen, cost, caught.
- [x] 8. Tab 3 Backtest: headline number and caught vs budget line chart.
- [x] 9. Tab 4 Bridge detail: ratings over years, reason, carries, crosses,
      FHWA InfoBridge link.
- [x] 10. Tab 5 Hazard flag: hazardous bridges shown first in the chosen list.
- [x] 11. Tab 6 Export: CSV download grouped by county.
- [x] 12. Tab 7 Methods: source, cost estimate note, NBI to SNBI note.
- [x] 13. Run the app, fix errors, confirm relative paths load.
- [x] 14. Commit and push to person-a-app.

## Known issue to hand to Person A

The panel has fracture_critical and underwater all False because the loader
checks for plain "Y" but NBI items 092A and 092B use the "Y" plus months form
(for example Y24, Y36). The hazard tab degrades gracefully and shows a note.
Fix belongs in bridges/loader.py (treat a value starting with Y as true).