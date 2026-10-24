# National Bridge Inventory, Massachusetts

Saved 2026-10-18 for Hackachusetts.

2022 FHWA delimited file for Massachusetts, reduced to the columns most useful in a spreadsheet. The live link is the full delimited file.

Rows in data.csv: 5252

Columns: structure_id, carries, crosses, county, year_built, kind, deck_condition, superstructure_condition, substructure_condition

How to use it

- Spreadsheets: open data.csv in Excel or Google Sheets.
- Fresh copy: run `python fetch.py` or `node fetch.js`. Both call the URL below and save the response.
- The preview on the Hackachusetts site is a few rows of this same table.

Direct request

https://www.fhwa.dot.gov/bridge/nbi/2022/delimited/MA22.txt

Check the source license before publishing a project.
