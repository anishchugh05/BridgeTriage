"""Fit the selected decline model and write annual score tables.

Run from the repository root:

    .venv/bin/python -m model.run

The model family is selected by five-fold bridge-grouped CV on 2012 -> 2016,
then refit on every eligible pair in that interval. The fitted model scores the
2018 and 2022 snapshots as of April 1 of each year.
"""

from __future__ import annotations

from pathlib import Path

from .data import load_panel
from .score import score
from .validate import FIT_END, FIT_START, select_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
SCORE_YEARS = (2018, 2022)


def run(panel_path=None, output_dir=None) -> dict:
    """Select, fit, score 2018/2022, and write both parquet outputs.

    Returns a dictionary with the selected model key, CV summary, and output
    paths so tests/callers can inspect the run programmatically.
    """
    panel = load_panel(panel_path)
    selected, chosen, summary = select_model(panel, FIT_START, FIT_END)

    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)

    paths = {}
    for year in SCORE_YEARS:
        table = panel[panel["year"] == year].copy()
        asof = f"{year}-04-01"
        scored = score(table, selected, asof=asof)
        path = destination / f"scores_{year}.parquet"
        scored.to_parquet(path, index=False)
        paths[year] = path
        print(f"wrote {path} ({len(scored)} scored bridges, as of {asof})")

    print(f"selected model: {chosen}")
    return {"model": chosen, "summary": summary, "paths": paths}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
