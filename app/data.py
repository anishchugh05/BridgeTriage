"""Legacy front-end data helpers with no fabricated-data fallback.

The future composition root uses the strict boundary in :mod:`model.data`.
These helpers remain temporarily for the stale Streamlit entry point, but they
always target the repository's real ``bridges/panel.parquet`` asset.  A missing
or unreadable production panel therefore remains a production failure; this
module never checks for or substitutes ``fake.parquet``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PANEL_PATH: Final[Path] = REPOSITORY_ROOT / "bridges" / "panel.parquet"
YEARS: Final[tuple[int, ...]] = (2012, 2016, 2018, 2022)


def _panel_path() -> Path:
    """Return the sole production panel path without existence fallback."""

    return PANEL_PATH


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    """Read only the real repository panel and propagate read failures."""

    return pd.read_parquet(_panel_path())


@st.cache_data(show_spinner=False)
def latest_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Legacy current-year map projection retained until app composition lands."""

    year = int(frame["year"].max())
    latest = frame.loc[frame["year"].eq(year)].copy(deep=True)
    return latest.dropna(subset=["lat", "lon"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def counties(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["county"].dropna().unique().tolist())


def history(frame: pd.DataFrame, structure_id: str) -> pd.DataFrame:
    records = frame.loc[frame["structure_id"].eq(structure_id)].sort_values("year")
    return records.loc[
        :, ["year", "deck", "super", "sub", "worst_rating"]
    ].reset_index(drop=True)


def using_fake() -> bool:
    """Compatibility signal: automatic fake-data use has been removed."""

    return False


__all__ = [
    "PANEL_PATH",
    "REPOSITORY_ROOT",
    "YEARS",
    "counties",
    "history",
    "latest_year",
    "load_panel",
    "using_fake",
]
