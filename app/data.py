# Loads the Person A panel from a relative path so it works after deploy.
import os

import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRIDGES = os.path.join(ROOT, "bridges")

YEARS = [2012, 2016, 2018, 2022]


def _panel_path():
    real = os.path.join(BRIDGES, "panel.parquet")
    if not os.path.exists(real):
        raise FileNotFoundError("The production bridge panel is unavailable.")
    return real


@st.cache_data(show_spinner=False)
def load_panel():
    df = pd.read_parquet(_panel_path())
    for c in ("lat", "lon", "adt", "worst_rating"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def latest_year(df):
    # The map and picker work on the newest year present.
    y = int(df["year"].max())
    out = df[df["year"] == y].copy()
    out = out.dropna(subset=["lat", "lon"])
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def counties(df):
    return sorted(df["county"].dropna().unique().tolist())


def history(df, structure_id):
    # Ratings across all years for one bridge, sorted by year.
    h = df[df["structure_id"] == structure_id].sort_values("year")
    return h[["year", "deck", "super", "sub", "worst_rating"]].reset_index(drop=True)
